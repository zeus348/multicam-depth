package dev.multicamdepth.depth

import android.graphics.Bitmap
import dev.multicamdepth.calib.Mat3
import dev.multicamdepth.calib.Rectification
import dev.multicamdepth.calib.StereoGeometry
import dev.multicamdepth.calib.StereoRig
import dev.multicamdepth.camera.Frame
import org.opencv.android.Utils
import org.opencv.calib3d.Calib3d
import org.opencv.calib3d.StereoSGBM
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDMatch
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.features2d.BFMatcher
import org.opencv.features2d.ORB
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.sign

class DepthResult(
    val reference: Bitmap,
    val depth: Bitmap,
    /** median depth of the centre patch (metres if the rig is metric), NaN if unknown */
    val centerDepth: Double,
    val validFraction: Double,
    val millis: Long,
)

/**
 * Rectify -> semi-global matching -> depth, for one stereo rig.
 * Same steps as tools/mcdepth/pipeline.py. Use from a single thread.
 */
class DepthEngine(private val rig: StereoRig, private val outW: Int, private val outH: Int, sensorOrientation: Int) {
    val rect: Rectification = StereoGeometry.rectify(rig.r, rig.t)
    val focal = StereoGeometry.rectifiedFocal(rig.ref, rig.aux, outW)
    private val kNew = doubleArrayOf(focal, 0.0, outW / 2.0, 0.0, focal, outH / 2.0, 0.0, 0.0, 1.0)
    val numDisparities = maxOf(16, ceil((if (rect.vertical) outH else outW) / 5.0 / 16).toInt() * 16)

    private val rotateCode = when (sensorOrientation) {
        90 -> Core.ROTATE_90_CLOCKWISE
        180 -> Core.ROTATE_180
        270 -> Core.ROTATE_90_COUNTERCLOCKWISE
        else -> -1
    }
    private val size = Size(outW.toDouble(), outH.toDouble())
    private val refMap1 = Mat()
    private val refMap2 = Mat()
    private val auxMap1 = Mat()
    private val auxMap2 = Mat()
    private val sgbm = StereoSGBM.create(
        0, numDisparities, BLOCK, 8 * BLOCK * BLOCK, 32 * BLOCK * BLOCK,
        1, 0, 10, 100, 2, StereoSGBM.MODE_SGBM_3WAY,
    )
    private val rectRef = Mat()
    private val rectAux = Mat()

    init {
        buildMap(rig.ref.k.let { doubleArrayOf(it.fx, 0.0, it.cx, 0.0, it.fy, it.cy, 0.0, 0.0, 1.0) }, rig.ref.dist, rect.r1, kNew, refMap1, refMap2)
        buildAuxMap(Mat3.identity)
    }

    private fun buildAuxMap(correction: DoubleArray) {
        val k = rig.aux.k
        buildMap(
            doubleArrayOf(k.fx, 0.0, k.cx, 0.0, k.fy, k.cy, 0.0, 0.0, 1.0),
            rig.aux.dist, rect.r2, Mat3.mul(correction, kNew), auxMap1, auxMap2,
        )
    }

    private fun buildMap(k: DoubleArray, dist: DoubleArray, r: DoubleArray, newK: DoubleArray, map1: Mat, map2: Mat) {
        val km = mat3(k)
        val rm = mat3(r)
        val nm = mat3(newK)
        val dm = MatOfDouble(*dist)
        Calib3d.initUndistortRectifyMap(km, dm, rm, nm, size, CvType.CV_16SC2, map1, map2)
        listOf(km, rm, nm, dm).forEach { it.release() }
    }

    private fun rectify(ref: Frame, aux: Frame) {
        val a = gray(ref)
        val b = gray(aux)
        Imgproc.remap(a, rectRef, refMap1, refMap2, Imgproc.INTER_LINEAR)
        Imgproc.remap(b, rectAux, auxMap1, auxMap2, Imgproc.INTER_LINEAR)
        a.release()
        b.release()
    }

    /**
     * Estimates the leftover misalignment between the rectified images from feature
     * matches and folds it into the aux map. Returns a one-line summary.
     */
    fun refine(ref: Frame, aux: Frame): String {
        buildAuxMap(Mat3.identity) // always start from the nominal calibration
        rectify(ref, aux)
        val orb = ORB.create(1500)
        val k1 = MatOfKeyPoint()
        val k2 = MatOfKeyPoint()
        val d1 = Mat()
        val d2 = Mat()
        val none = Mat()
        orb.detectAndCompute(rectRef, none, k1, d1)
        orb.detectAndCompute(rectAux, none, k2, d2)
        val p1 = ArrayList<Double>()
        val p2 = ArrayList<Double>()
        if (!d1.empty() && !d2.empty()) {
            val matches = MatOfDMatch()
            BFMatcher.create(Core.NORM_HAMMING, true).match(d1, d2, matches)
            val a = k1.toArray()
            val b = k2.toArray()
            val limit = 0.05 * (if (rect.vertical) outW else outH)
            for (m in matches.toArray()) {
                if (m.distance >= 64) continue
                val pa = a[m.queryIdx].pt
                val pb = b[m.trainIdx].pt
                // keep only matches that could be a valid stereo correspondence
                val along = (if (rect.vertical) pa.y - pb.y else pa.x - pb.x) * sign(rect.baseline)
                val across = if (rect.vertical) pa.x - pb.x else pa.y - pb.y
                if (abs(across) < limit && along > -limit && along < numDisparities + limit) {
                    p1 += pa.x; p1 += pa.y
                    p2 += pb.x; p2 += pb.y
                }
            }
            matches.release()
        }
        listOf(k1, k2, d1, d2, none).forEach { it.release() }
        val fit = StereoGeometry.fitRowAlignment(p1.toDoubleArray(), p2.toDoubleArray(), rect.vertical, outW / 2.0, outH / 2.0)
        fit.homography?.let { buildAuxMap(it) }
        return if (fit.homography != null) "aligned on ${fit.inliers} features" else "alignment skipped (${fit.inliers} usable features)"
    }

    fun compute(ref: Frame, aux: Frame): DepthResult {
        val start = System.nanoTime()
        rectify(ref, aux)

        // bring the pair into the form the matcher expects: horizontal baseline, reference on the left
        var left = rectRef
        var right = rectAux
        val temps = ArrayList<Mat>()
        fun both(op: (Mat, Mat) -> Unit) {
            val l = Mat().also { op(left, it) }
            val r = Mat().also { op(right, it) }
            temps += l; temps += r
            left = l; right = r
        }
        if (rect.vertical) both { s, d -> Core.transpose(s, d) }
        val flipped = rect.baseline < 0
        if (flipped) both { s, d -> Core.flip(s, d, 1) }

        val disp = Mat() // CV_16S, disparity * 16
        sgbm.compute(left, right, disp)
        if (flipped) Core.flip(disp, disp, 1)
        if (rect.vertical) {
            val t = Mat()
            Core.transpose(disp, t)
            disp.release()
            t.copyTo(disp)
            t.release()
        }
        temps.forEach { it.release() }

        // statistics
        val valid = Mat()
        Core.compare(disp, Scalar(0.0), valid, Core.CMP_GT)
        val validFraction = Core.countNonZero(valid).toDouble() / (outW * outH)
        val pw = outW / 10
        val ph = outH / 10
        val patch = disp.submat(outH / 2 - ph / 2, outH / 2 + ph / 2, outW / 2 - pw / 2, outW / 2 + pw / 2).clone()
        val values = ShortArray(patch.rows() * patch.cols())
        patch.get(0, 0, values)
        patch.release()
        val good = values.filter { it > 0 }.sorted()
        val centerDepth = if (good.size > values.size / 4) {
            focal * abs(rect.baseline) * 16.0 / good[good.size / 2]
        } else {
            Double.NaN
        }

        // colour: near = warm, far = cool, unknown = black
        val disp8 = Mat()
        disp.convertTo(disp8, CvType.CV_8U, 255.0 / (numDisparities * 16.0))
        val color = Mat()
        Imgproc.applyColorMap(disp8, color, Imgproc.COLORMAP_TURBO)
        Imgproc.cvtColor(color, color, Imgproc.COLOR_BGR2RGBA)
        Core.bitwise_not(valid, valid)
        color.setTo(Scalar(0.0, 0.0, 0.0, 255.0), valid)
        val refRgba = Mat()
        Imgproc.cvtColor(rectRef, refRgba, Imgproc.COLOR_GRAY2RGBA)
        Imgproc.drawMarker(
            refRgba, org.opencv.core.Point(outW / 2.0, outH / 2.0), Scalar(255.0, 220.0, 0.0, 255.0),
            Imgproc.MARKER_CROSS, 20, 2,
        )
        val result = DepthResult(
            toBitmap(refRgba), toBitmap(color), centerDepth, validFraction,
            (System.nanoTime() - start) / 1_000_000,
        )
        listOf(disp, valid, disp8, color, refRgba).forEach { it.release() }
        return result
    }

    private fun toBitmap(m: Mat): Bitmap {
        if (rotateCode >= 0) Core.rotate(m, m, rotateCode)
        return Bitmap.createBitmap(m.cols(), m.rows(), Bitmap.Config.ARGB_8888).also { Utils.matToBitmap(m, it) }
    }

    private fun gray(f: Frame) = Mat(f.height, f.width, CvType.CV_8UC1).apply { put(0, 0, f.nv21, 0, f.width * f.height) }

    private fun mat3(v: DoubleArray) = Mat(3, 3, CvType.CV_64F).apply { put(0, 0, *v) }

    private companion object {
        const val BLOCK = 5
    }
}
