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
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.features2d.BFMatcher
import org.opencv.features2d.ORB
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.ceil
import kotlin.math.hypot
import kotlin.math.sign

/** What the left-hand image shows; the overlay makes misalignment visible. */
enum class View(val label: String) {
    REFERENCE("reference camera"), AUX("second camera"), OVERLAY("overlay: red = reference, cyan = second");

    fun next() = entries[(ordinal + 1) % entries.size]
}

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

    /** true once [refine] has succeeded for the current calibration */
    var aligned = false
        private set

    /** true if alignment needed the coarse fallback: depth is then relative, not metres */
    var selfCalibrated = false
        private set

    private var correction = Mat3.identity
    private var baselineSign = 1.0 // -1: the images show the reported camera order is reversed
    private var coarseNote = ""
    private var lastUsable = 0
    private var previous: Mat? = null // last disparity, for temporal smoothing

    private class Matches(val p1: DoubleArray, val p2: DoubleArray) {
        val size get() = p1.size / 2
    }

    /**
     * Aligns the rectified pair using feature matches; same logic as StereoDepth.refine in
     * tools/mcdepth/pipeline.py. Always restarts from the nominal calibration.
     * Returns a one-line summary for the status display.
     */
    fun refine(ref: Frame, aux: Frame): String {
        setCorrection(Mat3.identity)
        baselineSign = 1.0
        selfCalibrated = false
        aligned = false
        previous?.release()
        previous = null

        val first = match(ref, aux)
        var note = fine(first)
        if (note == null) {
            val coarse = coarse(first)
            if (coarse != null) {
                setCorrection(coarse)
                note = fine(match(ref, aux))
                if (note != null) {
                    selfCalibrated = true
                    note += "\ncalibration far off ($coarseNote): depth is relative"
                } else {
                    setCorrection(Mat3.identity)
                }
            }
        }
        aligned = note != null
        return note ?: "not aligned yet (${first.size} matches, $lastUsable fit the calibration), retrying"
    }

    private fun setCorrection(c: DoubleArray) {
        correction = c
        buildAuxMap(c)
    }

    private fun match(ref: Frame, aux: Frame): Matches {
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
            for (m in matches.toArray()) {
                if (m.distance >= 64) continue
                p1 += a[m.queryIdx].pt.x; p1 += a[m.queryIdx].pt.y
                p2 += b[m.trainIdx].pt.x; p2 += b[m.trainIdx].pt.y
            }
            matches.release()
        }
        listOf(k1, k2, d1, d2, none).forEach { it.release() }
        return Matches(p1.toDoubleArray(), p2.toDoubleArray())
    }

    /** Fits the leftover row misalignment on top of the current correction. Null if it fails. */
    private fun fine(m: Matches): String? {
        val ax = if (rect.vertical) 1 else 0
        val other = 1 - ax
        val limit = 0.05 * (if (rect.vertical) outW else outH)
        val nominal = sign(rect.baseline)
        fun along(i: Int) = (m.p1[i * 2 + ax] - m.p2[i * 2 + ax]) * nominal
        fun across(i: Int) = abs(m.p1[i * 2 + other] - m.p2[i * 2 + other])

        // Nearly all true disparities share one sign. If it is the opposite of what the
        // calibration says, the reported camera order is the wrong way round.
        val rows = (0 until m.size).filter { across(it) < limit && abs(along(it)) < numDisparities + limit }
        val median = rows.map { along(it) }.sorted().let { if (it.isEmpty()) 0.0 else it[it.size / 2] }
        val flip = rows.size >= 30 && median < -2.0
        val s = if (flip) -1.0 else 1.0

        val usable = rows.filter { along(it) * s > -limit && along(it) * s < numDisparities + limit }
        val q1 = DoubleArray(usable.size * 2) { m.p1[usable[it / 2] * 2 + it % 2] }
        val q2 = DoubleArray(usable.size * 2) { m.p2[usable[it / 2] * 2 + it % 2] }
        lastUsable = usable.size
        val fit = StereoGeometry.fitRowAlignment(q1, q2, rect.vertical, outW / 2.0, outH / 2.0)
        val h = fit.homography ?: return null
        setCorrection(Mat3.mul(h, correction))
        baselineSign = s
        return "aligned on ${fit.inliers} features" + if (flip) ", camera order corrected" else ""
    }

    /**
     * For a calibration that is badly off: similarity transform taking aux points onto ref
     * points, with the offset along the baseline chosen so every disparity stays positive.
     */
    private fun coarse(m: Matches): DoubleArray? {
        if (m.size < 20) return null
        val from = MatOfPoint2f(*Array(m.size) { Point(m.p2[it * 2], m.p2[it * 2 + 1]) })
        val to = MatOfPoint2f(*Array(m.size) { Point(m.p1[it * 2], m.p1[it * 2 + 1]) })
        val inliers = Mat()
        val affine = Calib3d.estimateAffinePartial2D(from, to, inliers, Calib3d.RANSAC, 3.0)
        val count = if (inliers.empty()) 0 else Core.countNonZero(inliers)
        val a = DoubleArray(6)
        if (!affine.empty()) affine.get(0, 0, a)
        listOf(from, to, inliers, affine).forEach { it.release() }
        if (count < 15) return null
        val scale = hypot(a[0], a[3])
        val angle = Math.toDegrees(atan2(a[3], a[0]))
        if (scale <= 0.5 || scale >= 2.0 || abs(angle) > 15) return null

        val ax = if (rect.vertical) 1 else 0
        val other = 1 - ax
        val nominal = sign(rect.baseline)
        val d = ArrayList<Double>()
        for (i in 0 until m.size) {
            val x = m.p2[i * 2]
            val y = m.p2[i * 2 + 1]
            val moved = doubleArrayOf(a[0] * x + a[1] * y + a[2], a[3] * x + a[4] * y + a[5])
            if (abs(moved[other] - m.p1[i * 2 + other]) >= 3.0) continue
            val v = (m.p1[i * 2 + ax] - moved[ax]) * nominal
            if (abs(v) < numDisparities) d += v
        }
        if (d.size < 15) return null
        d.sort()
        val shift = 2.0 - d[(d.size * 0.02).toInt()]
        val h = doubleArrayOf(a[0], a[1], a[2], a[3], a[4], a[5], 0.0, 0.0, 1.0)
        h[ax * 3 + 2] -= shift * nominal
        coarseNote = "scale %.3f, rot %.1f deg, offset %.0f,%.0f px".format(scale, angle, a[2], a[5])
        return h
    }

    fun compute(ref: Frame, aux: Frame, view: View = View.REFERENCE): DepthResult {
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
        val flipped = rect.baseline * baselineSign < 0
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
        smooth(disp)

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
        when (view) {
            View.REFERENCE -> Imgproc.cvtColor(rectRef, refRgba, Imgproc.COLOR_GRAY2RGBA)
            View.AUX -> Imgproc.cvtColor(rectAux, refRgba, Imgproc.COLOR_GRAY2RGBA)
            View.OVERLAY -> {
                // reference in red, aux in cyan: grey where the two agree, coloured fringes where
                // they do not. Fringes should only run along the baseline direction.
                val alpha = Mat(rectRef.size(), CvType.CV_8UC1, Scalar(255.0))
                Core.merge(listOf(rectRef, rectAux, rectAux, alpha), refRgba)
                alpha.release()
            }
        }
        Imgproc.drawMarker(
            refRgba, Point(outW / 2.0, outH / 2.0), Scalar(255.0, 220.0, 0.0, 255.0),
            Imgproc.MARKER_CROSS, 20, 2,
        )
        val result = DepthResult(
            toBitmap(refRgba), toBitmap(color), centerDepth, validFraction,
            (System.nanoTime() - start) / 1_000_000,
        )
        listOf(disp, valid, disp8, color, refRgba).forEach { it.release() }
        return result
    }

    /**
     * Blends the disparity with the previous frame where the two agree, which removes
     * most frame-to-frame flicker without smearing pixels whose depth really changed.
     */
    private fun smooth(disp: Mat) {
        val cur = Mat()
        disp.convertTo(cur, CvType.CV_32F)
        val prev = previous
        if (prev != null && prev.size() == cur.size()) {
            val diff = Mat()
            val mask = Mat()
            val valid = Mat()
            Core.absdiff(cur, prev, diff)
            Core.compare(diff, Scalar(SMOOTH_MAX_JUMP * 16.0), mask, Core.CMP_LT)
            Core.compare(prev, Scalar(0.0), valid, Core.CMP_GT)
            Core.bitwise_and(mask, valid, mask)
            Core.compare(cur, Scalar(0.0), valid, Core.CMP_GT)
            Core.bitwise_and(mask, valid, mask)
            val blended = Mat()
            Core.addWeighted(cur, 1 - SMOOTH_KEEP, prev, SMOOTH_KEEP, 0.0, blended)
            blended.copyTo(cur, mask)
            cur.convertTo(disp, CvType.CV_16S)
            listOf(diff, mask, valid, blended, prev).forEach { it.release() }
        }
        previous = cur
    }

    private fun toBitmap(m: Mat): Bitmap {
        if (rotateCode >= 0) Core.rotate(m, m, rotateCode)
        return Bitmap.createBitmap(m.cols(), m.rows(), Bitmap.Config.ARGB_8888).also { Utils.matToBitmap(m, it) }
    }

    private fun gray(f: Frame) = Mat(f.height, f.width, CvType.CV_8UC1).apply { put(0, 0, f.nv21, 0, f.width * f.height) }

    private fun mat3(v: DoubleArray) = Mat(3, 3, CvType.CV_64F).apply { put(0, 0, *v) }

    private companion object {
        const val BLOCK = 5
        const val SMOOTH_KEEP = 0.6 // weight of the previous frame
        const val SMOOTH_MAX_JUMP = 1.5 // pixels of disparity; larger changes are taken as real
    }
}
