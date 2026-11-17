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
import kotlin.math.acos
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
    /** depths shown as the warmest and the coolest colour */
    val nearest: Double,
    val farthest: Double,
    val millis: Long,
)

/**
 * Rectify -> semi-global matching -> depth, for one stereo rig.
 * Same steps as StereoDepth in tools/mcdepth/pipeline.py. Use from a single thread.
 */
class DepthEngine(private val rig: StereoRig, private val outW: Int, private val outH: Int, sensorOrientation: Int) {
    val focal = StereoGeometry.rectifiedFocal(rig.ref, rig.aux, outW)
    private val cx = outW / 2.0
    private val cy = outH / 2.0
    private val kNew = doubleArrayOf(focal, 0.0, cx, 0.0, focal, cy, 0.0, 0.0, 1.0)
    private val refK = rig.ref.k.let { doubleArrayOf(it.fx, 0.0, it.cx, 0.0, it.fy, it.cy, 0.0, 0.0, 1.0) }
    private val nominalAuxK = rig.aux.k.let { doubleArrayOf(it.fx, 0.0, it.cx, 0.0, it.fy, it.cy, 0.0, 0.0, 1.0) }

    // ---- geometry in use: the calibration as given, or measured from the images ----
    lateinit var rect: Rectification
        private set
    var numDisparities = 16
        private set
    private var auxK = nominalAuxK
    private var correction = Mat3.identity
    private var baselineSign = 1.0 // -1: the images show the reported camera order is reversed
    private var sgbm: StereoSGBM? = null

    /** true once [refine] has succeeded */
    var aligned = false
        private set

    /** true if the geometry was measured from the images: depth is then relative, not metres */
    var selfCalibrated = false
        private set

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
    private val rectRef = Mat()
    private val rectAux = Mat()
    private val validRef = Mat()
    private val validAux = Mat()
    private var gridX = Mat()
    private var gridY = Mat()
    private var rangeLo = Double.NaN // colour range in disparity * 16, smoothed over time
    private var rangeHi = Double.NaN
    private var previous: Mat? = null // last disparity, for temporal smoothing

    init {
        reset()
    }

    /** Back to the calibration as given. */
    private fun reset() {
        correction = Mat3.identity
        baselineSign = 1.0
        selfCalibrated = false
        setGeometry(rig.r, rig.t, nominalAuxK)
    }

    private fun setGeometry(r: DoubleArray, t: DoubleArray, auxCameraMatrix: DoubleArray) {
        auxK = auxCameraMatrix
        rect = StereoGeometry.rectify(r, t)
        val n = maxOf(16, ceil((if (rect.vertical) outH else outW) / 8.0 / 16).toInt() * 16)
        if (n != numDisparities || sgbm == null) {
            numDisparities = n
            sgbm = StereoSGBM.create(
                0, n, BLOCK, 8 * BLOCK * BLOCK, 32 * BLOCK * BLOCK,
                1, 0, 10, 100, 2, StereoSGBM.MODE_SGBM_3WAY,
            )
        }
        buildMaps()
    }

    private fun buildMaps() {
        buildMap(refK, rig.ref.dist, rect.r1, kNew, refMap1, refMap2)
        buildMap(auxK, rig.aux.dist, rect.r2, Mat3.mul(correction, kNew), auxMap1, auxMap2)
        coverage(rig.ref.width, rig.ref.height, refMap1, refMap2, validRef)
        coverage(rig.aux.width, rig.aux.height, auxMap1, auxMap2, validAux)
        previous?.release()
        previous = null
        rangeLo = Double.NaN
        rangeHi = Double.NaN
    }

    private fun buildMap(k: DoubleArray, dist: DoubleArray, r: DoubleArray, newK: DoubleArray, map1: Mat, map2: Mat) {
        val km = mat3(k)
        val rm = mat3(r)
        val nm = mat3(newK)
        val dm = MatOfDouble(*dist)
        Calib3d.initUndistortRectifyMap(km, dm, rm, nm, size, CvType.CV_16SC2, map1, map2)
        listOf(km, rm, nm, dm).forEach { it.release() }
    }

    /** Which rectified pixels actually come from inside a camera's image. */
    private fun coverage(width: Int, height: Int, map1: Mat, map2: Mat, out: Mat) {
        val full = Mat(height, width, CvType.CV_8UC1, Scalar(255.0))
        val kernel = Mat.ones(5, 5, CvType.CV_8UC1)
        Imgproc.remap(full, out, map1, map2, Imgproc.INTER_NEAREST)
        Imgproc.erode(out, out, kernel)
        full.release()
        kernel.release()
    }

    private fun rectify(ref: Frame, aux: Frame) {
        val a = gray(ref)
        val b = gray(aux)
        Imgproc.remap(a, rectRef, refMap1, refMap2, Imgproc.INTER_LINEAR)
        Imgproc.remap(b, rectAux, auxMap1, auxMap2, Imgproc.INTER_LINEAR)
        a.release()
        b.release()
    }

    // ---- alignment from image content -------------------------------------------------

    private class Matches(val p1: DoubleArray, val p2: DoubleArray) {
        val size get() = p1.size / 2
    }

    private class Fine(val applied: Boolean, val matches: Int, val usable: Int, val inliers: Int, val flipped: Boolean)

    /**
     * Aligns the rectified pair using feature matches.
     *
     * 1. Fine: fit the leftover row misalignment, assuming the calibration is nearly right.
     *    Also detects a baseline that points the wrong way.
     * 2. Self-calibration, if 1 fails or only lines up a minority of the matches: the
     *    calibration is unusable, which is common on phones that publish placeholder values.
     *    Measure how the second camera is rotated against the first and in which direction
     *    parallax runs, rebuild the geometry from that, then run 1 again. Depth is then relative.
     *
     * Always restarts from the calibration as given. Returns a summary for the status display.
     */
    fun refine(ref: Frame, aux: Frame): String {
        aligned = false
        reset()
        val first = fine(ref, aux)
        if (first.applied && first.inliers >= 0.5 * first.usable) return done(first, null)
        reset()
        val detail = selfCalibrate(ref, aux)
        if (detail != null) {
            val second = fine(ref, aux)
            if (second.applied && second.inliers >= first.inliers) {
                selfCalibrated = true
                return done(second, detail)
            }
        }
        reset()
        if (first.applied) return done(fine(ref, aux), null)
        return "not aligned yet (${first.matches} matches, ${first.usable} fit the calibration), retrying"
    }

    private fun done(f: Fine, detail: String?): String {
        aligned = f.applied
        return "aligned on ${f.inliers} of ${f.usable} features" +
            (if (f.flipped) ", camera order corrected" else "") +
            (if (detail != null) "\nself-calibrated, depth is relative\n$detail" else "")
    }

    private fun match(a: Mat, b: Mat): Matches {
        val orb = ORB.create(1500)
        val k1 = MatOfKeyPoint()
        val k2 = MatOfKeyPoint()
        val d1 = Mat()
        val d2 = Mat()
        val none = Mat()
        orb.detectAndCompute(a, none, k1, d1)
        orb.detectAndCompute(b, none, k2, d2)
        val p1 = ArrayList<Double>()
        val p2 = ArrayList<Double>()
        if (!d1.empty() && !d2.empty()) {
            val matches = MatOfDMatch()
            BFMatcher.create(Core.NORM_HAMMING, true).match(d1, d2, matches)
            val ka = k1.toArray()
            val kb = k2.toArray()
            for (m in matches.toArray()) {
                if (m.distance >= 64) continue
                p1 += ka[m.queryIdx].pt.x; p1 += ka[m.queryIdx].pt.y
                p2 += kb[m.trainIdx].pt.x; p2 += kb[m.trainIdx].pt.y
            }
            matches.release()
        }
        listOf(k1, k2, d1, d2, none).forEach { it.release() }
        return Matches(p1.toDoubleArray(), p2.toDoubleArray())
    }

    /** Fits the leftover row misalignment on top of the geometry in use. */
    private fun fine(ref: Frame, aux: Frame): Fine {
        rectify(ref, aux)
        val m = match(rectRef, rectAux)
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
        val fit = StereoGeometry.fitRowAlignment(q1, q2, rect.vertical, cx, cy)
        val h = fit.homography ?: return Fine(false, m.size, usable.size, fit.inliers, false)
        correction = Mat3.mul(h, correction)
        baselineSign = s
        buildMaps()
        return Fine(true, m.size, usable.size, fit.inliers, flip)
    }

    /**
     * Rebuilds the stereo geometry from the images. Returns a description, or null if the
     * scene does not allow it (then nothing has changed).
     */
    private fun selfCalibrate(ref: Frame, aux: Frame): String? {
        // both images at the shared focal length, otherwise untouched
        val a = plain(ref, refK, rig.ref.dist)
        val b = plain(aux, nominalAuxK, rig.aux.dist)
        val m = match(a, b)
        a.release()
        b.release()
        val n = m.size
        if (n < 20) return null

        val from = MatOfPoint2f(*Array(n) { Point(m.p2[it * 2], m.p2[it * 2 + 1]) })
        val to = MatOfPoint2f(*Array(n) { Point(m.p1[it * 2], m.p1[it * 2 + 1]) })
        val inlierMat = Mat()
        val affine = Calib3d.estimateAffinePartial2D(from, to, inlierMat, Calib3d.RANSAC, 3.0)
        val sim = DoubleArray(6)
        val flags = ByteArray(n)
        val ok = !affine.empty() && inlierMat.total() == n.toLong()
        if (ok) {
            affine.get(0, 0, sim)
            inlierMat.get(0, 0, flags)
        }
        listOf(from, to, inlierMat, affine).forEach { it.release() }
        if (!ok || flags.count { it.toInt() != 0 } < 15) return null

        // as a camera rotation plus a focal-length correction, which unlike the 2D fit is
        // also right towards the image edges
        val (rough, scale) = StereoGeometry.rotationFromSimilarity(sim, focal, cx, cy)
        val rays1 = DoubleArray(n * 2) { (m.p1[it] - if (it % 2 == 0) cx else cy) / focal }
        val rays2 = DoubleArray(n * 3) {
            when (it % 3) {
                0 -> (m.p2[it / 3 * 2] - cx) / focal * scale
                1 -> (m.p2[it / 3 * 2 + 1] - cy) / focal * scale
                else -> 1.0
            }
        }
        val (rH, agree) = StereoGeometry.refineRotation(rough, rays2, rays1, BooleanArray(n) { flags[it].toInt() != 0 }, 2.0 / focal)
        val angle = Math.toDegrees(acos(((rH[0] + rH[4] + rH[8] - 1) / 2).coerceIn(-1.0, 1.0)))
        if (agree < 15 || scale <= 0.5 || scale >= 2.0 || angle > 20) return null

        fun project(rot: DoubleArray, x: Double, y: Double, z: Double): DoubleArray {
            val q = Mat3.mulVec(rot, doubleArrayOf(x, y, z))
            return doubleArrayOf(q[0] / q[2] * focal + cx, q[1] / q[2] * focal + cy)
        }
        fun ray2(i: Int) = Triple(rays2[i * 3], rays2[i * 3 + 1], rays2[i * 3 + 2])

        // direction of the baseline in the image: measured from parallax if the scene has
        // any, otherwise as reported. Its sign always comes from the report.
        val residuals = DoubleArray(n * 2)
        for (i in 0 until n) {
            val (x, y, z) = ray2(i)
            val moved = project(rH, x, y, z)
            residuals[i * 2] = m.p1[i * 2] - moved[0]
            residuals[i * 2 + 1] = m.p1[i * 2 + 1] - moved[1]
        }
        val c2 = Mat3.mulVec(Mat3.transpose(rig.r), rig.t).map { -it }
        val hintLen = hypot(c2[0], c2[1]).coerceAtLeast(1e-12)
        var dx = c2[0] / hintLen
        var dy = c2[1] / hintLen
        val measured = StereoGeometry.parallaxDirection(residuals, maxOf(outW, outH) / 5.0)
        if (measured != null) {
            val s = if (measured.x * dx + measured.y * dy < 0) -1.0 else 1.0
            dx = measured.x * s
            dy = measured.y * s
        }
        val baseline = hypot(hypot(c2[0], c2[1]), c2[2])
        val rRel = Mat3.transpose(rH)
        val t = Mat3.mulVec(rRel, doubleArrayOf(dx, dy, 0.0)).map { -baseline * it }.toDoubleArray()
        val scaledAuxK = nominalAuxK.copyOf().also {
            it[0] /= scale
            it[4] /= scale
        }
        setGeometry(rRel, t, scaledAuxK)

        // the fit put the dominant surface at zero disparity, so anything farther would be
        // negative: shift until the farthest matched points sit just above zero
        val ax = if (rect.vertical) 1 else 0
        val other = 1 - ax
        val sgn = sign(rect.baseline)
        val d = ArrayList<Double>()
        for (i in 0 until n) {
            val (x, y, z) = ray2(i)
            val r1 = project(rect.r1, rays1[i * 2], rays1[i * 2 + 1], 1.0)
            val r2 = project(rect.r2, x, y, z)
            if (abs(r1[other] - r2[other]) >= 3.0) continue
            val v = (r1[ax] - r2[ax]) * sgn
            if (abs(v) < numDisparities) d += v
        }
        if (d.size < 15) {
            reset()
            return null
        }
        d.sort()
        correction = Mat3.identity.also { it[ax * 3 + 2] = -(2.0 - d[(d.size * 0.02).toInt()]) * sgn }
        buildMaps()
        return "second camera: rotated %.1f deg, focal x%.3f\nbaseline direction in image: %.0f deg (%s)".format(
            angle, 1 / scale, Math.toDegrees(atan2(dy, dx)),
            if (measured != null) "measured, ${measured.votes} votes" else "as reported, scene too flat to measure",
        )
    }

    /** A frame resampled to the shared camera matrix, without any rotation. */
    private fun plain(frame: Frame, k: DoubleArray, dist: DoubleArray): Mat {
        val map1 = Mat()
        val map2 = Mat()
        buildMap(k, dist, Mat3.identity, kNew, map1, map2)
        val src = gray(frame)
        val out = Mat()
        Imgproc.remap(src, out, map1, map2, Imgproc.INTER_LINEAR)
        listOf(map1, map2, src).forEach { it.release() }
        return out
    }

    // ---- depth -----------------------------------------------------------------------

    fun compute(ref: Frame, aux: Frame, view: View = View.REFERENCE): DepthResult {
        val start = System.nanoTime()
        rectify(ref, aux)

        // bring everything into the form the matcher expects: horizontal baseline,
        // reference camera on the left
        val temps = ArrayList<Mat>()
        val flipped = rect.baseline * baselineSign < 0
        fun canonical(src: Mat): Mat {
            var m = src
            if (rect.vertical) m = Mat().also { Core.transpose(m, it); temps += it }
            if (flipped) m = Mat().also { Core.flip(m, it, 1); temps += it }
            return m
        }
        val left = canonical(rectRef)
        val right = canonical(rectAux)
        val validLeft = canonical(validRef)
        val validRight = canonical(validAux)

        // The matcher cannot fill the first numDisparities columns; padding moves that
        // blind strip off the image.
        val nd = numDisparities
        fun padded(m: Mat) = Mat().also {
            Core.copyMakeBorder(m, it, 0, 0, nd, 0, Core.BORDER_CONSTANT, Scalar(0.0))
            temps += it
        }
        val wide = Mat().also { temps += it }
        sgbm!!.compute(padded(left), padded(right), wide)
        val disp = wide.submat(0, left.rows(), nd, nd + left.cols()).clone() // CV_16S, disparity * 16

        // drop pixels outside the reference image and matches that land outside the second one
        if (gridX.rows() != disp.rows() || gridX.cols() != disp.cols()) {
            val w = disp.cols()
            val h = disp.rows()
            gridX = Mat(h, w, CvType.CV_32F)
            gridY = Mat(h, w, CvType.CV_32F)
            val xs = FloatArray(w) { it.toFloat() }
            for (y in 0 until h) {
                gridX.put(y, 0, xs)
                gridY.put(y, 0, FloatArray(w) { y.toFloat() })
            }
        }
        val shift = Mat().also { temps += it }
        disp.convertTo(shift, CvType.CV_32F, 1 / 16.0)
        Core.max(shift, Scalar(0.0), shift)
        Core.subtract(gridX, shift, shift)
        val landed = Mat().also { temps += it }
        Imgproc.remap(validRight, landed, shift, gridY, Imgproc.INTER_NEAREST)
        val bad = Mat().also { temps += it }
        val outside = Mat().also { temps += it }
        Core.compare(landed, Scalar(0.0), bad, Core.CMP_EQ)
        Core.compare(validLeft, Scalar(0.0), outside, Core.CMP_EQ)
        Core.bitwise_or(bad, outside, bad)
        disp.setTo(Scalar(-16.0), bad)

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
        // The colour range follows the scene (2nd to 98th percentile of disparity), eased
        // over time so it does not pump.
        val all = ShortArray(outW * outH)
        disp.get(0, 0, all)
        val sample = (all.indices step 7).map { all[it] }.filter { it > 0 }.sorted()
        if (sample.size >= 100) {
            val lo = sample[(sample.size * 0.02).toInt()].toDouble()
            val hi = maxOf(sample[(sample.size * 0.98).toInt()].toDouble(), lo + 16.0)
            rangeLo = if (rangeLo.isNaN()) lo else rangeLo * RANGE_KEEP + lo * (1 - RANGE_KEEP)
            rangeHi = if (rangeHi.isNaN()) hi else rangeHi * RANGE_KEEP + hi * (1 - RANGE_KEEP)
        }
        val lo = if (rangeLo.isNaN()) 0.0 else rangeLo
        val hi = if (rangeHi.isNaN()) numDisparities * 16.0 else rangeHi
        val disp8 = Mat()
        disp.convertTo(disp8, CvType.CV_8U, 255.0 / (hi - lo), -lo * 255.0 / (hi - lo))
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
            focal * abs(rect.baseline) * 16.0 / hi, focal * abs(rect.baseline) * 16.0 / maxOf(lo, 1.0),
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
        const val RANGE_KEEP = 0.85 // weight of the previous colour range
        const val SMOOTH_KEEP = 0.6 // weight of the previous frame
        const val SMOOTH_MAX_JUMP = 1.5 // pixels of disparity; larger changes are taken as real
    }
}
