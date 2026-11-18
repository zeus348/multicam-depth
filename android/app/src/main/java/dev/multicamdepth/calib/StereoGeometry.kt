package dev.multicamdepth.calib

import java.util.Random
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.asin
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Pure stereo geometry, no Android or OpenCV types so it runs in plain JVM tests.
 * Mirrors tools/mcdepth/geometry.py.
 *
 * Matrices are row-major DoubleArray(9), vectors DoubleArray(3).
 * Camera frame: x right, y down, z forward. Relative pose: X_aux = R * X_ref + T.
 */
class Intrinsics(val fx: Double, val fy: Double, val cx: Double, val cy: Double)

class CameraModel(
    val id: String,
    val width: Int,
    val height: Int,
    val k: Intrinsics,
    /** OpenCV order: k1 k2 p1 p2 k3 */
    val dist: DoubleArray = DoubleArray(5),
)

class StereoRig(
    val ref: CameraModel,
    val aux: CameraModel,
    val r: DoubleArray,
    val t: DoubleArray,
    /** false when the baseline was assumed: depth is then relative, not metres */
    val metric: Boolean,
    /** human-readable note on where each part of the calibration came from */
    val source: String,
)

class Rectification(
    /** reference camera -> rectified frame */
    val r1: DoubleArray,
    /** aux camera -> rectified frame */
    val r2: DoubleArray,
    /** signed position of the aux camera along the baseline axis, same unit as T */
    val baseline: Double,
    /** true: baseline along image y, epipolar lines are columns */
    val vertical: Boolean,
)

object Mat3 {
    val identity get() = doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    fun mul(a: DoubleArray, b: DoubleArray) = DoubleArray(9) { i ->
        val r = i / 3
        val c = i % 3
        a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c]
    }

    fun transpose(a: DoubleArray) = DoubleArray(9) { i -> a[(i % 3) * 3 + i / 3] }

    fun mulVec(a: DoubleArray, v: DoubleArray) = DoubleArray(3) { r ->
        a[r * 3] * v[0] + a[r * 3 + 1] * v[1] + a[r * 3 + 2] * v[2]
    }

    fun det(m: DoubleArray) =
        m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6])

    /** Solves m * x = b, or null if m is singular. */
    fun solve(m: DoubleArray, b: DoubleArray): DoubleArray? {
        val d = det(m)
        if (abs(d) < 1e-12) return null
        return DoubleArray(3) { col ->
            val mc = m.copyOf()
            for (r in 0..2) mc[r * 3 + col] = b[r]
            det(mc) / d
        }
    }
}

object StereoGeometry {

    /** Android LENS_POSE_ROTATION (x, y, z, w): sensor coordinates -> camera coordinates. */
    fun quatToRot(q: FloatArray): DoubleArray {
        val n = sqrt((q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).toDouble())
        val x = q[0] / n
        val y = q[1] / n
        val z = q[2] / n
        val w = q[3] / n
        return doubleArrayOf(
            1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w,
            2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w,
            2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y,
        )
    }

    /**
     * Android defines p_cam_i = R_i * (p_sensor - t_i) for each camera.
     * Returns (R, T) with X_2 = R * X_1 + T.
     */
    fun relativePose(q1: FloatArray, t1: FloatArray, q2: FloatArray, t2: FloatArray): Pair<DoubleArray, DoubleArray> {
        val r1 = quatToRot(q1)
        val r2 = quatToRot(q2)
        val d = DoubleArray(3) { (t1[it] - t2[it]).toDouble() }
        return Mat3.mul(r2, Mat3.transpose(r1)) to Mat3.mulVec(r2, d)
    }

    /** Active-array pixels -> stream pixels (centre crop to the stream aspect, then scale). */
    fun scaleIntrinsics(k: Intrinsics, arrayW: Int, arrayH: Int, outW: Int, outH: Int): Intrinsics {
        val s = max(outW.toDouble() / arrayW, outH.toDouble() / arrayH)
        return Intrinsics(
            k.fx * s,
            k.fy * s,
            (k.cx - arrayW / 2.0) * s + outW / 2.0,
            (k.cy - arrayH / 2.0) * s + outH / 2.0,
        )
    }

    fun rectify(r: DoubleArray, t: DoubleArray): Rectification {
        val rt = Mat3.transpose(r)
        val c2 = Mat3.mulVec(rt, t).map { -it }.toDoubleArray() // aux centre in reference coordinates
        val norm = sqrt(c2[0] * c2[0] + c2[1] * c2[1] + c2[2] * c2[2])
        require(norm > 1e-9) { "cameras are at the same position; no stereo baseline" }
        val vertical = abs(c2[1]) > abs(c2[0])
        val k = if (vertical) 1 else 0
        val sign = if (c2[k] >= 0) 1.0 else -1.0
        val e = DoubleArray(3) { c2[it] / norm * sign }

        val zMean = doubleArrayOf(r[6], r[7], 1.0 + r[8])
        val dot = zMean[0] * e[0] + zMean[1] * e[1] + zMean[2] * e[2]
        val z = normalize(DoubleArray(3) { zMean[it] - dot * e[it] })
        val x: DoubleArray
        val y: DoubleArray
        if (vertical) {
            y = e
            x = cross(y, z)
        } else {
            x = e
            y = cross(z, x)
        }
        val rRect = doubleArrayOf(x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1], z[2])
        return Rectification(rRect, Mat3.mul(rRect, rt), sign * norm, vertical)
    }

    /** Shared focal length of the rectified pair: the narrower camera's, since only its view overlaps. */
    fun rectifiedFocal(ref: CameraModel, aux: CameraModel, outW: Int): Double =
        max(ref.k.fx * outW / ref.width, aux.k.fx * outW / aux.width)

    class Alignment(val homography: DoubleArray?, val inliers: Int)

    /**
     * Residual misalignment from matched points (x, y pairs flattened) in the rectified images.
     * Fits v1 = a*u2 + b*v2 + c for the coordinate v that should agree in both images and
     * returns the homography to apply to image 2. homography is null if the fit is not trustworthy.
     */
    fun fitRowAlignment(
        p1: DoubleArray,
        p2: DoubleArray,
        vertical: Boolean,
        centerX: Double,
        centerY: Double,
        iterations: Int = 300,
        threshold: Double = 1.0,
    ): Alignment {
        val n = p1.size / 2
        if (n < MIN_INLIERS) return Alignment(null, n)
        val iu = if (vertical) 1 else 0 // index of the along-baseline coordinate
        val iv = 1 - iu
        val cu = if (vertical) centerY else centerX
        val cv = if (vertical) centerX else centerY
        val u = DoubleArray(n) { p2[it * 2 + iu] - cu }
        val v = DoubleArray(n) { p2[it * 2 + iv] - cv }
        val target = DoubleArray(n) { p1[it * 2 + iv] - cv }
        fun residual(s: DoubleArray, i: Int) = abs(s[0] * u[i] + s[1] * v[i] + s[2] - target[i])

        val rng = Random(7)
        var best: DoubleArray? = null
        var bestCount = 0
        repeat(iterations) {
            val i = rng.nextInt(n)
            val j = rng.nextInt(n)
            val k = rng.nextInt(n)
            if (i == j || j == k || i == k) return@repeat
            val sol = Mat3.solve(
                doubleArrayOf(u[i], v[i], 1.0, u[j], v[j], 1.0, u[k], v[k], 1.0),
                doubleArrayOf(target[i], target[j], target[k]),
            ) ?: return@repeat
            val count = (0 until n).count { residual(sol, it) < threshold }
            if (count > bestCount) {
                bestCount = count
                best = sol
            }
        }
        var sol = best ?: return Alignment(null, 0)
        if (bestCount < MIN_INLIERS) return Alignment(null, bestCount)
        var count = bestCount
        repeat(2) { // least squares on the inliers
            val ata = DoubleArray(9)
            val atb = DoubleArray(3)
            for (i in 0 until n) {
                if (residual(sol, i) >= threshold) continue
                val row = doubleArrayOf(u[i], v[i], 1.0)
                for (a in 0..2) {
                    atb[a] += row[a] * target[i]
                    for (b in 0..2) ata[a * 3 + b] += row[a] * row[b]
                }
            }
            sol = Mat3.solve(ata, atb) ?: return Alignment(null, count)
            count = (0 until n).count { residual(sol, it) < threshold }
            if (count < MIN_INLIERS) return Alignment(null, count)
        }
        val (a, b, c) = sol
        if (abs(a) > 0.1 || abs(b - 1) > 0.1) return Alignment(null, count)
        val cAbs = c + cv - a * cu - b * cv
        val h = if (vertical) {
            doubleArrayOf(b, a, cAbs, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        } else {
            doubleArrayOf(1.0, 0.0, 0.0, a, b, cAbs, 0.0, 0.0, 1.0)
        }
        return Alignment(h, count)
    }

    private const val MIN_INLIERS = 12

    /**
     * Device ("sensor") coordinates -> camera coordinates, derived from SENSOR_ORIENTATION alone.
     * Used when a phone publishes a placeholder LENS_POSE_ROTATION that does not match how its
     * images are oriented. Device axes: x right, y up, z out of the screen.
     */
    fun orientationRotation(sensorOrientation: Int, frontFacing: Boolean): DoubleArray {
        val a = Math.toRadians(sensorOrientation.toDouble())
        val c = cos(a)
        val s = sin(a)
        val rx = if (frontFacing) -1.0 else 1.0 // scene-right along device x
        return doubleArrayOf(c * rx, -s, 0.0, -s * rx, -c, 0.0, 0.0, 0.0, if (frontFacing) 1.0 else -1.0)
    }

    /** Angle between two rotations, degrees. */
    fun angleBetween(a: DoubleArray, b: DoubleArray): Double {
        val d = Mat3.mul(a, Mat3.transpose(b))
        return Math.toDegrees(acos(((d[0] + d[4] + d[8] - 1) / 2).coerceIn(-1.0, 1.0)))
    }

    class Direction(val x: Double, val y: Double, val votes: Int)

    /**
     * Direction along which matched points are displaced by depth. After the two images are
     * aligned on the dominant surface, every remaining displacement is parallax, which runs
     * along the baseline. [residuals]: displacement vectors, x y flattened.
     * Null when there is no clear common direction.
     */
    fun parallaxDirection(
        residuals: DoubleArray,
        maxLen: Double,
        minLen: Double = 4.0,
        tol: Double = 1.5,
        maxCandidates: Int = 200,
    ): Direction? {
        val xs = ArrayList<Double>()
        val ys = ArrayList<Double>()
        for (i in 0 until residuals.size / 2) {
            val len = hypot(residuals[i * 2], residuals[i * 2 + 1])
            if (len > minLen && len < maxLen) {
                xs += residuals[i * 2]
                ys += residuals[i * 2 + 1]
            }
        }
        val n = xs.size
        if (n < MIN_INLIERS) return null
        fun onLine(i: Int, ux: Double, uy: Double) = abs(xs[i] * uy - ys[i] * ux) < tol
        var bestCount = 0
        var bx = 0.0
        var by = 0.0
        for (c in 0 until n step max(1, n / maxCandidates)) {
            val len = hypot(xs[c], ys[c])
            val ux = xs[c] / len
            val uy = ys[c] / len
            val count = (0 until n).count { onLine(it, ux, uy) }
            if (count > bestCount) {
                bestCount = count
                bx = ux
                by = uy
            }
        }
        if (bestCount < MIN_INLIERS || bestCount < 0.5 * n) return null
        var sxx = 0.0
        var sxy = 0.0
        var syy = 0.0
        for (i in 0 until n) {
            if (!onLine(i, bx, by)) continue
            sxx += xs[i] * xs[i]
            sxy += xs[i] * ys[i]
            syy += ys[i] * ys[i]
        }
        val angle = 0.5 * atan2(2 * sxy, sxx - syy) // principal axis of the scatter
        return Direction(cos(angle), sin(angle), bestCount)
    }

    /**
     * How image points move when the phone turns, as a homography taking a pixel of the
     * earlier frame to the same scene point in the later one (rotation only; depth is not
     * needed for that).
     *
     * @param deviceRotation gyro rotation R with p_earlier = R * p_later (device frame)
     * @param deviceToCamera device axes -> camera axes, see [orientationRotation]
     * @param rect camera -> rectified view (identity for an unrectified image)
     */
    fun motionHomography(
        deviceRotation: DoubleArray,
        deviceToCamera: DoubleArray,
        rect: DoubleArray,
        f: Double,
        cx: Double,
        cy: Double,
    ): DoubleArray {
        val toView = Mat3.mul(rect, deviceToCamera)
        val m = Mat3.mul(Mat3.mul(toView, Mat3.transpose(deviceRotation)), Mat3.transpose(toView))
        val k = doubleArrayOf(f, 0.0, cx, 0.0, f, cy, 0.0, 0.0, 1.0)
        val kInv = doubleArrayOf(1 / f, 0.0, -cx / f, 0.0, 1 / f, -cy / f, 0.0, 0.0, 1.0)
        return Mat3.mul(Mat3.mul(k, m), kInv)
    }

    /** Content at (x, y) in the left image sits at (x, y + a + b*x + c*y) in the right. */
    class RowShift(val a: Double, val b: Double, val c: Double, val blocks: Int)

    /**
     * Frame-by-frame misalignment between two rectified images, across the baseline.
     *
     * While a phone moves, its cameras stop lining up by a few pixels: the sensors are not
     * exposed at exactly the same instant, their rolling shutters differ, and optical
     * stabilisation and autofocus shift one lens but not the other. Stereo matching needs
     * rows to agree to about a pixel, so this is measured for every frame.
     *
     * Inputs are row-mean profiles: for each of several vertical bands of the image (left
     * to right), the mean of every row. Averaging along the baseline makes a profile
     * insensitive to depth, so the shift between the two cameras' profiles is pure
     * misalignment. It is measured for a grid of blocks and a plane is fitted.
     * Mirrors row_shift_plane in tools/mcdepth/geometry.py.
     */
    fun rowShiftPlane(
        profilesLeft: Array<FloatArray>,
        profilesRight: Array<FloatArray>,
        width: Int,
        maxShift: Int,
        segments: Int = 3,
        minScore: Double = 0.4,
    ): RowShift {
        val bands = profilesLeft.size
        val xs = ArrayList<Double>()
        val ys = ArrayList<Double>()
        val shifts = ArrayList<Double>()
        for (band in 0 until bands) {
            val l = profilesLeft[band]
            val r = profilesRight[band]
            val n = l.size - 2
            // derivative: ignores brightness differences between the cameras
            val dl = DoubleArray(n) { (l[it + 2] - l[it]).toDouble() }
            val dr = DoubleArray(n) { (r[it + 2] - r[it]).toDouble() }
            for (seg in 0 until segments) {
                val y0 = max(maxShift, seg * n / segments)
                val y1 = min(n - maxShift, (seg + 1) * n / segments)
                if (y1 - y0 < 16) continue
                var energy = 0.0
                for (y in y0 until y1) energy += dl[y] * dl[y]
                if (energy / (y1 - y0) < 0.05) continue // no texture across the baseline here
                val scores = DoubleArray(2 * maxShift + 1) { i ->
                    val sft = i - maxShift
                    var dot = 0.0
                    var other = 0.0
                    for (y in y0 until y1) {
                        dot += dl[y] * dr[y + sft]
                        other += dr[y + sft] * dr[y + sft]
                    }
                    dot / sqrt(energy * other + 1e-9)
                }
                val k = scores.indices.maxByOrNull { scores[it] }!!
                // weak, or the true shift is outside the search range
                if (scores[k] < minScore || k == 0 || k == 2 * maxShift) continue
                val denom = scores[k - 1] - 2 * scores[k] + scores[k + 1]
                val sub = if (abs(denom) > 1e-9) 0.5 * (scores[k - 1] - scores[k + 1]) / denom else 0.0
                xs += (band + 0.5) * width / bands
                ys += (y0 + y1) / 2.0 + 1
                shifts += k - maxShift + sub
            }
        }
        val n = shifts.size
        if (n == 0) return RowShift(0.0, 0.0, 0.0, 0)
        val constant = RowShift(shifts.sorted()[n / 2], 0.0, 0.0, n)
        if (n < 5) return constant
        var keep = BooleanArray(n) { true }
        var sol = doubleArrayOf(0.0, 0.0, 0.0)
        repeat(2) {
            val ata = DoubleArray(9)
            val atb = DoubleArray(3)
            for (i in 0 until n) {
                if (!keep[i]) continue
                val row = doubleArrayOf(1.0, xs[i], ys[i])
                for (p in 0..2) {
                    atb[p] += row[p] * shifts[i]
                    for (q in 0..2) ata[p * 3 + q] += row[p] * row[q]
                }
            }
            sol = Mat3.solve(ata, atb) ?: return constant
            keep = BooleanArray(n) { abs(sol[0] + sol[1] * xs[it] + sol[2] * ys[it] - shifts[it]) < 1.5 }
            if (keep.count { it } < 5) return constant
        }
        if (abs(sol[1]) > 0.05 || abs(sol[2]) > 0.05) return constant
        return RowShift(sol[0], sol[1], sol[2], keep.count { it })
    }

    class Line(val slope: Double, val offset: Double, val inliers: Int)

    /**
     * Robust y = slope * x + offset. Null if there are too few points or x does not vary
     * enough to pin down the slope.
     */
    fun fitLine(x: FloatArray, y: FloatArray, tol: Double = 0.75, iterations: Int = 200): Line? {
        val n = x.size
        if (n < 50) return null
        val sorted = x.sortedArray()
        val spread = sorted[(n * 0.95).toInt()] - sorted[(n * 0.05).toInt()]
        if (spread < 1.0) return null
        val rng = Random(3)
        var bestCount = 0
        var ba = 0.0
        var bb = 0.0
        repeat(iterations) {
            val i = rng.nextInt(n)
            val j = rng.nextInt(n)
            if (abs(x[i] - x[j]) < 0.5 * spread) return@repeat
            val a = (y[i] - y[j]).toDouble() / (x[i] - x[j])
            val b = y[i] - a * x[i]
            val count = (0 until n).count { abs(a * x[it] + b - y[it]) < tol }
            if (count > bestCount) {
                bestCount = count
                ba = a
                bb = b
            }
        }
        if (bestCount < 50) return null
        // least squares on the inliers
        var sx = 0.0
        var sy = 0.0
        var sxx = 0.0
        var sxy = 0.0
        var m = 0
        for (i in 0 until n) {
            if (abs(ba * x[i] + bb - y[i]) >= tol) continue
            sx += x[i]; sy += y[i]; sxx += x[i].toDouble() * x[i]; sxy += x[i].toDouble() * y[i]; m++
        }
        val den = m * sxx - sx * sx
        if (abs(den) < 1e-9) return null
        val a = (m * sxy - sx * sy) / den
        val b = (sy - a * sx) / m
        return Line(a, b, (0 until n).count { abs(a * x[it] + b - y[it]) < tol })
    }

    /** Rotation vector (axis * angle, radians) -> rotation matrix. */
    fun rodrigues(w: DoubleArray): DoubleArray {
        val angle = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2])
        if (angle < 1e-12) return Mat3.identity
        val k = DoubleArray(3) { w[it] / angle }
        val kx = doubleArrayOf(0.0, -k[2], k[1], k[2], 0.0, -k[0], -k[1], k[0], 0.0)
        val kx2 = Mat3.mul(kx, kx)
        val id = Mat3.identity
        return DoubleArray(9) { id[it] + sin(angle) * kx[it] + (1 - cos(angle)) * kx2[it] }
    }

    /**
     * Camera rotation that explains a measured image-to-image similarity [m] (2x3, row-major)
     * taking aux pixels onto ref pixels, both at focal [f] and principal point [cx], [cy].
     * Returns (R, scale): R rotates aux rays into the ref frame.
     */
    fun rotationFromSimilarity(m: DoubleArray, f: Double, cx: Double, cy: Double): Pair<DoubleArray, Double> {
        val scale = hypot(m[0], m[3])
        val roll = atan2(m[3], m[0])
        val qx = m[0] * cx + m[1] * cy + m[2]
        val qy = m[3] * cx + m[4] * cy + m[5]
        val v = normalize(doubleArrayOf((qx - cx) / f, (qy - cy) / f, 1.0))
        // smallest rotation taking the optical axis onto v: about (z cross v) by asin(|z cross v|)
        val sinA = hypot(v[0], v[1])
        val point = if (sinA < 1e-12) Mat3.identity else {
            val a = asin(sinA.coerceAtMost(1.0))
            rodrigues(doubleArrayOf(-v[1] / sinA * a, v[0] / sinA * a, 0.0))
        }
        val rz = doubleArrayOf(cos(roll), -sin(roll), 0.0, sin(roll), cos(roll), 0.0, 0.0, 0.0, 1.0)
        return Mat3.mul(point, rz) to scale
    }

    /**
     * Least-squares polish of a rotation [r] taking aux rays (x y z flattened) onto ref points
     * ([target], normalised image coordinates, x y flattened). [mask] says which pairs to trust
     * in the first round; later rounds use every pair that agrees within [tol].
     * Returns (rotation, number of agreeing pairs).
     */
    fun refineRotation(
        r: DoubleArray,
        raysAux: DoubleArray,
        target: DoubleArray,
        mask: BooleanArray,
        tol: Double,
        iterations: Int = 3,
    ): Pair<DoubleArray, Int> {
        val n = target.size / 2
        var rot = r
        val xy = DoubleArray(n * 2)
        fun project() {
            for (i in 0 until n) {
                val q = Mat3.mulVec(rot, doubleArrayOf(raysAux[i * 3], raysAux[i * 3 + 1], raysAux[i * 3 + 2]))
                xy[i * 2] = q[0] / q[2]
                xy[i * 2 + 1] = q[1] / q[2]
            }
        }
        fun agrees(i: Int) = hypot(target[i * 2] - xy[i * 2], target[i * 2 + 1] - xy[i * 2 + 1]) < tol
        var use = mask
        for (round in 0 until iterations) {
            project()
            if (round > 0) use = BooleanArray(n) { agrees(it) }
            if (use.count { it } < MIN_INLIERS) break
            val ata = DoubleArray(9)
            val atb = DoubleArray(3)
            for (i in 0 until n) {
                if (!use[i]) continue
                val x = xy[i * 2]
                val y = xy[i * 2 + 1]
                // effect of a small extra rotation on the projected point
                val rows = arrayOf(
                    doubleArrayOf(-x * y, 1 + x * x, -y) to target[i * 2] - x,
                    doubleArrayOf(-(1 + y * y), x * y, x) to target[i * 2 + 1] - y,
                )
                for ((j, d) in rows) for (a in 0..2) {
                    atb[a] += j[a] * d
                    for (b in 0..2) ata[a * 3 + b] += j[a] * j[b]
                }
            }
            val w = Mat3.solve(ata, atb) ?: break
            rot = Mat3.mul(rodrigues(w), rot)
        }
        project()
        return rot to (0 until n).count { agrees(it) }
    }

    private fun normalize(v: DoubleArray): DoubleArray {
        val n = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        return DoubleArray(3) { v[it] / n }
    }

    private fun cross(a: DoubleArray, b: DoubleArray) = doubleArrayOf(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )
}
