package dev.multicamdepth.calib

import java.util.Random
import kotlin.math.abs
import kotlin.math.max
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
