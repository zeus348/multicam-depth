package dev.multicamdepth

import dev.multicamdepth.calib.AssumedBaseline
import dev.multicamdepth.calib.CalibrationBuilder
import dev.multicamdepth.calib.Intrinsics
import dev.multicamdepth.calib.Mat3
import dev.multicamdepth.calib.PhysicalCamera
import dev.multicamdepth.calib.StereoGeometry
import dev.multicamdepth.camera.FramePairer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Random
import kotlin.math.cos
import kotlin.math.sin

class GeometryTest {
    private fun rotX(a: Double) = doubleArrayOf(1.0, 0.0, 0.0, 0.0, cos(a), -sin(a), 0.0, sin(a), cos(a))
    private fun rotZ(a: Double) = doubleArrayOf(cos(a), -sin(a), 0.0, sin(a), cos(a), 0.0, 0.0, 0.0, 1.0)

    /** r, t for an aux camera at position c2 (reference coordinates) with orientation r. */
    private fun pose(r: DoubleArray, c2: DoubleArray) = r to Mat3.mulVec(r, c2).map { -it }.toDoubleArray()

    @Test
    fun rectifiedPointsShareRowAndDepthAndDifferByBaseline() {
        val (r, t) = pose(Mat3.mul(rotX(0.02), rotZ(-0.03)), doubleArrayOf(-0.012, 0.002, 0.001))
        val rect = StereoGeometry.rectify(r, t)
        assertFalse(rect.vertical)
        assertTrue(rect.baseline < 0) // aux is to the left of the reference
        val rng = Random(1)
        repeat(20) {
            val p = doubleArrayOf(rng.nextDouble() - 0.5, rng.nextDouble() - 0.5, 0.3 + rng.nextDouble() * 3)
            val a = Mat3.mulVec(rect.r1, p)
            val inAux = Mat3.mulVec(r, p).also { for (i in 0..2) it[i] += t[i] }
            val b = Mat3.mulVec(rect.r2, inAux)
            assertEquals(a[1], b[1], 1e-12)
            assertEquals(a[2], b[2], 1e-12)
            assertEquals(rect.baseline, a[0] - b[0], 1e-12)
        }
    }

    @Test
    fun verticalRigIsDetected() {
        val (r, t) = pose(Mat3.identity, doubleArrayOf(0.001, 0.011, 0.0))
        val rect = StereoGeometry.rectify(r, t)
        assertTrue(rect.vertical)
        assertTrue(rect.baseline > 0)
    }

    @Test
    fun relativePoseMatchesPythonReference() {
        // same case as tools/tests/test_pipeline.py::test_relative_pose_from_android_quaternions
        val q = floatArrayOf(1f, 0f, 0f, 0f)
        val (r, t) = StereoGeometry.relativePose(q, floatArrayOf(0f, 0f, 0f), q, floatArrayOf(0.01f, 0f, 0f))
        for (i in 0..8) assertEquals(Mat3.identity[i], r[i], 1e-9)
        assertEquals(-0.01, t[0], 1e-9)
        assertEquals(0.0, t[1], 1e-9)
    }

    @Test
    fun intrinsicsFollowCentreCrop() {
        val k = StereoGeometry.scaleIntrinsics(Intrinsics(3000.0, 3000.0, 2000.0, 1500.0), 4000, 3000, 1280, 720)
        assertEquals(960.0, k.fx, 1e-9)
        assertEquals(640.0, k.cx, 1e-9)
        assertEquals(360.0, k.cy, 1e-9)
    }

    @Test
    fun rowAlignmentRecoversKnownCorrectionDespiteOutliers() {
        val rng = Random(3)
        val n = 200
        val p1 = DoubleArray(n * 2)
        val p2 = DoubleArray(n * 2)
        val a = 0.004
        val b = 1.01
        val c = -3.0
        for (i in 0 until n) {
            val u = rng.nextDouble() * 640
            val v = rng.nextDouble() * 480
            p2[i * 2] = u
            p2[i * 2 + 1] = v
            p1[i * 2] = u + rng.nextDouble() * 40
            p1[i * 2 + 1] = if (i % 5 == 0) rng.nextDouble() * 480 else a * u + b * v + c + rng.nextGaussian() * 0.2
        }
        val fit = StereoGeometry.fitRowAlignment(p1, p2, false, 320.0, 240.0)
        val h = fit.homography
        assertNotNull(h)
        assertTrue(fit.inliers > 140)
        assertEquals(a, h!![3], 2e-4)
        assertEquals(b, h[4], 2e-4)
        assertEquals(c, h[5], 0.1)
    }

    @Test
    fun rowAlignmentRefusesTooFewPoints() {
        assertNull(StereoGeometry.fitRowAlignment(DoubleArray(10), DoubleArray(10), false, 0.0, 0.0).homography)
    }

    private fun camera(id: String, focal: Float, pose: Boolean, x: Float = 0f) = PhysicalCamera(
        id, focal, 6.4f, 4.8f, 4000, 3000, null, null,
        if (pose) floatArrayOf(1f, 0f, 0f, 0f) else null,
        if (pose) floatArrayOf(x, 0f, 0f) else null,
        listOf(1280 to 960),
    )

    @Test
    fun calibrationFallsBackWhenPhonePublishesNothing() {
        val rig = CalibrationBuilder.build(camera("2", 6.0f, false), camera("3", 2.2f, false), 1280, 960, true, AssumedBaseline.MINUS_Y)
        assertFalse(rig.metric)
        assertEquals(6.0 / 6.4 * 1280, rig.ref.k.fx, 1e-3)
        val rect = StereoGeometry.rectify(rig.r, rig.t)
        assertTrue(rect.vertical)
        assertEquals(-CalibrationBuilder.ASSUMED_BASELINE_M, rect.baseline, 1e-12)
    }

    @Test
    fun calibrationUsesPublishedPose() {
        val rig = CalibrationBuilder.build(camera("2", 6.0f, true), camera("3", 2.2f, true, 0.01f), 1280, 960, true, AssumedBaseline.PLUS_X)
        assertTrue(rig.metric)
        assertEquals(0.01, StereoGeometry.rectify(rig.r, rig.t).baseline, 1e-7)
    }

    @Test
    fun pairerMatchesNearestTimestampsAndDropsOrphans() {
        val p = FramePairer<Long>(10) { it }
        assertNull(p.addRef(100))
        assertEquals(100L to 103L, p.addAux(103))
        assertNull(p.addRef(200)) // its partner never arrives
        assertNull(p.addRef(300))
        assertEquals(300L to 298L, p.addAux(298))
        assertEquals(1, p.dropped)
    }

    @Test
    fun pairerPrefersTheCloserOfTwoCandidates() {
        val p = FramePairer<Long>(10) { it }
        assertNull(p.addRef(100))
        assertNull(p.addRef(108))
        assertEquals(108L to 109L, p.addAux(109))
    }
}
