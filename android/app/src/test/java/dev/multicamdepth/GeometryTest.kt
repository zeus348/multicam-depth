package dev.multicamdepth

import dev.multicamdepth.calib.CalibrationBuilder
import dev.multicamdepth.calib.Intrinsics
import dev.multicamdepth.calib.Mat3
import dev.multicamdepth.calib.PhysicalCamera
import dev.multicamdepth.calib.StereoGeometry
import dev.multicamdepth.camera.FrameSync
import dev.multicamdepth.fusion.GyroIntegrator
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

    private fun camera(id: String, focal: Float, q: FloatArray?, t: FloatArray?) =
        PhysicalCamera(id, focal, 6.4f, 4.8f, 4000, 3000, null, null, q, t, listOf(1280 to 960))

    /** Spec-conforming rotation for a back camera whose sensor is mounted at 90 degrees. */
    private val portraitBack = floatArrayOf(0.70710677f, -0.70710677f, 0f, 0f)

    @Test
    fun orientationRotationIsWhatAConformingPhoneReports() {
        val expected = StereoGeometry.orientationRotation(90, false)
        val want = doubleArrayOf(0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0)
        for (i in 0..8) assertEquals(want[i], expected[i], 1e-12)
        assertEquals(0.0, StereoGeometry.angleBetween(StereoGeometry.quatToRot(portraitBack), expected), 1e-3)
        assertEquals(1.0, Mat3.det(StereoGeometry.orientationRotation(270, true)), 1e-12)
    }

    @Test
    fun calibrationFallsBackWhenPhonePublishesNothing() {
        val rig = CalibrationBuilder.build(camera("2", 6.0f, null, null), camera("3", 2.2f, null, null), 1280, 960, true, 90, false)
        assertFalse(rig.metric)
        assertEquals(6.0 / 6.4 * 1280, rig.ref.k.fx, 1e-3)
        assertEquals(CalibrationBuilder.ASSUMED_BASELINE_M, StereoGeometry.rectify(rig.r, rig.t).baseline, 1e-12)
    }

    @Test
    fun calibrationUsesPublishedPose() {
        // second camera 10 mm towards the top of the phone: image +x points to the phone's
        // bottom on a 90-degree sensor, so it is on the negative side of a horizontal rig
        val rig = CalibrationBuilder.build(
            camera("2", 6.0f, portraitBack, floatArrayOf(0f, 0f, 0f)),
            camera("3", 2.2f, portraitBack, floatArrayOf(0f, 0.01f, 0f)), 1280, 960, true, 90, false,
        )
        assertTrue(rig.metric)
        val rect = StereoGeometry.rectify(rig.r, rig.t)
        assertFalse(rect.vertical)
        assertEquals(-0.01, rect.baseline, 1e-7)
        val flipped = CalibrationBuilder.build(
            camera("2", 6.0f, portraitBack, floatArrayOf(0f, 0f, 0f)),
            camera("3", 2.2f, portraitBack, floatArrayOf(0f, 0.01f, 0f)), 1280, 960, true, 90, false, flip = true,
        )
        assertEquals(0.01, StereoGeometry.rectify(flipped.r, flipped.t).baseline, 1e-7)
    }

    @Test
    fun placeholderPoseFromARealPhoneIsReinterpreted() {
        // values a tester's phone reported: identical rotations that ignore the sensor
        // mounting, and a 2 mm lens spacing
        val q = floatArrayOf(0f, 1f, 0f, 0f)
        val rig = CalibrationBuilder.build(
            camera("3", 12.2f, q, floatArrayOf(0.0018f, 0.0010f, 0f)),
            camera("4", 5.6f, q, floatArrayOf(0f, 0f, 0f)), 1280, 960, false, 90, false,
        )
        assertFalse(rig.metric)
        assertTrue(rig.source.contains("placeholder"))
        val rect = StereoGeometry.rectify(rig.r, rig.t)
        assertTrue(rect.vertical) // 1.8 mm along device x is along image y
        assertEquals(CalibrationBuilder.ASSUMED_BASELINE_M, kotlin.math.abs(rect.baseline), 1e-9)
    }

    @Test
    fun parallaxDirectionIgnoresStrayMatches() {
        val rng = Random(5)
        val ux = cos(1.1)
        val uy = sin(1.1)
        val res = ArrayList<Double>()
        repeat(80) {
            val len = rng.nextDouble() * 80 - 40
            res += len * ux + rng.nextGaussian() * 0.3
            res += len * uy + rng.nextGaussian() * 0.3
        }
        repeat(25) {
            res += rng.nextDouble() * 120 - 60
            res += rng.nextDouble() * 120 - 60
        }
        val d = StereoGeometry.parallaxDirection(res.toDoubleArray(), 200.0)
        assertNotNull(d)
        assertTrue(kotlin.math.abs(d!!.x * ux + d.y * uy) > 0.995)
        val noise = DoubleArray(200) { rng.nextDouble() * 120 - 60 }
        assertNull(StereoGeometry.parallaxDirection(noise, 200.0))
    }

    @Test
    fun rotationIsRecoveredFromASimilarityAndPolished() {
        val f = 480.0
        val cx = 320.0
        val cy = 240.0
        val truth = StereoGeometry.rodrigues(doubleArrayOf(0.026, -0.052, 0.044)) // aux rays -> ref frame
        val rng = Random(2)
        val n = 120
        val rays = DoubleArray(n * 3)
        val target = DoubleArray(n * 2)
        for (i in 0 until n) {
            val ray = doubleArrayOf((rng.nextDouble() - 0.5) * 1.2, (rng.nextDouble() - 0.5) * 0.9, 1.0)
            val q = Mat3.mulVec(truth, ray)
            for (k in 0..2) rays[i * 3 + k] = ray[k]
            target[i * 2] = q[0] / q[2] + rng.nextGaussian() * 0.0005
            target[i * 2 + 1] = q[1] / q[2] + rng.nextGaussian() * 0.0005
        }
        // the similarity a 2D fit would find near the image centre: roll plus a shift
        val c = Mat3.mulVec(truth, doubleArrayOf(0.0, 0.0, 1.0))
        val roll = 0.044
        val m = doubleArrayOf(
            cos(roll), -sin(roll), 0.0,
            sin(roll), cos(roll), 0.0,
        )
        m[2] = cx + c[0] / c[2] * f - (m[0] * cx + m[1] * cy) + 4.0 // a few pixels off, as in practice
        m[5] = cy + c[1] / c[2] * f - (m[3] * cx + m[4] * cy) - 3.0
        val (rough, scale) = StereoGeometry.rotationFromSimilarity(m, f, cx, cy)
        assertEquals(1.0, scale, 1e-9)
        assertTrue(StereoGeometry.angleBetween(rough, truth) < 1.5)
        val (polished, agree) = StereoGeometry.refineRotation(rough, rays, target, BooleanArray(n) { true }, 3.0 / f)
        assertTrue(StereoGeometry.angleBetween(polished, truth) < 0.05)
        assertTrue(agree > n * 0.9)
    }

    @Test
    fun syncGroupsFramesOfTwoCamerasAndDropsOrphans() {
        val s = FrameSync<Long>(2, 10) { it }
        assertNull(s.add(0, 100))
        assertEquals(listOf(100L, 103L), s.add(1, 103))
        assertNull(s.add(0, 200)) // its partner never arrives
        assertNull(s.add(0, 300))
        assertEquals(listOf(300L, 298L), s.add(1, 298))
        assertEquals(1, s.dropped)
    }

    @Test
    fun syncWaitsForAllThreeCameras() {
        val s = FrameSync<Long>(3, 10) { it }
        assertNull(s.add(0, 100))
        assertNull(s.add(2, 104))
        assertNull(s.add(0, 133))
        assertEquals(listOf(100L, 101L, 104L), s.add(1, 101))
        assertNull(s.add(1, 134))
        assertEquals(listOf(133L, 134L, 137L), s.add(2, 137))
        assertEquals(0, s.dropped)
    }

    @Test
    fun gyroIntegrationRecoversAConstantTurn() {
        val gyro = GyroIntegrator()
        for (i in 0..100) gyro.add(i * 5_000_000L, 0.0, 0.4, 0.0) // 200 Hz for half a second
        val r = gyro.rotation(50_000_000, 450_000_000)!!
        val want = StereoGeometry.rodrigues(doubleArrayOf(0.0, 0.4 * 0.4, 0.0))
        for (i in 0..8) assertEquals(want[i], r[i], 1e-9)
        assertEquals(0.4, gyro.peakRate(0, 500_000_000)!!, 1e-12)
        assertNull(gyro.rotation(450_000_000, 900_000_000)) // not covered by samples
    }

    @Test
    fun turningThePhoneLeftMovesTheSceneRightInTheUprightView() {
        // portrait phone, back camera mounted at 90 degrees: upright "right" is image -y
        val theta = 0.05
        val turnLeft = StereoGeometry.rodrigues(doubleArrayOf(0.0, theta, 0.0)) // about the phone's up axis
        val f = 480.0
        val h = StereoGeometry.motionHomography(
            turnLeft, StereoGeometry.orientationRotation(90, false), Mat3.identity, f, 320.0, 240.0,
        )
        val p = Mat3.mulVec(h, doubleArrayOf(320.0, 240.0, 1.0))
        assertEquals(320.0, p[0] / p[2], 1e-9)
        assertEquals(240.0 - f * kotlin.math.tan(theta), p[1] / p[2], 1e-9)
    }

    @Test
    fun lineFitIsRobustAndRefusesAFlatScene() {
        val rng = Random(1)
        val n = 500
        val x = FloatArray(n) { 2 + rng.nextFloat() * 28 }
        val y = FloatArray(n) { if (it % 6 == 0) rng.nextFloat() * 60 else 1.8f * x[it] - 4 + rng.nextGaussian().toFloat() * 0.1f }
        val line = StereoGeometry.fitLine(x, y)
        assertNotNull(line)
        assertEquals(1.8, line!!.slope, 0.02)
        assertEquals(-4.0, line.offset, 0.3)
        assertTrue(line.inliers > 380)
        assertNull(StereoGeometry.fitLine(FloatArray(n) { 5f }, y))
    }
}
