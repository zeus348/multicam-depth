package dev.multicamdepth.calib

import kotlin.math.abs

/** What one physical camera reports about itself, as plain data (see CameraCatalog). */
class PhysicalCamera(
    val id: String,
    val focalMm: Float,
    val sensorWidthMm: Float,
    val sensorHeightMm: Float,
    /** pre-correction active array, the coordinate system of [intrinsics] */
    val arrayWidth: Int,
    val arrayHeight: Int,
    /** LENS_INTRINSIC_CALIBRATION: fx fy cx cy s, or null */
    val intrinsics: FloatArray?,
    /** LENS_DISTORTION: k1 k2 k3 p1 p2 (Android order), or null */
    val distortion: FloatArray?,
    /** LENS_POSE_ROTATION quaternion x y z w, or null */
    val poseRotation: FloatArray?,
    /** LENS_POSE_TRANSLATION in metres, or null */
    val poseTranslation: FloatArray?,
    /** YUV_420_888 output sizes as (width, height) */
    val yuvSizes: List<Pair<Int, Int>>,
) {
    /** horizontal field of view proxy: larger = wider lens */
    val fovRatio get() = sensorWidthMm / focalMm
}

/** Used when the phone publishes no camera positions. Axis is in sensor image coordinates. */
enum class AssumedBaseline(val x: Double, val y: Double) {
    PLUS_X(1.0, 0.0), MINUS_X(-1.0, 0.0), PLUS_Y(0.0, 1.0), MINUS_Y(0.0, -1.0);

    fun next() = entries[(ordinal + 1) % entries.size]
}

object CalibrationBuilder {
    const val ASSUMED_BASELINE_M = 0.012

    /**
     * @param distorted true if the delivered images still contain lens distortion
     *        (distortion correction off or unsupported), so coefficients must be applied.
     */
    fun build(
        ref: PhysicalCamera,
        aux: PhysicalCamera,
        width: Int,
        height: Int,
        distorted: Boolean,
        assumed: AssumedBaseline,
    ): StereoRig {
        val notes = ArrayList<String>()
        val refModel = model(ref, width, height, distorted, notes)
        val auxModel = model(aux, width, height, distorted, notes)

        var r = Mat3.identity
        var t = doubleArrayOf(-assumed.x * ASSUMED_BASELINE_M, -assumed.y * ASSUMED_BASELINE_M, 0.0)
        var metric = false
        if (hasPose(ref) && hasPose(aux)) {
            val (pr, pt) = StereoGeometry.relativePose(
                ref.poseRotation!!, ref.poseTranslation!!, aux.poseRotation!!, aux.poseTranslation!!,
            )
            if (pt.any { abs(it) > 1e-4 }) {
                r = pr
                t = pt
                metric = true
            }
        }
        notes += if (metric) "pose: published by device" else "pose: ASSUMED ($assumed, ${ASSUMED_BASELINE_M * 1000} mm)"
        return StereoRig(refModel, auxModel, r, t, metric, notes.joinToString("; "))
    }

    private fun hasPose(c: PhysicalCamera) = c.poseRotation?.size == 4 && c.poseTranslation?.size == 3

    private fun model(c: PhysicalCamera, w: Int, h: Int, distorted: Boolean, notes: MutableList<String>): CameraModel {
        val pub = c.intrinsics
        val native = if (pub != null && pub.size >= 4 && pub[0] > 0f && pub[1] > 0f) {
            notes += "cam ${c.id} intrinsics: published"
            Intrinsics(pub[0].toDouble(), pub[1].toDouble(), pub[2].toDouble(), pub[3].toDouble())
        } else {
            notes += "cam ${c.id} intrinsics: from focal length"
            val f = c.focalMm.toDouble() / c.sensorWidthMm * c.arrayWidth
            Intrinsics(f, f, c.arrayWidth / 2.0, c.arrayHeight / 2.0)
        }
        val d = c.distortion
        val dist = if (distorted && d != null && d.size >= 5) {
            // Android: k1 k2 k3 p1 p2  ->  OpenCV: k1 k2 p1 p2 k3
            doubleArrayOf(d[0].toDouble(), d[1].toDouble(), d[3].toDouble(), d[4].toDouble(), d[2].toDouble())
        } else {
            DoubleArray(5)
        }
        return CameraModel(c.id, w, h, StereoGeometry.scaleIntrinsics(native, c.arrayWidth, c.arrayHeight, w, h), dist)
    }
}
