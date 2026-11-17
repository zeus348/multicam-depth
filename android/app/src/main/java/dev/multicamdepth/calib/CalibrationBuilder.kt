package dev.multicamdepth.calib

import kotlin.math.sqrt

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

object CalibrationBuilder {
    const val ASSUMED_BASELINE_M = 0.012

    /** Lens modules are several millimetres wide, so a smaller reported spacing is a placeholder. */
    private const val MIN_PLAUSIBLE_BASELINE_M = 0.004

    /**
     * @param distorted true if the delivered images still contain lens distortion
     *        (distortion correction off or unsupported), so coefficients must be applied.
     * @param flip swap which side the second camera is on; the user's override for phones
     *        that report it wrongly (near and far would otherwise come out swapped).
     */
    fun build(
        ref: PhysicalCamera,
        aux: PhysicalCamera,
        width: Int,
        height: Int,
        distorted: Boolean,
        sensorOrientation: Int,
        frontFacing: Boolean,
        flip: Boolean = false,
    ): StereoRig {
        val notes = ArrayList<String>()
        val refModel = model(ref, width, height, distorted, notes)
        val auxModel = model(aux, width, height, distorted, notes)

        var r = Mat3.identity
        var t = doubleArrayOf(-ASSUMED_BASELINE_M, 0.0, 0.0)
        var metric = false
        if (hasPose(ref) && hasPose(aux)) {
            val r1 = StereoGeometry.quatToRot(ref.poseRotation!!)
            val r2 = StereoGeometry.quatToRot(aux.poseRotation!!)
            val expected = StereoGeometry.orientationRotation(sensorOrientation, frontFacing)
            val d = DoubleArray(3) { (ref.poseTranslation!![it] - aux.poseTranslation!![it]).toDouble() }
            if (StereoGeometry.angleBetween(r1, r2) < 0.01 && StereoGeometry.angleBetween(r1, expected) > 20) {
                // Both lenses report the same rotation, and it disagrees with how the images are
                // oriented: a placeholder. Positions are still given in device axes, so map
                // them into the image using the sensor orientation instead.
                t = Mat3.mulVec(expected, d)
                notes += "pose rotation: placeholder, using sensor orientation"
            } else {
                r = Mat3.mul(r2, Mat3.transpose(r1))
                t = Mat3.mulVec(r2, d)
                notes += "pose rotation: published"
            }
            val length = sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2])
            when {
                length >= MIN_PLAUSIBLE_BASELINE_M -> {
                    metric = true
                    notes += "baseline: published"
                }
                length > 1e-5 -> {
                    // keep the direction, which is all a placeholder can still tell us
                    t = DoubleArray(3) { t[it] / length * ASSUMED_BASELINE_M }
                    notes += "baseline: published %.1f mm is implausible, assuming %.0f mm".format(
                        length * 1000, ASSUMED_BASELINE_M * 1000,
                    )
                }
                else -> {
                    r = Mat3.identity
                    t = doubleArrayOf(-ASSUMED_BASELINE_M, 0.0, 0.0)
                    notes += "baseline: none published, assumed"
                }
            }
        } else {
            notes += "pose: none published, assumed"
        }
        if (flip) {
            t = DoubleArray(3) { -t[it] }
            notes += "camera sides swapped by user"
        }
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
