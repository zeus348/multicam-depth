package dev.multicamdepth.camera

import android.graphics.ImageFormat
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraMetadata
import dev.multicamdepth.calib.PhysicalCamera
import kotlin.math.abs
import kotlin.math.ln

/**
 * A logical camera together with two or three of its physical cameras. [ref] is the camera
 * every depth map is expressed in; [aux2], when present, forms a second stereo pair with it.
 */
class CameraPair(
    val logicalId: String,
    val facing: Int,
    val sensorOrientation: Int,
    val syncType: Int?,
    val timestampSource: Int?,
    val canDisableDistortionCorrection: Boolean,
    val hasDistortionCorrection: Boolean,
    val ref: PhysicalCamera,
    val aux: PhysicalCamera,
    val aux2: PhysicalCamera? = null,
) {
    val label: String
        get() {
            val side = if (facing == CameraMetadata.LENS_FACING_FRONT) "front" else "back"
            fun name(c: PhysicalCamera) = "${c.id} (${"%.1f".format(c.focalMm)} mm)"
            return "$side: ${name(ref)} + ${name(aux)}" + (aux2?.let { " + ${name(it)}" } ?: "")
        }

    /** Largest 4:3 (else any) YUV size both cameras offer, at most [maxWidth] wide. */
    fun commonSize(maxWidth: Int = 1280): Pair<Int, Int>? {
        var shared = ref.yuvSizes.intersect(aux.yuvSizes.toSet())
        aux2?.let { shared = shared.intersect(it.yuvSizes.toSet()) }
        val common = shared.filter { it.first <= maxWidth }
        return common.filter { it.first * 3 == it.second * 4 }.maxByOrNull { it.first }
            ?: common.maxByOrNull { it.first * it.second }
    }
}

object CameraCatalog {

    /**
     * All usable combinations: three-camera sets first, then pairs; back cameras first,
     * most similar fields of view first.
     */
    fun pairs(manager: CameraManager): List<CameraPair> {
        val out = ArrayList<CameraPair>()
        for (id in manager.cameraIdList) {
            val chars = manager.getCameraCharacteristics(id)
            val caps = chars.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES) ?: continue
            if (CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_LOGICAL_MULTI_CAMERA !in caps) continue
            val physical = chars.physicalCameraIds.mapNotNull { pid ->
                runCatching { describe(pid, manager.getCameraCharacteristics(pid)) }.getOrNull()
            }
            val modes = chars.get(CameraCharacteristics.DISTORTION_CORRECTION_AVAILABLE_MODES)
            fun combo(ref: PhysicalCamera, aux: PhysicalCamera, aux2: PhysicalCamera?) = CameraPair(
                logicalId = id,
                facing = chars.get(CameraCharacteristics.LENS_FACING) ?: CameraMetadata.LENS_FACING_BACK,
                sensorOrientation = chars.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90,
                syncType = chars.get(CameraCharacteristics.LOGICAL_MULTI_CAMERA_SENSOR_SYNC_TYPE),
                timestampSource = chars.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE),
                canDisableDistortionCorrection =
                    modes?.contains(CameraMetadata.DISTORTION_CORRECTION_MODE_OFF) == true,
                hasDistortionCorrection = modes != null && modes.any { it != CameraMetadata.DISTORTION_CORRECTION_MODE_OFF },
                ref = ref,
                aux = aux,
                aux2 = aux2,
            )
            for (i in physical.indices) for (j in i + 1 until physical.size) {
                val (ref, aux) = listOf(physical[i], physical[j]).sortedBy { it.fovRatio }
                out += combo(ref, aux, null)
            }
            if (physical.size >= 3) {
                // narrowest, middle, widest: the middle one overlaps both others, so it is the
                // reference; the wider one covers its whole view, the narrower one its centre
                val byFov = physical.sortedBy { it.fovRatio }
                out += combo(byFov[1], byFov[2], byFov[0])
            }
        }
        return out.filter { it.commonSize() != null }.sortedWith(
            compareBy<CameraPair> { it.facing != CameraMetadata.LENS_FACING_BACK }
                .thenBy { it.aux2 == null }
                .thenBy { abs(ln(it.aux.fovRatio / it.ref.fovRatio)) },
        )
    }

    /** Text dump of what the device exposes; shown when nothing usable is found and useful in bug reports. */
    fun report(manager: CameraManager): String = buildString {
        for (id in manager.cameraIdList) {
            val c = manager.getCameraCharacteristics(id)
            val caps = c.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES) ?: IntArray(0)
            val logical = CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_LOGICAL_MULTI_CAMERA in caps
            append("camera $id facing=${c.get(CameraCharacteristics.LENS_FACING)} logical=$logical")
            append(" physical=${c.physicalCameraIds}\n")
            append("  focal=${c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.toList()}")
            append(" intrinsics=${c.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION) != null}")
            append(" pose=${c.get(CameraCharacteristics.LENS_POSE_TRANSLATION) != null}\n")
        }
    }

    private fun describe(id: String, c: CameraCharacteristics): PhysicalCamera? {
        val focal = c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.firstOrNull() ?: return null
        val sensor = c.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE) ?: return null
        val array = c.get(CameraCharacteristics.SENSOR_INFO_PRE_CORRECTION_ACTIVE_ARRAY_SIZE)
            ?: c.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE) ?: return null
        val sizes = c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
            ?.getOutputSizes(ImageFormat.YUV_420_888)?.map { it.width to it.height } ?: return null
        return PhysicalCamera(
            id = id,
            focalMm = focal,
            sensorWidthMm = sensor.width,
            sensorHeightMm = sensor.height,
            arrayWidth = array.width(),
            arrayHeight = array.height(),
            intrinsics = c.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION),
            distortion = c.get(CameraCharacteristics.LENS_DISTORTION),
            poseRotation = c.get(CameraCharacteristics.LENS_POSE_ROTATION),
            poseTranslation = c.get(CameraCharacteristics.LENS_POSE_TRANSLATION),
            yuvSizes = sizes,
        )
    }
}
