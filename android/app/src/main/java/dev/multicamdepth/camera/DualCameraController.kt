package dev.multicamdepth.camera

import android.annotation.SuppressLint
import android.graphics.ImageFormat
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.params.OutputConfiguration
import android.hardware.camera2.params.SessionConfiguration
import android.media.Image
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import java.util.concurrent.Executor

/**
 * Opens one logical camera and streams two of its physical cameras at the same time.
 * Every matched pair of frames is delivered to [onPair] on the camera thread.
 */
class DualCameraController(
    private val manager: CameraManager,
    private val pair: CameraPair,
    private val width: Int,
    private val height: Int,
    /** true: ask the phone for uncorrected images (the app then undistorts them itself) */
    private val disableDistortionCorrection: Boolean,
    private val onPair: (ref: Frame, aux: Frame) -> Unit,
    private val onError: (String) -> Unit,
) {
    private val thread = HandlerThread("camera").apply { start() }
    private val handler = Handler(thread.looper)
    private val executor = Executor { handler.post(it) }
    private val pairer = FramePairer<Frame>(PAIR_TOLERANCE_NS) { it.timestampNs }
    private var device: CameraDevice? = null
    private var refReader: ImageReader? = null
    private var auxReader: ImageReader? = null

    val droppedFrames get() = pairer.dropped

    @SuppressLint("MissingPermission") // the activity only constructs this after the permission is granted
    fun open() {
        try {
            manager.openCamera(pair.logicalId, executor, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    device = camera
                    startSession(camera)
                }

                override fun onDisconnected(camera: CameraDevice) = camera.close()

                override fun onError(camera: CameraDevice, error: Int) {
                    camera.close()
                    onError("camera error $error")
                }
            })
        } catch (e: Exception) {
            onError("cannot open camera: ${e.message}")
        }
    }

    fun close() {
        handler.post {
            device?.close()
            device = null
            refReader?.close()
            auxReader?.close()
            pairer.clear()
            thread.quitSafely()
        }
    }

    private fun startSession(camera: CameraDevice) {
        val ref = reader { pairer.addRef(it) }
        val aux = reader { pairer.addAux(it) }
        refReader = ref
        auxReader = aux
        val outputs = listOf(
            OutputConfiguration(ref.surface).apply { setPhysicalCameraId(pair.ref.id) },
            OutputConfiguration(aux.surface).apply { setPhysicalCameraId(pair.aux.id) },
        )
        val callback = object : CameraCaptureSession.StateCallback() {
            override fun onConfigured(session: CameraCaptureSession) {
                try {
                    val request = camera.createCaptureRequest(CameraDevice.TEMPLATE_RECORD).apply {
                        addTarget(ref.surface)
                        addTarget(aux.surface)
                        // anything that moves the lens or warps the image breaks the calibration
                        set(CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE, CameraMetadata.LENS_OPTICAL_STABILIZATION_MODE_OFF)
                        set(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE, CameraMetadata.CONTROL_VIDEO_STABILIZATION_MODE_OFF)
                        if (disableDistortionCorrection && pair.canDisableDistortionCorrection) {
                            set(CaptureRequest.DISTORTION_CORRECTION_MODE, CameraMetadata.DISTORTION_CORRECTION_MODE_OFF)
                        }
                    }
                    session.setRepeatingRequest(request.build(), null, handler)
                } catch (e: Exception) {
                    onError("cannot start capture: ${e.message}")
                }
            }

            override fun onConfigureFailed(session: CameraCaptureSession) =
                onError("this phone refused to stream cameras ${pair.ref.id} and ${pair.aux.id} together at ${width}x$height")
        }
        try {
            camera.createCaptureSession(SessionConfiguration(SessionConfiguration.SESSION_REGULAR, outputs, executor, callback))
        } catch (e: Exception) {
            onError("cannot configure cameras: ${e.message}")
        }
    }

    private fun reader(add: (Frame) -> Pair<Frame, Frame>?): ImageReader =
        ImageReader.newInstance(width, height, ImageFormat.YUV_420_888, 3).apply {
            setOnImageAvailableListener({ r ->
                val image = r.acquireNextImage() ?: return@setOnImageAvailableListener
                val frame = image.use { Frame(it.timestamp, it.width, it.height, toNv21(it)) }
                add(frame)?.let { (a, b) -> onPair(a, b) }
            }, handler)
        }

    private companion object {
        /** Half a frame at 30 fps: sensors that are not hardware synced still pair up. */
        const val PAIR_TOLERANCE_NS = 16_000_000L

        fun toNv21(image: Image): ByteArray {
            val w = image.width
            val h = image.height
            val out = ByteArray(w * h * 3 / 2)
            val y = image.planes[0]
            val yBuf = y.buffer
            if (y.rowStride == w) {
                yBuf.get(out, 0, w * h)
            } else {
                for (row in 0 until h) {
                    yBuf.position(row * y.rowStride)
                    yBuf.get(out, row * w, w)
                }
            }
            val u = image.planes[1]
            val v = image.planes[2]
            val uBuf = u.buffer
            val vBuf = v.buffer
            if (v.pixelStride == 2 && v.rowStride == w) {
                // already interleaved as V U V U ...; the V buffer is one byte short of the full plane
                val n = minOf(vBuf.remaining(), w * h / 2)
                vBuf.get(out, w * h, n)
                if (n < w * h / 2) out[w * h * 3 / 2 - 1] = uBuf.get(uBuf.limit() - 1)
            } else {
                var o = w * h
                for (row in 0 until h / 2) for (col in 0 until w / 2) {
                    out[o++] = vBuf.get(row * v.rowStride + col * v.pixelStride)
                    out[o++] = uBuf.get(row * u.rowStride + col * u.pixelStride)
                }
            }
            return out
        }
    }
}
