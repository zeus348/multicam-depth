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
 * Opens one logical camera and streams two or three of its physical cameras at the same
 * time. Every synchronised set of frames is delivered to [onFrames] on the camera thread;
 * the third frame is null when only two cameras are running.
 *
 * Not every phone can run three sensors at once. If the three-stream session is refused,
 * or delivers nothing, the controller falls back to two and says so through [onNotice].
 */
class MultiCameraController(
    private val manager: CameraManager,
    private val pair: CameraPair,
    private val width: Int,
    private val height: Int,
    /** true: ask the phone for uncorrected images (the app then undistorts them itself) */
    private val disableDistortionCorrection: Boolean,
    private val onFrames: (ref: Frame, aux: Frame, aux2: Frame?) -> Unit,
    private val onNotice: (String) -> Unit,
    private val onError: (String) -> Unit,
) {
    private val thread = HandlerThread("camera").apply { start() }
    private val handler = Handler(thread.looper)
    private val executor = Executor { handler.post(it) }
    private var sync = FrameSync<Frame>(2, PAIR_TOLERANCE_NS) { it.timestampNs }
    private var device: CameraDevice? = null
    private var readers: List<ImageReader> = emptyList()
    private var delivered = 0L
    private var activeSession: CameraCaptureSession? = null
    private var activeRequest: CaptureRequest.Builder? = null

    /**
     * Freezes the lenses where they are. Autofocus keeps nudging a lens, which shifts and
     * rescales its image by several pixels and so undoes the alignment between the cameras.
     */
    fun lockFocus() = focusTrigger(CameraMetadata.CONTROL_AF_TRIGGER_START)

    /** Lets autofocus run again, e.g. before re-aligning on a scene at a different distance. */
    fun unlockFocus() = focusTrigger(CameraMetadata.CONTROL_AF_TRIGGER_CANCEL)

    private fun focusTrigger(trigger: Int) {
        handler.post {
            val session = activeSession ?: return@post
            val request = activeRequest ?: return@post
            try {
                request.set(CaptureRequest.CONTROL_AF_TRIGGER, trigger)
                session.capture(request.build(), null, handler)
                request.set(CaptureRequest.CONTROL_AF_TRIGGER, CameraMetadata.CONTROL_AF_TRIGGER_IDLE)
            } catch (e: Exception) {
                // session already closed or replaced: nothing to lock
            }
        }
    }

    /** how many cameras are actually streaming */
    @Volatile var activeCameras = 0
        private set

    @SuppressLint("MissingPermission") // the activity only constructs this after the permission is granted
    fun open() {
        try {
            manager.openCamera(pair.logicalId, executor, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    device = camera
                    startSession(camera, if (pair.aux2 != null) 3 else 2)
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
            activeSession = null
            device?.close()
            device = null
            readers.forEach { it.close() }
            readers = emptyList()
            thread.quitSafely()
        }
    }

    private fun startSession(camera: CameraDevice, cameras: Int) {
        activeSession = null
        activeRequest = null
        val old = readers // still in use by the session being replaced; closed once the new one runs
        val ids = listOfNotNull(pair.ref.id, pair.aux.id, pair.aux2?.id).take(cameras)
        val groups = FrameSync<Frame>(cameras, PAIR_TOLERANCE_NS) { it.timestampNs }
        sync = groups
        delivered = 0
        val created = ids.indices.map { stream -> reader(stream, groups) }
        readers = created
        val outputs = created.mapIndexed { i, r -> OutputConfiguration(r.surface).apply { setPhysicalCameraId(ids[i]) } }

        fun fallBack(why: String) {
            if (cameras == 3 && device === camera) {
                onNotice("three cameras at once: $why. Using two.")
                startSession(camera, 2)
            } else {
                onError("this phone refused to stream cameras ${ids.joinToString(" and ")} together at ${width}x$height ($why)")
            }
        }

        val callback = object : CameraCaptureSession.StateCallback() {
            override fun onConfigured(session: CameraCaptureSession) {
                old.forEach { it.close() }
                try {
                    // settings for individual lenses need a request created with their ids
                    val base = try {
                        check(pair.perLensStabiliser)
                        camera.createCaptureRequest(CameraDevice.TEMPLATE_RECORD, ids.toSet()).also { b ->
                            ids.forEach {
                                b.setPhysicalCameraKey(
                                    CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE,
                                    CameraMetadata.LENS_OPTICAL_STABILIZATION_MODE_OFF, it,
                                )
                            }
                        }
                    } catch (e: Exception) {
                        // not offered, or refused: the whole-camera setting below still applies
                        camera.createCaptureRequest(CameraDevice.TEMPLATE_RECORD)
                    }
                    val request = base.apply {
                        created.forEach { addTarget(it.surface) }
                        // anything that moves the lens or warps the image breaks the calibration
                        set(CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE, CameraMetadata.LENS_OPTICAL_STABILIZATION_MODE_OFF)
                        // a fixed frame rate stops auto-exposure from stretching exposures (and motion blur) in dim light
                        pair.fixedFps?.let { set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, android.util.Range(it, it)) }
                        set(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE, CameraMetadata.CONTROL_VIDEO_STABILIZATION_MODE_OFF)
                        if (disableDistortionCorrection && pair.canDisableDistortionCorrection) {
                            set(CaptureRequest.DISTORTION_CORRECTION_MODE, CameraMetadata.DISTORTION_CORRECTION_MODE_OFF)
                        }
                    }
                    session.setRepeatingRequest(request.build(), null, handler)
                    activeSession = session
                    activeRequest = request
                    activeCameras = cameras
                    if (cameras == 3) {
                        // some phones accept the session but never deliver all three streams
                        handler.postDelayed({
                            if (device === camera && sync === groups && delivered == 0L) fallBack("no frames arrived")
                        }, STARTUP_TIMEOUT_MS)
                    }
                } catch (e: Exception) {
                    fallBack(e.message ?: "capture could not start")
                }
            }

            override fun onConfigureFailed(session: CameraCaptureSession) = fallBack("refused by the phone")
        }
        try {
            camera.createCaptureSession(SessionConfiguration(SessionConfiguration.SESSION_REGULAR, outputs, executor, callback))
        } catch (e: Exception) {
            fallBack(e.message ?: "session could not be created")
        }
    }

    private fun reader(stream: Int, groups: FrameSync<Frame>): ImageReader =
        ImageReader.newInstance(width, height, ImageFormat.YUV_420_888, 3).apply {
            setOnImageAvailableListener({ r ->
                val image = try {
                    r.acquireNextImage()
                } catch (e: IllegalStateException) {
                    null // reader was closed while this callback was queued
                } ?: return@setOnImageAvailableListener
                val frame = image.use { Frame(it.timestamp, it.width, it.height, toNv21(it)) }
                groups.add(stream, frame)?.let {
                    delivered++
                    onFrames(it[0], it[1], it.getOrNull(2))
                }
            }, handler)
        }

    private companion object {
        /** Half a frame at 30 fps: sensors that are not hardware synced still pair up. */
        const val PAIR_TOLERANCE_NS = 16_000_000L
        const val STARTUP_TIMEOUT_MS = 4000L

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
