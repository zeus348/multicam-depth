package dev.multicamdepth

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.hardware.SensorManager
import android.hardware.camera2.CameraManager
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.view.View
import android.view.WindowManager
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.ImageView
import android.widget.Spinner
import android.widget.TextView
import androidx.core.content.FileProvider
import dev.multicamdepth.calib.CalibrationBuilder
import dev.multicamdepth.calib.StereoRig
import dev.multicamdepth.camera.CameraCatalog
import dev.multicamdepth.camera.CameraPair
import dev.multicamdepth.camera.MultiCameraController
import dev.multicamdepth.calib.StereoGeometry
import dev.multicamdepth.fusion.GyroTracker
import dev.multicamdepth.camera.Frame
import dev.multicamdepth.depth.DepthEngine
import dev.multicamdepth.depth.View as DepthView
import dev.multicamdepth.record.SessionRecorder
import org.opencv.android.OpenCVLoader
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs

class MainActivity : Activity() {
    private companion object {
        const val ALIGN_RETRY_FRAMES = 10
        const val STEADY_RAD_S = 0.3 // slower than this the phone counts as held still
        const val KEEP_STEADY = 0.6 // weight of the previous depth frame
        const val KEEP_MOVING = 0.85 // while turning the new frame is blurrier, so trust it less
    }

    private lateinit var cameraManager: CameraManager
    private lateinit var status: TextView
    private lateinit var info: TextView
    private lateinit var refView: ImageView
    private lateinit var depthView: ImageView
    private lateinit var recordButton: Button
    private lateinit var baselineButton: Button
    private lateinit var distortionButton: Button
    private var appUndistorts = true
    @Volatile private var viewMode = DepthView.REFERENCE

    private val depthThread = HandlerThread("depth").apply { start() }
    private val depthHandler = Handler(depthThread.looper)
    private val depthBusy = AtomicBoolean()

    private var pairs: List<CameraPair> = emptyList()
    private var pair: CameraPair? = null
    private var flipSides = false
    private var controller: MultiCameraController? = null
    private var rig: StereoRig? = null
    private var rig2: StereoRig? = null
    private lateinit var gyro: GyroTracker
    @Volatile private var notice = ""

    // touched only on the depth thread
    private var engine: DepthEngine? = null
    private var engine2: DepthEngine? = null // reference camera + third camera
    private var alignNote = ""
    private var alignNote2 = ""
    private var lastFrameNs = 0L
    private var deviceToCamera = dev.multicamdepth.calib.Mat3.identity
    private var clocksShared = false
    private var frameCount = 0

    @Volatile private var alignRequested = true
    @Volatile private var recorder: SessionRecorder? = null
    private var lastSession: File? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        cameraManager = getSystemService(CameraManager::class.java)
        gyro = GyroTracker(getSystemService(SensorManager::class.java))
        status = findViewById(R.id.statusText)
        info = findViewById(R.id.infoText)
        refView = findViewById(R.id.refView)
        depthView = findViewById(R.id.depthView)
        recordButton = findViewById(R.id.recordButton)
        baselineButton = findViewById(R.id.baselineButton)

        recordButton.setOnClickListener { toggleRecording() }
        findViewById<Button>(R.id.alignButton).setOnClickListener { alignRequested = true }
        findViewById<Button>(R.id.shareButton).setOnClickListener { shareLast() }
        distortionButton = findViewById(R.id.distortionButton)
        distortionButton.setOnClickListener {
            appUndistorts = !appUndistorts
            restartCamera()
        }
        refView.setOnClickListener { viewMode = viewMode.next() }
        baselineButton.setOnClickListener {
            flipSides = !flipSides
            restartCamera()
        }

        if (!OpenCVLoader.initLocal()) {
            status.text = "OpenCV failed to load"
            return
        }
        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            setUpPairs()
        } else {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), 1)
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        if (grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
            setUpPairs()
            startCamera()
        } else {
            status.text = "Camera permission is needed to see anything."
        }
    }

    private fun setUpPairs() {
        pairs = CameraCatalog.pairs(cameraManager)
        if (pairs.isEmpty()) {
            status.text = "This phone does not let apps stream two of its cameras at once " +
                "(no logical multi-camera with a shared output size)."
            info.text = CameraCatalog.report(cameraManager)
            return
        }
        val spinner = findViewById<Spinner>(R.id.pairSpinner)
        spinner.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, pairs.map { it.label })
        spinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                if (pair !== pairs[position]) {
                    pair = pairs[position]
                    restartCamera()
                }
            }

            override fun onNothingSelected(parent: AdapterView<*>?) = Unit
        }
        pair = pairs[0]
    }

    override fun onResume() {
        super.onResume()
        startCamera()
    }

    override fun onPause() {
        stopCamera()
        super.onPause()
    }

    override fun onDestroy() {
        depthThread.quitSafely()
        super.onDestroy()
    }

    private fun restartCamera() {
        stopCamera()
        startCamera()
    }

    private fun startCamera() {
        val p = pair ?: return
        if (controller != null) return
        val (w, h) = p.commonSize() ?: return
        // Who removes lens distortion: the app (from published coefficients) or the phone.
        // Phones differ in which of the two actually works, so it can be switched.
        val distorted = if (p.canDisableDistortionCorrection && p.hasDistortionCorrection) appUndistorts else !p.hasDistortionCorrection
        distortionButton.visibility = if (p.canDisableDistortionCorrection && p.hasDistortionCorrection) View.VISIBLE else View.GONE
        distortionButton.text = "Lens distortion removed by: ${if (appUndistorts) "app" else "phone"}. Tap to switch"
        val front = p.facing == android.hardware.camera2.CameraMetadata.LENS_FACING_FRONT
        val newRig = CalibrationBuilder.build(p.ref, p.aux, w, h, distorted, p.sensorOrientation, front, flipSides)
        val newRig2 = p.aux2?.let { CalibrationBuilder.build(p.ref, it, w, h, distorted, p.sensorOrientation, front, flipSides) }
        rig = newRig
        rig2 = newRig2
        notice = ""
        gyro.start()
        baselineButton.visibility = View.VISIBLE
        baselineButton.text = "Near and far look swapped? Tap to flip" + if (flipSides) " (flipped)" else ""

        // live depth runs at reduced resolution; recordings keep the full frames
        val scale = if (w > 800) 2 else 1
        val orientation = p.sensorOrientation
        depthHandler.post {
            engine = try {
                DepthEngine(newRig, w / scale, h / scale, orientation)
            } catch (e: Exception) {
                runOnUiThread { status.text = "calibration unusable: ${e.message}" }
                null
            }
            engine2 = newRig2?.let {
                try {
                    DepthEngine(it, w / scale, h / scale, orientation)
                } catch (e: Exception) {
                    null
                }
            }
            alignRequested = true
            alignNote = ""
            alignNote2 = ""
            lastFrameNs = 0
            deviceToCamera = StereoGeometry.orientationRotation(orientation, front)
            // the gyroscope can only be related to the frames if both use the same clock
            clocksShared = p.timestampSource == android.hardware.camera2.CameraMetadata.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME
            val e = engine
            val text = buildString {
                append("stream ${w}x$h, depth ${w / scale}x${h / scale}\n")
                append("sync=${p.syncType} (1 = hardware synced)  clock=${p.timestampSource} (1 = same as IMU)\n")
                append(newRig.source.replace("; ", "\n")).append('\n')
                append(describe(newRig, p))
                if (newRig2 != null) append("third camera ${p.aux2?.id}: ${newRig2.source.replace("; ", ", ")}\n")
                if (e != null) {
                    append("baseline %.1f mm %s, focal %.0f px\n".format(
                        abs(e.rect.baseline) * 1000, if (e.rect.vertical) "vertical" else "horizontal", e.focal))
                    append("usable range about %.2f m to %.1f m\n".format(
                        e.focal * abs(e.rect.baseline) / e.numDisparities, e.focal * abs(e.rect.baseline) / 1.0))
                }
            }
            runOnUiThread { info.text = text }
        }
        controller = MultiCameraController(
            cameraManager, p, w, h, distorted, ::onFrames,
            onNotice = { notice = it },
            onError = { msg -> runOnUiThread { status.text = msg } },
        ).also { it.open() }
    }

    /** Raw numbers behind the calibration, for diagnosing a phone whose depth looks wrong. */
    private fun describe(rig: StereoRig, p: CameraPair) = buildString {
        for ((cam, model) in listOf(p.ref to rig.ref, p.aux to rig.aux)) {
            append("cam ${cam.id}: array ${cam.arrayWidth}x${cam.arrayHeight}, sensor %.2fx%.2f mm\n".format(cam.sensorWidthMm, cam.sensorHeightMm))
            append("  published K: ${cam.intrinsics?.joinToString { "%.1f".format(it) } ?: "none"}\n")
            append("  stream K: %.1f %.1f %.1f %.1f\n".format(model.k.fx, model.k.fy, model.k.cx, model.k.cy))
            append("  published dist: ${cam.distortion?.joinToString { "%.4f".format(it) } ?: "none"}\n")
            append("  pose t (mm): ${cam.poseTranslation?.joinToString { "%.2f".format(it * 1000) } ?: "none"}\n")
            append("  pose q: ${cam.poseRotation?.joinToString { "%.4f".format(it) } ?: "none"}\n")
        }
        val angle = Math.toDegrees(kotlin.math.acos(((rig.r[0] + rig.r[4] + rig.r[8] - 1) / 2).coerceIn(-1.0, 1.0)))
        append("relative: T (mm) ${rig.t.joinToString { "%.2f".format(it * 1000) }}, rotation %.2f deg\n".format(angle))
        append("distortion modes: off=${p.canDisableDistortionCorrection} on=${p.hasDistortionCorrection}\n")
    }

    private fun stopCamera() {
        if (recorder != null) toggleRecording()
        controller?.close()
        controller = null
        gyro.stop()
    }

    /** Camera thread. [aux2] is null unless three cameras are streaming. */
    private fun onFrames(ref: Frame, aux: Frame, aux2: Frame?) {
        recorder?.add(ref, aux, aux2)
        if (!depthBusy.compareAndSet(false, true)) return // depth is slower than the camera: skip frames
        depthHandler.post {
            try {
                val e = engine ?: return@post
                val e2 = if (aux2 != null) engine2 else null
                val start = System.nanoTime()

                // What the gyroscope says happened since the frame we last showed.
                val turn = if (clocksShared && lastFrameNs != 0L) gyro.integrator.rotation(lastFrameNs, ref.timestampNs) else null
                val rate = if (clocksShared) gyro.integrator.peakRate(ref.timestampNs - 100_000_000, ref.timestampNs + 40_000_000) else null
                val steady = rate == null || rate < STEADY_RAD_S
                lastFrameNs = ref.timestampNs

                // Align on a sharp frame: the first frames after opening are dark or blurred
                // and a turning phone skews the two rolling shutters differently, so wait
                // until it is held still and keep trying until it works.
                frameCount++
                val retry = frameCount % ALIGN_RETRY_FRAMES == 0
                if (steady) {
                    val all = alignRequested
                    alignRequested = false
                    if (all || (!e.aligned && retry)) alignNote = e.refine(ref, aux)
                    if (e2 != null && aux2 != null && (all || (!e2.aligned && retry))) alignNote2 = e2.refine(ref, aux2)
                } else if (alignRequested) {
                    alignNote = "hold the phone still to align"
                }

                val disp = e.match(ref, aux)
                var fusion = ""
                if (e2 != null && aux2 != null) {
                    fusion = if (e.aligned && e2.aligned) {
                        val other = e2.match(ref, aux2)
                        e.absorb(disp, other, e2).also { other.release() }
                    } else {
                        "third camera: $alignNote2"
                    }
                }
                val motion = turn?.let { StereoGeometry.motionHomography(it, deviceToCamera, e.rect.r1, e.focal, e.centerX, e.centerY) }
                val mode = viewMode
                val r = e.finish(disp, mode, motion, if (steady) KEEP_STEADY else KEEP_MOVING, start)

                val unit = if (rig?.metric == true && !e.selfCalibrated) "m" else "(relative)"
                val center = if (r.centerDepth.isNaN()) "no reading" else "%.2f %s".format(r.centerDepth, unit)
                val rec = recorder?.let { "  REC ${it.written.get()} sets, ${it.skipped.get()} skipped" } ?: ""
                val motionNote = when {
                    !clocksShared -> "gyro: unusable, camera clock differs"
                    rate == null -> "gyro: no data"
                    else -> "gyro: %.2f rad/s, %s".format(rate, if (motion != null) "compensating" else "starting")
                }
                val line = "centre: $center   ${(r.validFraction * 100).toInt()}% valid   ${r.millis} ms\n" +
                    "colours: red %.2f to blue %.2f $unit\n".format(r.nearest, r.farthest) +
                    "left: ${mode.label} (tap image to switch)\n$alignNote\n" +
                    (if (fusion.isNotEmpty()) "$fusion\n" else "") +
                    (if (notice.isNotEmpty()) "$notice\n" else "") +
                    "$motionNote$rec"
                runOnUiThread {
                    refView.setImageBitmap(r.reference)
                    depthView.setImageBitmap(r.depth)
                    status.text = line
                }
            } catch (ex: Exception) {
                runOnUiThread { status.text = "depth failed: ${ex.message}" }
            } finally {
                depthBusy.set(false)
            }
        }
    }

    private fun toggleRecording() {
        val active = recorder
        if (active != null) {
            recorder = null
            recordButton.setText(R.string.record)
            lastSession = active.dir
            Thread { active.finish() }.start()
            info.text = "saved ${active.dir}\n\n${info.text}"
            return
        }
        val p = pair ?: return
        val r = rig ?: return
        val name = "session_" + SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val three = controller?.activeCameras == 3
        recorder = SessionRecorder(File(getExternalFilesDir(null), name), p, r, if (three) rig2 else null, getSystemService(SensorManager::class.java))
        recordButton.setText(R.string.stop)
    }

    private fun shareLast() {
        val session = lastSession ?: getExternalFilesDir(null)?.listFiles { f -> f.isDirectory }?.maxByOrNull { it.name }
        if (session == null) {
            status.text = "Nothing recorded yet."
            return
        }
        status.text = "Zipping ${session.name}..."
        Thread {
            val zip = File(cacheDir, "exports/${session.name}.zip")
            SessionRecorder.zip(session, zip)
            val uri = FileProvider.getUriForFile(this, "dev.multicamdepth.files", zip)
            val send = Intent(Intent.ACTION_SEND).setType("application/zip")
                .putExtra(Intent.EXTRA_STREAM, uri).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            runOnUiThread { startActivity(Intent.createChooser(send, session.name)) }
        }.start()
    }
}
