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
import dev.multicamdepth.camera.DualCameraController
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
    private var controller: DualCameraController? = null
    private var rig: StereoRig? = null

    // touched only on the depth thread
    private var engine: DepthEngine? = null
    private var alignNote = ""
    private var frameCount = 0

    @Volatile private var alignRequested = true
    @Volatile private var recorder: SessionRecorder? = null
    private var lastSession: File? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        cameraManager = getSystemService(CameraManager::class.java)
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
        val newRig = CalibrationBuilder.build(
            p.ref, p.aux, w, h, distorted, p.sensorOrientation,
            p.facing == android.hardware.camera2.CameraMetadata.LENS_FACING_FRONT, flipSides,
        )
        rig = newRig
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
            alignRequested = true
            alignNote = ""
            val e = engine
            val text = buildString {
                append("stream ${w}x$h, depth ${w / scale}x${h / scale}\n")
                append("sync=${p.syncType} (1 = hardware synced)  clock=${p.timestampSource} (1 = same as IMU)\n")
                append(newRig.source.replace("; ", "\n")).append('\n')
                append(describe(newRig, p))
                if (e != null) {
                    append("baseline %.1f mm %s, focal %.0f px\n".format(
                        abs(e.rect.baseline) * 1000, if (e.rect.vertical) "vertical" else "horizontal", e.focal))
                    append("usable range about %.2f m to %.1f m\n".format(
                        e.focal * abs(e.rect.baseline) / e.numDisparities, e.focal * abs(e.rect.baseline) / 1.0))
                }
            }
            runOnUiThread { info.text = text }
        }
        controller = DualCameraController(cameraManager, p, w, h, distorted, ::onPair) { msg ->
            runOnUiThread { status.text = msg }
        }.also { it.open() }
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
    }

    /** Camera thread. */
    private fun onPair(ref: Frame, aux: Frame) {
        recorder?.add(ref, aux)
        if (!depthBusy.compareAndSet(false, true)) return // depth is slower than the camera: skip frames
        depthHandler.post {
            try {
                val e = engine ?: return@post
                // the first frames after opening are often dark or blurred, so keep trying
                // until an alignment succeeds
                frameCount++
                if (alignRequested || (!e.aligned && frameCount % ALIGN_RETRY_FRAMES == 0)) {
                    alignRequested = false
                    alignNote = e.refine(ref, aux)
                }
                val mode = viewMode
                val r = e.compute(ref, aux, mode)
                val unit = if (rig?.metric == true && !e.selfCalibrated) "m" else "(relative)"
                val center = if (r.centerDepth.isNaN()) "no reading" else "%.2f %s".format(r.centerDepth, unit)
                val rec = recorder?.let { "  REC ${it.written.get()} pairs, ${it.skipped.get()} skipped" } ?: ""
                val line = "centre: $center   ${(r.validFraction * 100).toInt()}% valid   ${r.millis} ms\n" +
                    "left: ${mode.label} (tap image to switch)\n$alignNote\nskew ${(ref.timestampNs - aux.timestampNs) / 1000} us$rec"
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
        recorder = SessionRecorder(File(getExternalFilesDir(null), name), p, r, getSystemService(SensorManager::class.java))
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
