package dev.multicamdepth.record

import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import dev.multicamdepth.calib.CameraModel
import dev.multicamdepth.calib.StereoRig
import dev.multicamdepth.camera.CameraPair
import dev.multicamdepth.camera.Frame
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedWriter
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.ThreadPoolExecutor
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/**
 * Writes one recording: JPEG pairs, per-frame sensor timestamps, IMU samples and the
 * calibration. Layout is described in docs/session-format.md.
 */
class SessionRecorder(
    val dir: File,
    pair: CameraPair,
    rig: StereoRig,
    /** the reference camera with the third camera, when three are streaming */
    rig2: StereoRig?,
    private val sensors: SensorManager,
) {
    private val threeCameras = rig2 != null
    private val writer = ThreadPoolExecutor(1, 1, 0, TimeUnit.SECONDS, ArrayBlockingQueue(QUEUE)) { r -> Thread(r, "recorder") }
    private val frames: BufferedWriter
    private val imu: BufferedWriter
    private val imuThread = HandlerThread("imu").apply { start() }
    private var index = 0
    val written = AtomicInteger()
    val skipped = AtomicInteger()

    private val imuListener = object : SensorEventListener {
        override fun onSensorChanged(e: SensorEvent) {
            val name = if (e.sensor.type == Sensor.TYPE_GYROSCOPE) "gyro" else "accel"
            imu.write("${e.timestamp},$name,${e.values[0]},${e.values[1]},${e.values[2]}\n")
        }

        override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit
    }

    init {
        File(dir, "ref").mkdirs()
        File(dir, "aux").mkdirs()
        if (threeCameras) File(dir, "aux2").mkdirs()
        File(dir, "calibration.json").writeText(calibrationJson(pair, rig, rig2).toString(2))
        frames = File(dir, "frames.csv").bufferedWriter()
        frames.write("index,t_ref_ns,t_aux_ns,ref_file,aux_file" + (if (threeCameras) ",t_aux2_ns,aux2_file" else "") + "\n")
        imu = File(dir, "imu.csv").bufferedWriter()
        imu.write("t_ns,sensor,x,y,z\n")
        val handler = Handler(imuThread.looper)
        for (type in intArrayOf(Sensor.TYPE_ACCELEROMETER, Sensor.TYPE_GYROSCOPE)) {
            sensors.getDefaultSensor(type)?.let {
                sensors.registerListener(imuListener, it, SensorManager.SENSOR_DELAY_FASTEST, handler)
            }
        }
    }

    /** Queues a pair for writing. If storage cannot keep up the pair is skipped, never blocking the camera. */
    fun add(ref: Frame, aux: Frame, aux2: Frame?) {
        if (threeCameras && aux2 == null) return // keep every row complete
        val i = index++
        try {
            writer.execute {
                val name = "%06d.jpg".format(i)
                jpeg(ref, File(dir, "ref/$name"))
                jpeg(aux, File(dir, "aux/$name"))
                var row = "$i,${ref.timestampNs},${aux.timestampNs},ref/$name,aux/$name"
                if (threeCameras && aux2 != null) {
                    jpeg(aux2, File(dir, "aux2/$name"))
                    row += ",${aux2.timestampNs},aux2/$name"
                }
                frames.write(row + "\n")
                written.incrementAndGet()
            }
        } catch (e: java.util.concurrent.RejectedExecutionException) {
            skipped.incrementAndGet()
        }
    }

    fun finish() {
        sensors.unregisterListener(imuListener)
        writer.shutdown()
        writer.awaitTermination(10, TimeUnit.SECONDS)
        frames.close()
        Handler(imuThread.looper).post { imu.close() }
        imuThread.quitSafely()
    }

    private fun jpeg(f: Frame, file: File) {
        FileOutputStream(file).use {
            YuvImage(f.nv21, ImageFormat.NV21, f.width, f.height, null).compressToJpeg(Rect(0, 0, f.width, f.height), 92, it)
        }
    }

    private fun calibrationJson(pair: CameraPair, rig: StereoRig, rig2: StereoRig?): JSONObject {
        fun cam(c: CameraModel) = JSONObject()
            .put("id", c.id).put("width", c.width).put("height", c.height)
            .put("intrinsics", JSONArray(doubleArrayOf(c.k.fx, c.k.fy, c.k.cx, c.k.cy)))
            .put("distortion", JSONArray(c.dist))

        fun raw(p: dev.multicamdepth.calib.PhysicalCamera) = JSONObject()
            .put("id", p.id).put("focal_mm", p.focalMm.toDouble())
            .put("sensor_mm", JSONArray(doubleArrayOf(p.sensorWidthMm.toDouble(), p.sensorHeightMm.toDouble())))
            .put("array", JSONArray(intArrayOf(p.arrayWidth, p.arrayHeight)))
            .put("lens_intrinsic_calibration", p.intrinsics?.let { JSONArray(it) } ?: JSONObject.NULL)
            .put("lens_distortion", p.distortion?.let { JSONArray(it) } ?: JSONObject.NULL)
            .put("lens_pose_rotation", p.poseRotation?.let { JSONArray(it) } ?: JSONObject.NULL)
            .put("lens_pose_translation", p.poseTranslation?.let { JSONArray(it) } ?: JSONObject.NULL)

        fun stereo(r: StereoRig) = JSONObject().put("R", JSONArray(r.r)).put("T", JSONArray(r.t)).put("metric", r.metric)

        return JSONObject()
            .put("format", 1)
            .put("device", "${Build.MANUFACTURER} ${Build.MODEL}")
            .put("android_sdk", Build.VERSION.SDK_INT)
            .put("logical_camera", pair.logicalId)
            .put("sensor_orientation", pair.sensorOrientation)
            // 1 = sensors start exposure together (calibrated), 0 = approximate
            .put("sync_type", pair.syncType ?: JSONObject.NULL)
            // 1 = camera timestamps share the clock of imu.csv (elapsedRealtimeNanos), 0 = unknown offset
            .put("timestamp_source", pair.timestampSource ?: JSONObject.NULL)
            .put("ref", cam(rig.ref))
            .put("aux", cam(rig.aux))
            .put("stereo", stereo(rig))
            .put("calibration_source", rig.source)
            .put("reported", JSONObject().put("ref", raw(pair.ref)).put("aux", raw(pair.aux)))
            .apply {
                if (rig2 != null && pair.aux2 != null) {
                    put("aux2", cam(rig2.aux))
                    put("stereo2", stereo(rig2)) // X_aux2 = R * X_ref + T
                    put("calibration_source2", rig2.source)
                    getJSONObject("reported").put("aux2", raw(pair.aux2))
                }
            }
    }

    companion object {
        private const val QUEUE = 8

        fun zip(session: File, out: File) {
            out.parentFile?.mkdirs()
            ZipOutputStream(out.outputStream().buffered()).use { zip ->
                session.walkTopDown().filter { it.isFile }.forEach { f ->
                    zip.putNextEntry(ZipEntry("${session.name}/${f.relativeTo(session).path}"))
                    f.inputStream().use { it.copyTo(zip) }
                    zip.closeEntry()
                }
            }
        }
    }
}
