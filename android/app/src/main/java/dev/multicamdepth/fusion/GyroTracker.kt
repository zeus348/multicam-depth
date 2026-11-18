package dev.multicamdepth.fusion

import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Handler
import android.os.HandlerThread

/** Feeds the phone's gyroscope into a [GyroIntegrator] while started. */
class GyroTracker(private val sensors: SensorManager) : SensorEventListener {
    val integrator = GyroIntegrator()
    private var thread: HandlerThread? = null

    /** false if the phone has no gyroscope */
    fun start(): Boolean {
        val gyro = sensors.getDefaultSensor(Sensor.TYPE_GYROSCOPE) ?: return false
        if (thread != null) return true
        val t = HandlerThread("gyro").apply { start() }
        thread = t
        return sensors.registerListener(this, gyro, SensorManager.SENSOR_DELAY_FASTEST, Handler(t.looper))
    }

    fun stop() {
        sensors.unregisterListener(this)
        thread?.quitSafely()
        thread = null
    }

    override fun onSensorChanged(e: SensorEvent) =
        integrator.add(e.timestamp, e.values[0].toDouble(), e.values[1].toDouble(), e.values[2].toDouble())

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit
}
