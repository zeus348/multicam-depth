package dev.multicamdepth.fusion

import dev.multicamdepth.calib.Mat3
import dev.multicamdepth.calib.StereoGeometry
import kotlin.math.sqrt

/**
 * Keeps recent gyroscope samples and integrates them into the device's rotation between
 * two instants. No Android types, so it runs in plain JVM tests; thread safe.
 * Mirrors integrate_gyro in tools/mcdepth/geometry.py.
 */
class GyroIntegrator(private val capacity: Int = 2048) {
    private val times = LongArray(capacity)
    private val rates = DoubleArray(capacity * 3)
    private var next = 0
    private var count = 0

    /** [x], [y], [z]: angular velocity in rad/s in the device frame, at [timeNs]. */
    @Synchronized
    fun add(timeNs: Long, x: Double, y: Double, z: Double) {
        times[next] = timeNs
        rates[next * 3] = x
        rates[next * 3 + 1] = y
        rates[next * 3 + 2] = z
        next = (next + 1) % capacity
        if (count < capacity) count++
    }

    private fun index(i: Int) = (next - count + i + capacity) % capacity

    /**
     * Rotation R of the device between [t0] and [t1], with p_at_t0 = R * p_at_t1 for a
     * direction p fixed to the device. Null if the samples do not cover the interval.
     */
    @Synchronized
    fun rotation(t0: Long, t1: Long): DoubleArray? {
        if (count < 2 || t1 < t0) return null
        if (times[index(0)] > t0 + SLACK_NS || times[index(count - 1)] < t1 - SLACK_NS) return null
        var r = Mat3.identity
        for (i in 0 until count) {
            val j = index(i)
            val start = maxOf(t0, if (i == 0) times[j] else times[index(i - 1)])
            // the last sample is held to the end of the interval
            val end = if (i == count - 1 && times[j] < t1) t1 else minOf(t1, times[j])
            if (end <= start) continue
            val dt = (end - start) * 1e-9
            r = Mat3.mul(r, StereoGeometry.rodrigues(doubleArrayOf(rates[j * 3] * dt, rates[j * 3 + 1] * dt, rates[j * 3 + 2] * dt)))
        }
        return r
    }

    /** Fastest rotation rate (rad/s) seen in [t0, t1], or null if there are no samples there. */
    @Synchronized
    fun peakRate(t0: Long, t1: Long): Double? {
        var peak: Double? = null
        for (i in 0 until count) {
            val j = index(i)
            if (times[j] < t0 || times[j] > t1) continue
            val rate = sqrt(rates[j * 3] * rates[j * 3] + rates[j * 3 + 1] * rates[j * 3 + 1] + rates[j * 3 + 2] * rates[j * 3 + 2])
            if (peak == null || rate > peak) peak = rate
        }
        return peak
    }

    private companion object {
        const val SLACK_NS = 20_000_000L
    }
}
