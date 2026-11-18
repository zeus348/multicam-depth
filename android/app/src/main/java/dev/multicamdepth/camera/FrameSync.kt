package dev.multicamdepth.camera

/** One camera frame copied out of the ImageReader. [nv21] is the Y plane followed by interleaved VU. */
class Frame(val timestampNs: Long, val width: Int, val height: Int, val nv21: ByteArray)

/**
 * Groups frames of several cameras by sensor timestamp.
 * Frames arrive in timestamp order per stream. A group is emitted when the oldest waiting
 * frame of every stream lies within [toleranceNs]; otherwise the oldest of them can never
 * be completed and is dropped. Not thread safe: call from one thread.
 */
class FrameSync<T>(private val streams: Int, private val toleranceNs: Long, private val timestamp: (T) -> Long) {
    private val queues = List(streams) { ArrayDeque<T>() }
    var dropped = 0L
        private set

    /** Adds a frame of stream [stream]; returns one frame per stream, in stream order, when a group completes. */
    fun add(stream: Int, frame: T): List<T>? {
        val q = queues[stream]
        q.addLast(frame)
        if (q.size > MAX_PENDING) drop(q) // bound memory if another camera stalls
        while (queues.all { it.isNotEmpty() }) {
            val oldest = queues.minByOrNull { timestamp(it.first()) }!!
            val newest = queues.maxOf { timestamp(it.first()) }
            if (newest - timestamp(oldest.first()) <= toleranceNs) return queues.map { it.removeFirst() }
            drop(oldest)
        }
        return null
    }

    fun clear() = queues.forEach { it.clear() }

    private fun drop(q: ArrayDeque<T>) {
        q.removeFirst()
        dropped++
    }

    private companion object {
        const val MAX_PENDING = 6
    }
}
