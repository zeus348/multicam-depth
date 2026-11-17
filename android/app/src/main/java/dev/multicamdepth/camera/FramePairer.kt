package dev.multicamdepth.camera

import kotlin.math.abs

/** One camera frame copied out of the ImageReader. [nv21] is the Y plane followed by interleaved VU. */
class Frame(val timestampNs: Long, val width: Int, val height: Int, val nv21: ByteArray)

/**
 * Pairs frames of the two cameras by sensor timestamp.
 * Frames arrive in timestamp order per camera. A frame is paired with the closest
 * frame of the other camera if they are within [toleranceNs]; unmatched frames are dropped.
 * Not thread safe: call from one thread.
 */
class FramePairer<T>(private val toleranceNs: Long, private val timestamp: (T) -> Long) {
    private val ref = ArrayDeque<T>()
    private val aux = ArrayDeque<T>()
    var dropped = 0L
        private set

    fun addRef(frame: T): Pair<T, T>? {
        ref.addLast(frame)
        return match()
    }

    fun addAux(frame: T): Pair<T, T>? {
        aux.addLast(frame)
        return match()
    }

    fun clear() {
        ref.clear()
        aux.clear()
    }

    private fun match(): Pair<T, T>? {
        while (ref.isNotEmpty() && aux.isNotEmpty()) {
            val a = ref.first()
            val b = aux.first()
            val diff = timestamp(a) - timestamp(b)
            if (abs(diff) <= toleranceNs) {
                // a later frame of the older queue might be an even better partner
                val betterRef = ref.getOrNull(1)?.let { abs(timestamp(it) - timestamp(b)) < abs(diff) } ?: false
                val betterAux = aux.getOrNull(1)?.let { abs(timestamp(a) - timestamp(it)) < abs(diff) } ?: false
                when {
                    betterRef -> drop(ref)
                    betterAux -> drop(aux)
                    else -> return ref.removeFirst() to aux.removeFirst()
                }
            } else if (diff < 0) {
                drop(ref) // ref frame is too old to ever get a partner
            } else {
                drop(aux)
            }
        }
        // bound memory if one camera stalls
        while (ref.size > MAX_PENDING) drop(ref)
        while (aux.size > MAX_PENDING) drop(aux)
        return null
    }

    private fun drop(q: ArrayDeque<T>) {
        q.removeFirst()
        dropped++
    }

    private companion object {
        const val MAX_PENDING = 4
    }
}
