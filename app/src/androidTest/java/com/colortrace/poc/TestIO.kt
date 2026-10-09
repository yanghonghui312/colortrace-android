package com.colortrace.poc

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.platform.app.InstrumentationRegistry
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Base64

/** base64 字节序解码扩展（金标 JSON 通用；小端，与导出脚本 tobytes 一致）。 */
fun String.toF32(): FloatArray {
    val bytes = Base64.getDecoder().decode(this)
    val fb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
    return FloatArray(bytes.size / 4).also { fb.get(it) }
}

fun String.toF64(): DoubleArray {
    val bytes = Base64.getDecoder().decode(this)
    val db = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asDoubleBuffer()
    return DoubleArray(bytes.size / 8).also { db.get(it) }
}

/** P1 对拍测试的共享工具（assets / base64 / PNG 解码 / u8 比对）。 */
object TestIO {
    fun assetText(name: String): String =
        InstrumentationRegistry.getInstrumentation().context.assets
            .open(name).bufferedReader().use { it.readText() }

    fun assetBytes(name: String): ByteArray =
        InstrumentationRegistry.getInstrumentation().context.assets.open(name)
            .use { it.readBytes() }

    /** PNG 资产 → (交错 RGB float [0,1], side)。与桌面 read_image 同语义（u8/255）。 */
    fun decodePngAsset(name: String): Pair<FloatArray, Int> {
        val bytes = assetBytes(name)
        val bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        require(bmp != null && bmp.width == bmp.height) { "PNG 解码失败或非正方形: $name" }
        val side = bmp.width
        val argb = bmp.copy(Bitmap.Config.ARGB_8888, false)
        val px = IntArray(side * side)
        argb.getPixels(px, 0, side, 0, 0, side, side)
        val out = FloatArray(px.size * 3)
        for (i in px.indices) {
            val v = px[i]
            out[3 * i] = (v shr 16 and 0xFF) / 255f
            out[3 * i + 1] = (v shr 8 and 0xFF) / 255f
            out[3 * i + 2] = (v and 0xFF) / 255f
        }
        argb.recycle()
        return Pair(out, side)
    }

    /** PNG 资产 → u8 RGB 字节（金标期望输出）。 */
    fun pngAssetToU8(name: String): ByteArray {
        val (f32, _) = decodePngAsset(name)
        val u8 = ByteArray(f32.size)
        for (i in f32.indices) u8[i] = (f32[i] * 255f + 0.5f).toInt().toByte()
        return u8
    }

    /** float [0,1] → u8（⚠ 与桌面 write_image 同为 banker's rounding：kotlin round = rint）。 */
    fun toU8Round(img: FloatArray): ByteArray {
        val out = ByteArray(img.size)
        for (i in img.indices) {
            val v = img[i].coerceIn(0f, 1f)
            out[i] = kotlin.math.round(v * 255.0).toInt().toByte()
        }
        return out
    }

    fun maxAbsDiff(a: ByteArray, b: ByteArray): Int {
        var m = 0
        for (i in a.indices) m = maxOf(m, kotlin.math.abs(a[i] - b[i]))
        return m
    }

    fun countDiffOver(a: ByteArray, b: ByteArray, tol: Int): Int {
        var n = 0
        for (i in a.indices) if (kotlin.math.abs(a[i] - b[i]) > tol) n++
        return n
    }
}
