package com.colortrace.poc

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import java.io.ByteArrayOutputStream
import java.io.File
import kotlin.math.abs

/**
 * 导出编码关卡（P2.8，2026-09-29）：导出 JPEG 必须 **q100 + 4:4:4**。
 *
 * 背景：`Bitmap.compress(JPEG, 95)` **不暴露色度采样**，Android/Skia 实际输出
 * **4:2:0**（色度横竖各减半，红/青边缘可见损失），而桌面 `io._imencode_params`
 * 早就是 q100 + 4:4:4。P2.8 起导出走 `encodeJpeg444`（OpenCV imencode）。
 *
 * 断言口径（解析 JPEG SOF0 分量采样因子，与桌面核对脚本同法）：
 *  1. `encodeJpeg444` 输出 = (1,1)(1,1)(1,1)（4:4:4），且能解码回原尺寸；
 *  2. 同等输入下它一定**大于** `Bitmap.compress(95)` 的输出（后者实测 4:2:0，仅记录）；
 *  3. 生产路径 `EngineRepository.saveBitmap` 落到 MediaStore 的字节同样是 4:4:4。
 *
 * 用高频色度图案（3px 周期红/青交替）放大两种采样的差异；@After 删除测试写入的
 * 相册记录（项目纪律：不许污染照片选择器）。
 */
@RunWith(AndroidJUnit4::class)
class ExportEncodeTest {

    private lateinit var report: JSONObject
    private val savedUris = mutableListOf<android.net.Uri>()

    private val ctx get() = InstrumentationRegistry.getInstrumentation().targetContext

    @Before
    fun setUp() {
        assertTrue("OpenCV native 加载失败", OpenCVLoader.initLocal())
        report = JSONObject()
        report.put("opencv", OpenCVLoader.OPENCV_VERSION)
        report.put("device", android.os.Build.MODEL)
        report.put("api", android.os.Build.VERSION.SDK_INT)
    }

    /** 高频色度图案（红/青 3px 交替）——专门放大 4:2:0 与 4:4:4 的差别。 */
    private fun patternBitmap(w: Int, h: Int): Bitmap {
        val px = IntArray(w * h)
        for (y in 0 until h) {
            for (x in 0 until w) {
                val warm = ((x / 3 + y / 3) % 2 == 0)
                val r = if (warm) 230 else 20
                val g = if (warm) 30 else 220
                val b = if (warm) 40 else 210
                px[y * w + x] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        return Bitmap.createBitmap(px, w, h, Bitmap.Config.ARGB_8888)
    }

    /** 解析 JPEG SOF0 各分量的 (H, V) 采样因子；找不到就抛。 */
    private fun samplingFactors(jpg: ByteArray): List<Pair<Int, Int>> {
        var i = 2
        while (i < jpg.size - 3) {
            if (jpg[i].toInt() and 0xFF != 0xFF) { i++; continue }
            val m = jpg[i + 1].toInt() and 0xFF
            if (m == 0xD8 || m == 0x01 || (m in 0xD0..0xD7)) { i += 2; continue }
            val len = ((jpg[i + 2].toInt() and 0xFF) shl 8) or (jpg[i + 3].toInt() and 0xFF)
            if (m == 0xC0) {
                val n = jpg[i + 9].toInt() and 0xFF
                return (0 until n).map {
                    val s = jpg[i + 10 + it * 3 + 1].toInt() and 0xFF
                    (s shr 4) to (s and 0x0F)
                }
            }
            i += 2 + len
        }
        throw AssertionError("JPEG 中找不到 SOF0 标记")
    }

    private fun legacyBytes(bmp: Bitmap, quality: Int): ByteArray {
        val bos = ByteArrayOutputStream()
        bmp.compress(Bitmap.CompressFormat.JPEG, quality, bos)
        return bos.toByteArray()
    }

    @Test
    fun exportJpegIs444() {
        val w = 1200; val h = 900
        val bmp = patternBitmap(w, h)

        val legacy95 = legacyBytes(bmp, 95)
        val legacy100 = legacyBytes(bmp, 100)
        val new = encodeJpeg444(bmp, 100)
        val legacy95Sampling = samplingFactors(legacy95)
        val legacy100Sampling = samplingFactors(legacy100)
        val newSampling = samplingFactors(new)

        report.put("legacy95_bytes", legacy95.size)
        report.put("legacy95_sampling", legacy95Sampling.toString())
        report.put("legacy100_sampling", legacy100Sampling.toString())
        report.put("new_bytes", new.size)
        report.put("new_sampling", newSampling.toString())
        println("[export] Bitmap.compress(95)=${legacy95.size}B $legacy95Sampling")
        println("[export] Bitmap.compress(100)=${legacy100.size}B $legacy100Sampling")
        println("[export] encodeJpeg444(100)=${new.size}B $newSampling")

        // 1) 新路径必须 4:4:4（三分量 H=V=1）
        assertEquals("新路径必须是 4:4:4", listOf(1 to 1, 1 to 1, 1 to 1), newSampling)
        // 2) 旧路径的采样只记录不断言（平台行为，未来 Android 可能变）
        report.put("legacy_was_420", legacy95Sampling == listOf(2 to 2, 1 to 1, 1 to 1))
        // 3) 高色度细节下 4:4:4 一定比 4:2:0 大
        assertTrue("4:4:4 输出应大于 4:2:0 的 $legacy95.size B（实得 ${new.size} B）",
            new.size > legacy95.size)
        // 4) 能解码回原尺寸，且像素损失在 JPEG 有损范围内
        val back = BitmapFactory.decodeByteArray(new, 0, new.size)
        assertNotNull("导出字节无法解码", back)
        assertEquals(w, back!!.width)
        assertEquals(h, back.height)
        val pa = IntArray(w * h); val pb = IntArray(w * h)
        bmp.getPixels(pa, 0, w, 0, 0, w, h)
        back.getPixels(pb, 0, w, 0, 0, w, h)
        var maxCh = 0
        for (i in pa.indices) {
            for (s in intArrayOf(16, 8, 0)) {
                val d = abs(((pa[i] shr s) and 0xFF) - ((pb[i] shr s) and 0xFF))
                if (d > maxCh) maxCh = d
            }
        }
        report.put("decode_max_channel_delta", maxCh)
        println("[export] 解码回读 maxChΔ=$maxCh")
        assertTrue("解码回读单通道偏差过大 $maxCh", maxCh <= 40)
        back.recycle(); bmp.recycle()
    }

    @Test
    fun savedFileOnMediaStoreIs444() {
        val bmp = patternBitmap(800, 600)
        val repo = EngineRepository(ctx)
        val saved = repo.saveBitmap(bmp, "colortrace_encodetest_${System.currentTimeMillis()}")
        assertNotNull("saveBitmap 返回 null", saved)
        savedUris += saved!!.first
        // 生产路径写出的字节：从 MediaStore 读回并核对采样因子
        val bytes = ctx.contentResolver.openInputStream(saved.first)!!.use { it.readBytes() }
        val sampling = samplingFactors(bytes)
        report.put("saved_bytes", bytes.size)
        report.put("saved_sampling", sampling.toString())
        report.put("saved_to_gallery", saved.second)
        println("[export] MediaStore 产物 ${bytes.size}B $sampling toGallery=${saved.second}")
        assertEquals("实际保存的文件必须是 4:4:4", listOf(1 to 1, 1 to 1, 1 to 1), sampling)
        bmp.recycle()
    }

    @After
    fun tearDown() {
        // 清相册：不污染照片选择器（项目纪律：测试产物不留进媒体库）
        for (u in savedUris) {
            runCatching { ctx.contentResolver.delete(u, null, null) }
                .onFailure { Log.w("ExportEncodeTest", "删除测试记录失败 $u", it) }
        }
        runCatching { File(ctx.filesDir, "export_encode_report.json")
            .writeText(report.toString(2)) }
        Log.i("ExportEncodeTest", "report=$report")
    }
}