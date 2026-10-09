package com.colortrace.poc

import android.graphics.Bitmap
import android.net.Uri
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import java.io.File

/**
 * 导出尺寸档（P2.21）：**原始尺寸** 与 **预压缩（短边 1920）**。
 *
 * 两条关卡：
 *  1. `decodeForExport` 的短边口径——横图/竖图都缩到短边 = 目标（长边按比例）、
 *     **原图更小不放大**、`shortSide = 0` 保持原尺寸（老口径不变）；
 *  2. **预压缩导出与"全尺寸导出再缩放"画面等价**——这是"缩放在解码这一步做"这个
 *     设计决定的前提（LUT 是从内容图的 128² 缩略图算出来的，见
 *     `EncoderEngine.thumbFromMat`）。等价 ⇒ 可以先缩再走管线，省掉数倍时间。
 *     ⚠️ 断言的不是逐位相等（两次重采样路径不同），阈值取实测后留裕量（见下）。
 *
 * 测试造的是真文件（file:// 也能被 ContentResolver 读），走完整"读头 → 采样 →
 * 精确缩放"路径；临时图 @After 清掉。
 */
@RunWith(AndroidJUnit4::class)
class ExportSizeTest {

    private val ctx get() = InstrumentationRegistry.getInstrumentation().targetContext
    private lateinit var repo: EngineRepository
    private val tmp = mutableListOf<File>()

    @Before
    fun setUp() {
        assertTrue("OpenCV native 加载失败", OpenCVLoader.initLocal())
        repo = EngineRepository(ctx)
    }

    @After
    fun tearDown() {
        for (f in tmp) f.delete()
        tmp.clear()
    }

    // ---- 造图/落盘 ----

    /** 渐变 + 肤色块的确定性图（覆盖高光/肤色/暗部，与 TiledTest 同思路）。 */
    private fun synthetic(w: Int, h: Int): Bitmap {
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val row = IntArray(w)
        for (y in 0 until h) {
            for (x in 0 until w) {
                val fx = x.toFloat() / w
                val fy = y.toFloat() / h
                val r = (30 + 200 * fx).toInt().coerceIn(0, 255)
                val g = (60 + 150 * fy).toInt().coerceIn(0, 255)
                val b = (230 - 140 * fx * fy).toInt().coerceIn(0, 255)
                row[x] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
            bmp.setPixels(row, 0, w, 0, y, w, 1)
        }
        return bmp
    }

    /** 造图落盘并返回 file:// uri（走真解码路径）。 */
    private fun sourceFile(w: Int, h: Int): Uri {
        val bmp = synthetic(w, h)
        val f = File(ctx.filesDir, "T-export-${w}x$h.png")
        f.outputStream().use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
        bmp.recycle()
        tmp.add(f)
        return Uri.fromFile(f)
    }

    private fun pixels(b: Bitmap): IntArray =
        IntArray(b.width * b.height).also { b.getPixels(it, 0, b.width, 0, 0, b.width, b.height) }

    // ---- 1. 短边口径 ----

    @Test
    fun shortSideTargetsAndNoUpscale() {
        val target = ExportSize.SOCIAL.shortSide
        assertEquals("档位口径就是 1920", 1920, target)

        // 横图 2600×2000：短边 2000 > 1920 ⇒ 长边 = round(1920×2600/2000) = 2496
        repo.decodeForExport(sourceFile(2600, 2000), target).let {
            assertEquals(2496, it.width)
            assertEquals("短边必须正好落在目标上", 1920, it.height)
            it.recycle()
        }

        // 竖图 2000×2600：同一档位，短边（宽）= 1920
        repo.decodeForExport(sourceFile(2000, 2600), target).let {
            assertEquals("竖图短边是宽", 1920, it.width)
            assertEquals(2496, it.height)
            it.recycle()
        }

        // 回归（2026-10-01 用户真机 3712×5568 → 1919×2880）：float32 下短边乘积落在
        // 1919.999x，截断会吃成 1919。1921×2882 是 float32 语义下**截断必失败**的最小
        // 尺寸对（PC 上按 Kotlin 同序 f32 运算穷举证得）；四舍五入后必须正好 1920。
        repo.decodeForExport(sourceFile(2882, 1921), target).let {
            assertEquals("截断改四舍五入后短边正好 1920", 1920, it.height)
            assertEquals(2880, it.width)
            it.recycle()
        }

        // 原图更小 ⇒ 不放大（1600×1200 短边 1200 < 1920）
        repo.decodeForExport(sourceFile(1600, 1200), target).let {
            assertEquals(1600, it.width)
            assertEquals(1200, it.height)
            it.recycle()
        }

        // shortSide = 0 = 老口径：原尺寸不动
        repo.decodeForExport(sourceFile(2600, 2000), ExportSize.ORIGINAL.shortSide).let {
            assertEquals(2600, it.width)
            assertEquals(2000, it.height)
            it.recycle()
        }
        println("[export-size] 短边口径：横 2496×1920 / 竖 1920×2496 / 小图不放大 / 原始不动")
    }

    // ---- 2. 画面等价（"缩放在解码这一步做"的前提）----

    @Test
    fun precompressedMatchesDownscaledFullExport() {
        val session = repo.loadPreset(TestIO.assetText("goldens/encoder_preset.json"))
        val full = synthetic(768, 512)

        // 基准：全尺寸走完管线，再缩到目标尺寸
        val outA = TiledPipeline(session.model, null).process(full, ProtectMode.OFF, 0.7f)
        val a = Bitmap.createScaledBitmap(outA, 192, 128, true)
        // 对照：先缩到目标尺寸，再走同一条管线
        val smallSrc = Bitmap.createScaledBitmap(full, 192, 128, true)
        val b = TiledPipeline(session.model, null).process(smallSrc, ProtectMode.OFF, 0.7f)

        assertEquals(a.width, b.width)
        assertEquals(a.height, b.height)
        val pa = pixels(a); val pb = pixels(b)
        var maxAbs = 0; var sum = 0L; var over2 = 0
        for (i in pa.indices) {
            val d = kotlin.math.abs(
                ((pa[i] shr 16) and 0xFF) - ((pb[i] shr 16) and 0xFF))
            val dg = kotlin.math.abs(
                ((pa[i] shr 8) and 0xFF) - ((pb[i] shr 8) and 0xFF))
            val db = kotlin.math.abs((pa[i] and 0xFF) - (pb[i] and 0xFF))
            val m = maxOf(d, dg, db)
            if (m > maxAbs) maxAbs = m
            if (m >= 2) over2++
            sum += d + dg + db
        }
        val mean = sum.toDouble() / (pa.size * 3)
        println("[export-size] 等价性：maxΔ=$maxAbs/255 平均Δ=%.3f/255 差≥2 的通道=".format(mean) +
                "$over2/${pa.size * 3}")

        // 实测（首跑，2026-10-01 真机）：maxΔ=1/255、平均 0.042/255、差≥2 的通道 **0/73728**。
        // 门限留裕量（平均 12×、最大 4×）；⚠️ 本测试用的是 768→192 的 4× 缩放，
        // 比真实预压缩（如 4000→2880 = 1.39×）**更狠**（重采样跨度更大、缩略图差异更大）
        // ⇒ 这里过，真实导出只会更接近。
        assertTrue("平均差必须远小于人眼阈值（实测 %.3f/255）".format(mean), mean <= 0.5)
        assertTrue("单点最大差不得超过 4/255（实测 $maxAbs）", maxAbs <= 4)
        assertTrue("不该有通道差到 2/255（实测 $over2）", over2 == 0)

        outA.recycle(); a.recycle(); b.recycle(); full.recycle()
    }
}
