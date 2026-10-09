package com.colortrace.poc

import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.colortrace.engine.ContentModel
import com.colortrace.engine.EncoderEngine
import com.colortrace.engine.LabConv
import com.colortrace.engine.StatEngine
import com.colortrace.engine.StatPresets
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.dnn.Net

/**
 * 统计类方法档金标对拍（P2.13）：reinhard（lab_stats）与 ot（ot_linear）。
 *
 * 金标（`scripts/export_android_stats_vectors.py` 产）：
 *  - `lab_stats_preset.json` / `ot_preset.json`：桌面真实 fit 产物（原样入库）；
 *  - `stats_content.png`：256² 内容图（与桌面同一起点）；
 *  - `stats_out_{lab,ot}_{s100,s050}.png`：期望输出（strength 1 / 0.5）；
 *  - `stats_golden.json`：ot 的 T 矩阵中间锚（内容统计下的 ot_linear_map 产物，
 *    float64）——移动端 Jacobi 特征分解与 numpy eigh 数值路径不同，T 容差
 *    1e-6 定位误差来源；最终 u8 对拍 ≤2/255（P1 同源关卡）。
 *
 * 另锁：分块管线（TiledPipeline）与整图路径 u8 逐位一致（骨架复用正确性）、
 * 不支持的 kind 明确报错（不静默降级）。
 */
@RunWith(AndroidJUnit4::class)
class StatPresetTest {

    private lateinit var report: JSONObject
    private lateinit var golden: JSONObject

    private val ctx get() = InstrumentationRegistry.getInstrumentation().targetContext

    @Before
    fun setUp() {
        assertTrue("OpenCV native 加载失败", OpenCVLoader.initLocal())
        report = JSONObject()
        report.put("device", android.os.Build.MODEL)
        golden = JSONObject(TestIO.assetText("goldens/stats_golden.json"))
    }

    /** u8 对拍：max|Δ| 与 ≤1/255 占比（金标 PNG vs 移动端输出）。 */
    private fun compareU8(got: FloatArray, w: Int, h: Int, pngName: String): JSONObject {
        fun u8(v: Float): Int = ((v.coerceIn(0f, 1f) * 255f + 0.5f).toInt())
        val pngBytes = TestIO.assetBytes("goldens/$pngName")
        val refBmp = BitmapFactory.decodeByteArray(pngBytes, 0, pngBytes.size)
        val ref = IntArray(w * h)
        refBmp.getPixels(ref, 0, w, 0, 0, w, h)
        var maxD = 0
        var le1 = 0
        var firstBad = -1
        for (i in ref.indices) {
            val g = (0xFF shl 24) or
                    ((u8(got[3 * i])) shl 16) or
                    ((u8(got[3 * i + 1])) shl 8) or
                    u8(got[3 * i + 2])
            val d = maxOf(
                kotlin.math.abs((ref[i] shr 16 and 0xFF) - (g shr 16 and 0xFF)),
                kotlin.math.abs((ref[i] shr 8 and 0xFF) - (g shr 8 and 0xFF)),
                kotlin.math.abs((ref[i] and 0xFF) - (g and 0xFF)))
            if (d > maxD) maxD = d
            if (d <= 1) le1++
            if (d > 1 && firstBad < 0) firstBad = i
        }
        refBmp.recycle()
        return JSONObject()
            .put("max_delta", maxD)
            .put("le1_ratio", le1.toDouble() / ref.size)
            .put("first_bad", firstBad)
    }

    @Test
    fun labStatsGolden() {
        val preset = StatPresets.parse(
            TestIO.assetText("goldens/lab_stats_preset.json")) as ContentModel.LabStats
        val bmp = BitmapFactory.decodeByteArray(
            TestIO.assetBytes("goldens/stats_content.png"), 0,
            TestIO.assetBytes("goldens/stats_content.png").size)
        val (rgb, w, h) = bitmapToRgbF32(bmp)
        bmp.recycle()

        // 整图路径：内容统计 → 全强度映射 → 强度混合（桌面 apply 语义）
        val (mu, std) = StatEngine.labStats(LabConv.rgbToLab(rgb))
        val mapped = StatEngine.mapLabStats(rgb, preset.refMean, preset.refStd, mu, std)

        val details = JSONObject()
        for ((name, s) in listOf("lab_s100" to 1.0f, "lab_s050" to 0.5f)) {
            val out = if (s >= 1f) mapped
            else FloatArray(rgb.size) { rgb[it] * (1f - s) + mapped[it] * s }
            val cmp = compareU8(out, w, h, "stats_out_$name.png")
            details.put(name, cmp)
            println("[stats] lab $name: maxΔ=${cmp.getInt("max_delta")} " +
                    "≤1占比=${"%.4f".format(cmp.getDouble("le1_ratio"))}")
            assertTrue("lab $name 超容差：$cmp", cmp.getInt("max_delta") <= 2)
        }
        report.put("lab_stats", details)
        saveReport()
    }

    @Test
    fun otGolden() {
        val preset = StatPresets.parse(
            TestIO.assetText("goldens/ot_preset.json")) as ContentModel.OtLinear
        val bmp = BitmapFactory.decodeByteArray(
            TestIO.assetBytes("goldens/stats_content.png"), 0,
            TestIO.assetBytes("goldens/stats_content.png").size)
        val (rgb, w, h) = bitmapToRgbF32(bmp)
        bmp.recycle()

        // 中间锚：T 矩阵（Jacobi vs numpy eigh——数值路径不同，容差 1e-6）
        val (mu, cov) = StatEngine.meanCov(rgb)
        val t = StatEngine.otLinearMap(cov, preset.refCov)
        val tRef = golden.getJSONArray("ot_T")
        var tMaxD = 0.0
        for (r in 0 until 3) for (c in 0 until 3) {
            val d = kotlin.math.abs(t[r][c] - tRef.getJSONArray(r).getDouble(c))
            if (d > tMaxD) tMaxD = d
        }
        println("[stats] ot T max|Δ|=$tMaxD")
        assertTrue("ot T 超容差 max|Δ|=$tMaxD", tMaxD <= 1e-6)

        val mapped = StatEngine.mapOtLinear(rgb, t, mu, preset.refMean)
        val details = JSONObject().put("t_max_delta", tMaxD)
        for ((name, s) in listOf("ot_s100" to 1.0f, "ot_s050" to 0.5f)) {
            val out = if (s >= 1f) mapped
            else FloatArray(rgb.size) { rgb[it] * (1f - s) + mapped[it] * s }
            val cmp = compareU8(out, w, h, "stats_out_$name.png")
            details.put(name, cmp)
            println("[stats] ot $name: maxΔ=${cmp.getInt("max_delta")} " +
                    "≤1占比=${"%.4f".format(cmp.getDouble("le1_ratio"))}")
            assertTrue("ot $name 超容差：$cmp", cmp.getInt("max_delta") <= 2)
        }
        report.put("ot_linear", details)
        saveReport()
    }

    /** 分块骨架一致性：统计类走 TiledPipeline 必须与整图路径 u8 逐位一致。 */
    @Test
    fun tiledEqualsMonolithic() {
        val preset = StatPresets.parse(
            TestIO.assetText("goldens/lab_stats_preset.json"))
        val bmp = BitmapFactory.decodeByteArray(
            TestIO.assetBytes("goldens/stats_content.png"), 0,
            TestIO.assetBytes("goldens/stats_content.png").size)
        val w = bmp.width; val h = bmp.height
        val mono = Session(preset, null as Net?).apply(bmp, ProtectMode.OFF, 0.5f)
        val tiled = TiledPipeline(preset, null as Net?)
            .process(bmp, ProtectMode.OFF, 0.5f)
        val a = IntArray(w * h); val b = IntArray(w * h)
        mono.getPixels(a, 0, w, 0, 0, w, h)
        tiled.getPixels(b, 0, w, 0, 0, w, h)
        var diffs = 0
        for (i in a.indices) if (a[i] != b[i]) diffs++
        println("[stats] tiled vs monolithic: u8 像素差=$diffs")
        assertTrue("分块≠整图：$diffs 像素", diffs == 0)
        mono.recycle(); tiled.recycle(); bmp.recycle()
        report.put("tiled_equality", JSONObject().put("px_diff", diffs))
        saveReport()
    }

    /** 不支持的 kind 明确报错（swot/phr/skin_protect——不静默降级）。
     *  P2.15 起 `region` 已支持，从拒绝清单移除（对拍见 RegionStatTest）。 */
    @Test
    fun rejectUnsupportedKinds() {
        for (kind in listOf("swot", "phr_lab", "skin_protect")) {
            val json = """{"version":1,"kind":"$kind","strength":1.0,"params":{}}"""
            try {
                StatPresets.parse(json)
                throw AssertionError("$kind 应被拒绝")
            } catch (e: IllegalArgumentException) {
                // 预期：报错信息里带上已支持档位
                assertTrue("报错应说明已支持档位",
                           e.message!!.contains("encoder"))
            }
        }
        report.put("reject_kinds", "ok")
        saveReport()
    }

    /**
     * 设备端 fit 冒烟（P2.13 算法选择）：样片 → reinhard/ot 会话跑通。
     * 数学口径：**fit(A) 再 apply(A) = 恒等**（把图的统计量对齐到自己）——
     * 所以用交叉 apply（fit 样片 stats_content、apply 到人像探针）断言非恒等，
     * 并检查自反性（apply 自身 ≈ 原图，≤2/255）。未知方法档明确报错。
     */
    @Test
    fun fitSampleStatSmoke() {
        val repo = EngineRepository(ctx)
        val sample = BitmapFactory.decodeByteArray(
            TestIO.assetBytes("goldens/stats_content.png"), 0,
            TestIO.assetBytes("goldens/stats_content.png").size)
        val target = BitmapFactory.decodeByteArray(
            TestIO.assetBytes("goldens/protect_probe_portrait.jpg"), 0,
            TestIO.assetBytes("goldens/protect_probe_portrait.jpg").size)
        val details = JSONObject()
        for (method in listOf("reinhard", "ot")) {
            val s = repo.fitSample(sample, method)
            assertTrue("$method 会话应是统计类模型",
                       s.model is ContentModel.LabStats ||
                       s.model is ContentModel.OtLinear)
            // ① 自反性：fit(A) → apply(A) ≈ A（统计量对齐自己的极限行为，≤2/255）
            val self = s.apply(sample, ProtectMode.OFF, 1f)
            val selfDiff = pixelDiff(sample, self)
            println("[stats] fitSample $method 自反性: maxΔ=${selfDiff.maxCh}")
            assertTrue("$method 自反性被破坏（maxΔ=${selfDiff.maxCh}）",
                       selfDiff.maxCh <= 2)
            // ② 交叉 apply：fit(A) → apply(B) 应非恒等（两图统计量不同）
            val cross = s.apply(target, ProtectMode.OFF, 1f)
            val crossDiff = pixelDiff(target, cross)
            println("[stats] fitSample $method 交叉 apply: 差 $crossDiff 像素")
            assertTrue("$method 交叉 apply 应产生追色效果", crossDiff.diff > 0)
            details.put(method, JSONObject()
                .put("self_max_delta", selfDiff.maxCh.toDouble())
                .put("cross_diff", crossDiff.diff))
            self.recycle(); cross.recycle()
        }
        try {
            repo.fitSample(sample, "swot")
            throw AssertionError("swot 设备端 fit 应被拒绝")
        } catch (e: IllegalArgumentException) { /* 预期 */ }
        sample.recycle(); target.recycle()
        report.put("fit_sample_smoke", details)
        saveReport()
    }

    private data class PxDiff(val diff: Int, val maxCh: Int)

    private fun pixelDiff(a: android.graphics.Bitmap,
                          b: android.graphics.Bitmap): PxDiff {
        val n = a.width * a.height
        val pa = IntArray(n); val pb = IntArray(n)
        a.getPixels(pa, 0, a.width, 0, 0, a.width, a.height)
        b.getPixels(pb, 0, b.width, 0, 0, b.width, b.height)
        var diff = 0
        var maxCh = 0
        for (i in pa.indices) {
            if (pa[i] == pb[i]) continue
            diff++
            val d = maxOf(
                kotlin.math.abs((pa[i] shr 16 and 0xFF) - (pb[i] shr 16 and 0xFF)),
                kotlin.math.abs((pa[i] shr 8 and 0xFF) - (pb[i] shr 8 and 0xFF)),
                kotlin.math.abs((pa[i] and 0xFF) - (pb[i] and 0xFF)))
            if (d > maxCh) maxCh = d
        }
        return PxDiff(diff, maxCh)
    }

    /**
     * chroma 保护金标（P2.13 续）：肤色锁定 × 统计类基方法——移动端走
     * TiledPipeline.process(CHROMA, protect)，应与桌面
     * `SkinProtectTransform(base, chroma, protect)` 逐像素一致（≤2/255）。
     * protect ∈ {1.0, 0.5} × {reinhard, ot} 四案。
     */
    @Test
    fun chromaProtectGolden() {
        val po = golden.getJSONObject("protect_outputs")
        val bmp = BitmapFactory.decodeByteArray(
            TestIO.assetBytes("goldens/stats_content.png"), 0,
            TestIO.assetBytes("goldens/stats_content.png").size)
        val w = bmp.width; val h = bmp.height
        val details = JSONObject()
        for ((presetName, tag) in listOf("lab_stats_preset.json" to "lab",
                                         "ot_preset.json" to "ot")) {
            val preset = StatPresets.parse(TestIO.assetText("goldens/$presetName"))
            for ((pname, p) in listOf("c1" to 1.0f, "c05" to 0.5f)) {
                val out = TiledPipeline(preset, null as Net?)
                    .process(bmp, ProtectMode.CHROMA, 1f, null, p)
                val cmp = bitmapCompare(out, "stats_prot_${tag}_$pname.png")
                details.put("${tag}_$pname", cmp)
                println("[stats] chroma ${tag}_$pname: maxΔ=${cmp.getInt("max_delta")} " +
                        "≤1占比=${"%.4f".format(cmp.getDouble("le1_ratio"))}")
                out.recycle()
                assertTrue("chroma ${tag}_$pname 超容差：$cmp",
                           cmp.getInt("max_delta") <= 2)
            }
        }
        bmp.recycle()
        report.put("chroma_protect", details)
        saveReport()
    }

    /** 与金标 PNG 的 u8 对比（maxΔ / ≤1 占比 / 首个差异像素）。 */
    private fun bitmapCompare(out: android.graphics.Bitmap, pngName: String): JSONObject {
        val pngBytes = TestIO.assetBytes("goldens/$pngName")
        val refBmp = BitmapFactory.decodeByteArray(pngBytes, 0, pngBytes.size)
        val w = out.width; val h = out.height
        val pa = IntArray(w * h); val pb = IntArray(w * h)
        out.getPixels(pa, 0, w, 0, 0, w, h)
        refBmp.getPixels(pb, 0, w, 0, 0, w, h)
        var maxD = 0
        var le1 = 0
        var firstBad = -1
        for (i in pa.indices) {
            val d = maxOf(
                kotlin.math.abs((pa[i] shr 16 and 0xFF) - (pb[i] shr 16 and 0xFF)),
                kotlin.math.abs((pa[i] shr 8 and 0xFF) - (pb[i] shr 8 and 0xFF)),
                kotlin.math.abs((pa[i] and 0xFF) - (pb[i] and 0xFF)))
            if (d > maxD) maxD = d
            if (d <= 1) le1++
            if (d > 1 && firstBad < 0) firstBad = i
        }
        refBmp.recycle()
        return JSONObject()
            .put("max_delta", maxD)
            .put("le1_ratio", le1.toDouble() / pa.size)
            .put("first_bad", firstBad)
    }

    private fun saveReport() {
        ctx.filesDir.mkdirs()
        ctx.openFileOutput("stat_preset_report.json", android.content.Context.MODE_PRIVATE)
            .use { it.write(report.toString().toByteArray()) }
    }
}
