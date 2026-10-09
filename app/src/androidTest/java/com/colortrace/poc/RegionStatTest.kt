package com.colortrace.poc

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.colortrace.engine.ContentModel
import com.colortrace.engine.RegionStatEngine
import com.colortrace.engine.StatPresets
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.dnn.Dnn
import org.opencv.dnn.Net

/**
 * 分区统计档金标对拍（P2.15）：region × reinhard/ot 完整移植。
 *
 * 金标（`scripts/export_android_stats_vectors.py` 产）：
 *  - `region_lab_preset.json` / `region_ot_preset.json`：桌面
 *    `fit_skin_protected(授权人像样片, skin_mode="region")` 真实产物（四区齐全）；
 *  - 期望输出 = **序列化→读回**后的预设重算（嵌套 ref 统计的 round 各自生效，
 *    外层 strength/protect 按案改值）——与移动端"预设 + UI 强度/保护强度"同起点；
 *  - `region_encoder_preset.json`：分区 × encoder（路线 0）——移动端只做嵌套解析
 *    放行（还原 encoder 会话），期望输出复用 P1 的 `p1_out_region100.png`。
 *
 * 容差：u8 逐像素 max|Δ| ≤ 2/255（P1 同源关卡）。
 */
@RunWith(AndroidJUnit4::class)
class RegionStatTest {

    private lateinit var report: JSONObject
    private lateinit var net: Net

    private val ctx get() = InstrumentationRegistry.getInstrumentation().targetContext

    @Before
    fun setUp() {
        assertTrue("OpenCV native 加载失败", OpenCVLoader.initLocal())
        report = JSONObject()
        report.put("device", android.os.Build.MODEL)
        net = readNetFromAsset("selfie_multiclass.onnx")
    }

    private fun readNetFromAsset(name: String): Net {
        val bytes = TestIO.assetBytes(name)
        val m = Mat(1, bytes.size, CvType.CV_8U)
        m.put(0, 0, bytes)
        return Dnn.readNetFromONNX(MatOfByte(m))
    }

    private fun decodeAsset(name: String): Bitmap {
        val b = TestIO.assetBytes(name)
        return BitmapFactory.decodeByteArray(b, 0, b.size)
            ?: throw IllegalArgumentException("解码失败: $name")
    }

    private fun u8Of(b: Bitmap): IntArray {
        val px = IntArray(b.width * b.height)
        b.getPixels(px, 0, b.width, 0, 0, b.width, b.height)
        return px
    }

    private fun toU8(f: FloatArray): IntArray {
        fun u8(v: Float): Int = ((v.coerceIn(0f, 1f) * 255f + 0.5f).toInt())
        return IntArray(f.size / 3) { p ->
            (0xFF shl 24) or (u8(f[3 * p]) shl 16) or
                    (u8(f[3 * p + 1]) shl 8) or u8(f[3 * p + 2])
        }
    }

    /** 与金标 PNG 的 u8 对比（maxΔ / ≤1 占比 / 首个差异像素）。 */
    private fun comparePng(out: Bitmap, pngName: String): JSONObject {
        val ref = decodeAsset("goldens/$pngName")
        val a = u8Of(out); val b = u8Of(ref)
        var maxD = 0; var le1 = 0; var firstBad = -1
        for (i in a.indices) {
            val d = maxOf(
                kotlin.math.abs((a[i] shr 16 and 0xFF) - (b[i] shr 16 and 0xFF)),
                kotlin.math.abs((a[i] shr 8 and 0xFF) - (b[i] shr 8 and 0xFF)),
                kotlin.math.abs((a[i] and 0xFF) - (b[i] and 0xFF)))
            if (d > maxD) maxD = d
            if (d <= 1) le1++
            if (d > 1 && firstBad < 0) firstBad = i
        }
        ref.recycle()
        return JSONObject().put("max_delta", maxD)
            .put("le1_ratio", le1.toDouble() / a.size).put("first_bad", firstBad)
    }

    private fun presetJson(tag: String) = TestIO.assetText("goldens/region_${tag}_preset.json")

    // ---- 1) 金标：region × reinhard/ot × {s=1,0.5} × {protect=1,0.5} ----

    @Test
    fun regionStatGolden() {
        val content = decodeAsset("goldens/stats_content.png")
        val meta = JSONObject(TestIO.assetText("goldens/stats_golden.json"))
            .getJSONObject("region_meta")
        val details = JSONObject()
        for (tag in listOf("lab", "ot")) {
            val model = StatPresets.parse(presetJson(tag))
            assertTrue("$tag 应解析为 RegionStat", model is ContentModel.RegionStat)
            val rs = model as ContentModel.RegionStat
            // 四区齐全（样片四区占比 >0.5%）
            for (k in StatPresets.REGION_KEYS) {
                assertTrue("$tag 的 $k 区应有映射", rs.regions[k] != null)
            }
            for ((s, p) in listOf(1.0f to 1.0f, 0.5f to 1.0f, 1.0f to 0.5f)) {
                val name = "${tag}_s%03d_p%03d".format((s * 100).toInt(), (p * 100).toInt())
                val out = TiledPipeline(model, net)
                    .process(content, ProtectMode.REGION, s, null, p)
                val cmp = comparePng(out, "stats_out_region_$name.png")
                details.put(name, cmp)
                println("[region] $name: maxΔ=${cmp.getInt("max_delta")} " +
                        "≤1占比=${"%.4f".format(cmp.getDouble("le1_ratio"))}")
                out.recycle()
                assertTrue("$name 超容差：$cmp", cmp.getInt("max_delta") <= 2)
            }
            // 占比来自样片侧硬掩码统计——抽查与金标一致（信息锚）
            val goldenRatios = meta.getJSONObject(tag).getJSONObject("ratios")
            for (k in StatPresets.REGION_KEYS) {
                assertEquals("$tag $k 占比不符", goldenRatios.getDouble(k),
                             rs.ratios[k]!!, 1e-6)
            }
        }
        content.recycle()
        report.put("region_golden", details)
        saveReport()
    }

    // ---- 2) 分块 ≡ 整图（段边界 + 类图换算；真机多段图） ----

    @Test
    fun tiledEqualsWhole() {
        // 真实人像放大到多段（w=1152 ⇒ 每段 455 行，h=777 ⇒ 2 段）
        val src = decodeAsset("goldens/protect_probe_portrait.jpg")
        val bmp = Bitmap.createScaledBitmap(src, 1152, 777, true)
        src.recycle()
        val w = bmp.width; val h = bmp.height
        val details = JSONObject()
        for (tag in listOf("lab", "ot")) {
            val model = StatPresets.parse(presetJson(tag))
            val pipe = TiledPipeline(model, net)
            for (protect in listOf(0f, 0.5f, 1f)) {
                for (s in listOf(0.5f, 1f)) {
                    val m = pipe.migrate(bmp, ProtectMode.REGION)
                    try {
                        val rendered = u8Of(pipe.render(m, s, null, protect))
                        val whole = toU8(RegionStatEngine.applyWhole(
                            m.regionStat!!, m.tiled!!, m.src, m.mapped, protect, s))
                        var d = 0; var firstBad = -1
                        for (i in rendered.indices) {
                            if (rendered[i] != whole[i]) { d++; if (firstBad < 0) firstBad = i }
                        }
                        val key = "${tag}_p$protect" + "_s$s"
                        details.put(key, JSONObject().put("px_diff", d))
                        println("[region] $key: tiled vs 整图 u8 差=$d/${rendered.size}")
                        assertTrue("$key 分块≠整图 $d 像素（首个@$firstBad）", d == 0)
                    } finally {
                        m.release()
                    }
                }
            }
        }
        bmp.recycle()
        report.put("tiled_equality", details)
        saveReport()
    }

    // ---- 3) 解析/序列化往返 + 诚实边界 ----

    @Test
    fun parseRoundTripAndEdges() {
        val model = StatPresets.parse(presetJson("lab")) as ContentModel.RegionStat
        val text = StatPresets.toPresetJson(model)
        val round = StatPresets.parse(text) as ContentModel.RegionStat
        assertEquals("luma 往返", model.luma, round.luma)
        assertEquals("基方法 往返", model.baseKind, round.baseKind)
        for (k in StatPresets.REGION_KEYS) {
            assertTrue("$k 是否建映射 往返", (model.regions[k] != null) == (round.regions[k] != null))
        }
        // 再序列化应字节一致（round 口径稳定 ⇒ 往返幂等）
        assertEquals("region 预设序列化往返幂等", text, StatPresets.toPresetJson(round))

        // luma 非 global → 诚实报错
        val bad = JSONObject(presetJson("lab"))
        bad.getJSONObject("params").put("luma", "region")
        try {
            StatPresets.parse(bad.toString())
            throw AssertionError("luma=region 应被拒绝")
        } catch (e: IllegalArgumentException) {
            assertTrue("报错应说明只支持 global", e.message!!.contains("global"))
        }

        // 分区预设基方法为 encoder → StatPresets 拒绝（loadPreset 另行放行）
        try {
            StatPresets.parse(TestIO.assetText("goldens/region_encoder_preset.json"))
            throw AssertionError("encoder 基的分区预设 StatPresets 应拒绝")
        } catch (e: IllegalArgumentException) {
            assertTrue("报错应提示 encoder 基", e.message!!.contains("encoder"))
        }
        report.put("parse_edges", "ok")
        saveReport()
    }

    // ---- 4) region × encoder（路线 0）：嵌套解析放行，零新数学 ----

    @Test
    fun regionEncoderRoute0() {
        val repo = EngineRepository(ctx)
        val s = repo.loadPreset(TestIO.assetText("goldens/region_encoder_preset.json"))
        assertTrue("分区×encoder 应还原为 encoder 会话（各区共享全局映射）",
                   s.model is ContentModel.Encoder)
        assertTrue("该会话 methodId 应为 encoder", s.methodId == "encoder")
        val content = decodeAsset("goldens/p1_content.png")
        val out = TiledPipeline(s.model, net)
            .process(content, ProtectMode.REGION, 1f, null, 1f)
        val cmp = comparePng(out, "p1_out_region100.png")
        println("[region] region×encoder: maxΔ=${cmp.getInt("max_delta")} " +
                "≤1占比=${"%.4f".format(cmp.getDouble("le1_ratio"))}")
        assertTrue("region×encoder 超容差：$cmp", cmp.getInt("max_delta") <= 1)
        out.recycle(); content.recycle()
        report.put("region_encoder", cmp)
        saveReport()
    }

    // ---- 5) 设备端 fit 分区档冒烟 ----

    @Test
    fun fitRegionSmoke() {
        val repo = EngineRepository(ctx)
        val sample = decodeAsset("goldens/protect_probe_portrait.jpg")   // 含人物
        val target = decodeAsset("goldens/stats_content.png")
        val details = JSONObject()
        for (m in listOf("reinhard", "ot")) {
            val s = repo.fitSample(sample, m, region = true)   // 升档形态（合并后 API）
            assertTrue("$m 应产出 RegionStat", s.model is ContentModel.RegionStat)
            assertTrue("$m 应带分区数据", s.isRegion)
            assertEquals("$m 方法 id 应仍是基方法（3 档）", m, s.methodId)
            assertEquals("$m 预设 key 应为基方法 kind",
                         if (m == "reinhard") "lab_stats" else "ot_linear", s.methodKind)
            val out = TiledPipeline(s.model, net)
                .process(target, ProtectMode.REGION, 1f, null, 1f)
            var diff = 0
            val a = u8Of(target); val b = u8Of(out)
            for (i in a.indices) if (a[i] != b[i]) diff++
            println("[region] fitSample $m 交叉 apply: 差 $diff 像素")
            assertTrue("$m 交叉 apply 应产生追色效果", diff > 0)
            out.recycle()
            details.put(m, JSONObject().put("cross_diff", diff))
        }
        try {
            repo.fitSample(sample, "swot")
            throw AssertionError("未知方法应被拒绝")
        } catch (e: IllegalArgumentException) { /* 预期 */ }
        sample.recycle(); target.recycle()
        report.put("fit_region_smoke", details)
        saveReport()
    }

    /**
     * 「分区档 ⊇ plain 档」闸门（2026-09-30 合并的前提）：同一样片、同一 fit，
     * `fitSample(x, m)`（plain）与 `fitSample(x, m, region = true)`（升档）在 **OFF**
     * 模式下出图必须**逐位一致**——分区档的 `params.global` 与 plain 是同一份统计，
     * OFF 走的就是它。有了这条，方法选择器才能只留 3 档算法（"用不用分区"交给
     * 保护模式），且升档不换调子。
     */
    @Test
    fun plainEqualsRegionAtOff() {
        val repo = EngineRepository(ctx)
        val sample = decodeAsset("goldens/protect_probe_portrait.jpg")
        val target = decodeAsset("goldens/stats_content.png")
        val details = JSONObject()
        for (m in listOf("reinhard", "ot")) {
            val plain = repo.fitSample(sample, m)
            val reg = repo.fitSample(sample, m, region = true)
            assertTrue("$m plain 不应带分区数据", !plain.isRegion)
            assertTrue("$m 升档应带分区数据", reg.isRegion)
            val a = u8Of(TiledPipeline(plain.model, net)
                .process(target, ProtectMode.OFF, 1f))
            val b = u8Of(TiledPipeline(reg.model, net)
                .process(target, ProtectMode.OFF, 1f))
            var d = 0
            for (i in a.indices) if (a[i] != b[i]) d++
            details.put(m, JSONObject().put("px_diff", d))
            println("[region] $m plain vs region@OFF: u8 差=$d/${a.size}")
            assertTrue("$m 升档后 OFF 出图≠plain（$d 像素）", d == 0)
        }
        sample.recycle(); target.recycle()
        report.put("plain_region_off_equality", details)
        saveReport()
    }

    /**
     * 无人物样片的皮肤保护关卡（2026-09-30 统一：缺席区一律回退全局，**含皮肤**）。
     *
     * 此前"样片无肤区 ⇒ 皮肤恒原色"让 protect 滑块在该情形下完全失效（0 与 1 都是
     * 原片）；统一后皮肤回退全局映射 ⇒ p=1 仍恒原色、p<1 皮肤掺入样片全局影调/色调。
     * 样片 `sample_portrait_warm`（无人物）、内容 `stats_content.png`（有皮肤）。
     */
    @Test
    fun regionNoSkinSampleProtectsSkin() {
        val content = decodeAsset("goldens/stats_content.png")
        val details = JSONObject()
        for (tag in listOf("lab", "ot")) {
            val model = StatPresets.parse(
                TestIO.assetText("goldens/region_noskin_${tag}_preset.json"))
            assertTrue("$tag 应解析为 RegionStat", model is ContentModel.RegionStat)
            val rs = model as ContentModel.RegionStat
            assertTrue("无人物样片的 skin 区应为 null", rs.regions["skin"] == null)
            assertTrue("无人物样片的 hair 区应为 null", rs.regions["hair"] == null)
            assertTrue("无人物样片的 cloth 区应为 null", rs.regions["cloth"] == null)
            assertTrue("bg 区应存在", rs.regions["bg"] != null)

            val pipe = TiledPipeline(model, net)
            // ① 金标：p ∈ {1, 0.5, 0}（strength=1）
            for (prot in listOf(1.0f, 0.5f, 0.0f)) {
                val name = "%s_p%03d".format(tag, (prot * 100).toInt())
                val out = pipe.process(content, ProtectMode.REGION, 1f, null, prot)
                val cmp = comparePng(out, "stats_out_region_noskin_$name.png")
                details.put(name, cmp)
                println("[region-noskin] $name: maxΔ=${cmp.getInt("max_delta")} " +
                        "≤1占比=${"%.4f".format(cmp.getDouble("le1_ratio"))}")
                out.recycle()
                assertTrue("$name 超容差：$cmp", cmp.getInt("max_delta") <= 2)
            }

            // ② 本地点态：高权重皮肤相对原图的偏离随 protect 单调下降（滑块不再失效）
            val m = pipe.migrate(content, ProtectMode.REGION)
            try {
                val n = m.w * m.h
                val ws = FloatArray(n).also { m.tiled!!.wSkin.get(0, 0, it) }
                val orig = u8Of(content)
                val strong = (0 until n).filter { ws[it] >= 0.7f }
                assertTrue("高权重皮肤像素太少（${strong.size}）", strong.size > 100)
                val dev = listOf(0f, 0.5f, 1f).map { p ->
                    val o = u8Of(pipe.render(m, 1f, null, p))
                    var s = 0.0
                    for (i in strong) s += maxCh(orig[i], o[i])
                    s / strong.size
                }
                details.put("${tag}_skin_dev", JSONObject()
                    .put("p0", dev[0]).put("p05", dev[1]).put("p1", dev[2]))
                println("[region-noskin] $tag 皮肤偏离 p0=${dev[0]} p05=${dev[1]} p1=${dev[2]}")
                assertTrue("$tag protect 未生效（偏离 $dev）",
                           dev[0] > dev[1] && dev[1] > dev[2])
                assertTrue("$tag protect=0 皮肤几乎没动（p0=${dev[0]}）",
                           dev[0] - dev[2] > 3.0)
            } finally {
                m.release()
            }
        }
        content.recycle()
        report.put("region_noskin", details)
        saveReport()
    }

    private fun maxCh(a: Int, b: Int): Int = maxOf(
        kotlin.math.abs((a shr 16 and 0xFF) - (b shr 16 and 0xFF)),
        kotlin.math.abs((a shr 8 and 0xFF) - (b shr 8 and 0xFF)),
        kotlin.math.abs((a and 0xFF) - (b and 0xFF)))

    private fun saveReport() {
        ctx.filesDir.mkdirs()
        ctx.openFileOutput("region_stat_report.json", android.content.Context.MODE_PRIVATE)
            .use { it.write(report.toString().toByteArray()) }
    }
}