package com.colortrace.poc

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.colortrace.engine.EncoderEngine
import com.colortrace.engine.EncoderPreset
import com.colortrace.engine.LabConv
import com.colortrace.engine.RegionEngine
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
import org.json.JSONObject

/**
 * 全分辨率分块管线的逐位等价回归（2026-09-28 P2.5）。
 *
 * 关卡：tiled 输出与整图路径（Session.apply 的引擎直调）**FloatArray 逐位相等**——
 * 两者跑的是同一批逐像素点态运算（LUT 查表 / withStrength / 权重融合 / luma
 * 替换 / LAB），分块只是换内存形态，不该有任何数值差异。
 * 整图路径与桌面的等价性由 P1Test 锁住 ⇒ tiled ≡ monolithic ≡ 桌面。
 */
@RunWith(AndroidJUnit4::class)
class TiledTest {

    private lateinit var report: JSONObject
    private lateinit var engine: EncoderEngine
    private lateinit var selfie: Net

    @Before
    fun setUp() {
        assertTrue("OpenCV native 加载失败", OpenCVLoader.initLocal())
        report = JSONObject()
        report.put("device", android.os.Build.MODEL)
        report.put("api", android.os.Build.VERSION.SDK_INT)
        val preset = EncoderPreset(TestIO.assetText("goldens/encoder_preset.json"))
        engine = EncoderEngine(preset, TestIO.assetBytes(preset.onnxName))
        selfie = readNetFromAsset("selfie_multiclass.onnx")
    }

    private fun readNetFromAsset(name: String): Net {
        val bytes = TestIO.assetBytes(name)
        val m = Mat(1, bytes.size, CvType.CV_8U)
        m.put(0, 0, bytes)
        return Dnn.readNetFromONNX(MatOfByte(m))
    }

    /** 确定性测试图：渐变天 + 肤色块 + 暗前景（覆盖高光/肤色/暗部三域）。 */
    private fun makeBitmap(w: Int, h: Int): Bitmap {
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val row = IntArray(w)
        for (y in 0 until h) {
            for (x in 0 until w) {
                val fy = y.toFloat() / h
                val fx = x.toFloat() / w
                // 天空渐变（蓝，亮度随 y 变化）+ 中部肤色椭圆 + 底部暗带
                val r = (40 + 160 * fx * (1 - fy)).toInt().coerceIn(0, 255)
                val g = (90 + 60 * fy).toInt().coerceIn(0, 255)
                val b = (200 - 80 * fy).toInt().coerceIn(0, 255)
                row[x] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
            // 肤色椭圆（中心 0.5,0.45 半径 0.3/0.25）盖在渐变上
            for (x in 0 until w) {
                val dx = (x - w * 0.5f) / (w * 0.3f)
                val dy = (y - h * 0.45f) / (h * 0.25f)
                if (dx * dx + dy * dy < 1f) {
                    row[x] = (0xFF shl 24) or (226 shl 16) or (172 shl 8) or 148
                }
            }
            if (y > h * 0.85f) {
                for (x in 0 until w) {
                    val v = 30 + (x * 40 / w)
                    row[x] = (0xFF shl 24) or (v shl 16) or (v shl 8) or (v + 6)
                }
            }
            bmp.setPixels(row, 0, w, 0, y, w, 1)
        }
        return bmp
    }

    private fun u8Of(bmp: Bitmap): IntArray {
        val px = IntArray(bmp.width * bmp.height)
        bmp.getPixels(px, 0, bmp.width, 0, 0, bmp.width, bmp.height)
        return px
    }

    @Test
    fun tiledEqualsMonolithic() {
        val w = 1024
        val h = 768            // 非方形，专抓行段换算错位
        val bmp = makeBitmap(w, h)
        val rgb = bitmapToRgbF32(bmp).let { Triple(it.first, it.second, it.third) }
        val region = RegionEngine(engine, selfie, 1.0f)

        val cases = listOf(
            Triple("plain", ProtectMode.OFF, 0.5f),
            Triple("chroma", ProtectMode.CHROMA, 0.5f),
            Triple("region", ProtectMode.REGION, 0.5f),
            Triple("region_s100", ProtectMode.REGION, 1.0f),
            Triple("off_s100", ProtectMode.OFF, 1.0f),
        )
        val details = JSONObject()
        for ((name, mode, s) in cases) {
            val mono = when (mode) {
                ProtectMode.OFF -> engine.apply(rgb.first, w, h, s)
                ProtectMode.CHROMA -> engine.applyChroma(rgb.first, w, h, 1.0f, s)
                ProtectMode.REGION -> region.apply(rgb.first, w, h, 1.0f, s)
            }
            // 比较口径：两边都落到 Bitmap 的 u8（产品输出就是 Bitmap），
            // 否则单边量化会引入 ≤0.5/255 的假差
            val monoU8 = u8Of(f32ToBitmap(mono, w, h))
            val tiledBmp = TiledPipeline(engine, selfie).process(bmp, mode, s)
            val tiledU8 = u8Of(tiledBmp)
            var diffs = 0
            var firstBad = -1
            for (i in monoU8.indices) {
                if (monoU8[i] != tiledU8[i]) { diffs++; if (firstBad < 0) firstBad = i }
            }
            details.put(name, JSONObject().put("px_diff", diffs))
            println("[tiled] $name: u8 像素差=$diffs/${monoU8.size}")
            assertTrue("$name tiled≠monolithic $diffs 个像素不同（应为 0，首个@$firstBad）",
                       diffs == 0)
        }

        // 分割不可用回退：tiled(region, net=null) ≡ engine.applyChroma（同 protect=1）
        val monoFallback = engine.applyChroma(rgb.first, w, h, 1.0f, 0.5f)
        val tiledFallback = TiledPipeline(engine, null).process(bmp, ProtectMode.REGION, 0.5f)
        val monoFallbackU8 = u8Of(f32ToBitmap(monoFallback, w, h))
        val fallbackU8 = u8Of(tiledFallback)
        var fallbackDiffs = 0
        for (i in monoFallbackU8.indices) {
            if (monoFallbackU8[i] != fallbackU8[i]) fallbackDiffs++
        }
        details.put("region_fallback", JSONObject().put("px_diff", fallbackDiffs))
        println("[tiled] region_fallback: u8 像素差=$fallbackDiffs")
        assertTrue("fallback tiled≠monolithic $fallbackDiffs 个像素不同（应为 0）",
                   fallbackDiffs == 0)

        // develop 集成（P2.6）：微调叠在自动追色后，分块与整图链路逐位一致
        val devParams = mapOf("exposure" to 0.75f, "contrast" to 28f,
                              "shadows" to 42f, "temp" to 32f, "vibrance" to 38f,
                              "split_high_sat" to 36f, "gamma" to 1.15f)
        val monoDevelop = com.colortrace.engine.DevelopTransform(devParams)
            .apply(engine.apply(rgb.first, w, h, 0.5f))
        val tiledDevelop = TiledPipeline(engine, selfie)
            .process(bmp, ProtectMode.OFF, 0.5f, devParams)
        val monoDevU8 = u8Of(f32ToBitmap(monoDevelop, w, h))
        val devU8 = u8Of(tiledDevelop)
        var devDiffs = 0
        for (i in monoDevU8.indices) {
            if (monoDevU8[i] != devU8[i]) devDiffs++
        }
        details.put("develop_chain", JSONObject().put("px_diff", devDiffs))
        println("[tiled] develop_chain: u8 像素差=$devDiffs")
        assertTrue("develop tiled≠monolithic $devDiffs 个像素不同（应为 0）",
                   devDiffs == 0)

        report.put("equality", details)
        saveReport()
    }

    /**
     * 两阶段拆分回归（P2.7）：`render(migrate(...))` 必须与一次性 `process(...)`
     * **逐位相同**（u8 0 差异），且同一个 Migration 连续 render 不同 strength
     * 仍逐位——这是「微调/强度只叠加、不重算迁移」的正确性关卡。
     */
    @Test
    fun splitRenderEqualsProcess() {
        val w = 1024
        val h = 768
        val bmp = makeBitmap(w, h)
        val devParams = mapOf("exposure" to 0.75f, "contrast" to 28f,
                              "shadows" to 42f, "temp" to 32f, "vibrance" to 38f,
                              "split_high_sat" to 36f, "gamma" to 1.15f)
        val cases = listOf(
            Triple("plain_s050", ProtectMode.OFF, 0.5f),
            Triple("plain_s100", ProtectMode.OFF, 1.0f),
            Triple("chroma_s050", ProtectMode.CHROMA, 0.5f),
            Triple("chroma_s025", ProtectMode.CHROMA, 0.25f),
            Triple("region_s050", ProtectMode.REGION, 0.5f),
            Triple("region_s100", ProtectMode.REGION, 1.0f),
        )
        val details = JSONObject()
        for ((name, mode, s) in cases) {
            val altS = 0.25f
            val refS = u8Of(TiledPipeline(engine, selfie).process(bmp, mode, s))
            val refAlt = u8Of(TiledPipeline(engine, selfie).process(bmp, mode, altS))
            val pipe = TiledPipeline(engine, selfie)
            val m = pipe.migrate(bmp, mode)     // 迁移只做一次
            try {
                val gotS = u8Of(pipe.render(m, s))
                val gotAlt = u8Of(pipe.render(m, altS))   // 同一 Migration 换强度
                var d1 = 0; var d2 = 0
                for (i in refS.indices) {
                    if (refS[i] != gotS[i]) d1++
                    if (refAlt[i] != gotAlt[i]) d2++
                }
                details.put(name, JSONObject().put("split_diff", d1)
                    .put("reuse_diff", d2))
                println("[split] $name: split差=$d1 reuse差=$d2 (${refS.size}px)")
                assertTrue("$name render(migrate)≠process: $d1 像素", d1 == 0)
                assertTrue("$name 同 Migration 换 strength≠process: $d2 像素", d2 == 0)
            } finally {
                m.release()
            }
        }

        // develop 叠加：缓存迁移后只跑 render 也必须与一次性 process 逐位一致
        val pipeDev = TiledPipeline(engine, selfie)
        val mDev = pipeDev.migrate(bmp, ProtectMode.OFF)
        try {
            val refDev = u8Of(TiledPipeline(engine, selfie)
                .process(bmp, ProtectMode.OFF, 0.5f, devParams))
            val gotDev = u8Of(pipeDev.render(mDev, 0.5f, devParams))
            var dd = 0
            for (i in refDev.indices) if (refDev[i] != gotDev[i]) dd++
            details.put("develop_split", JSONObject().put("px_diff", dd))
            println("[split] develop: split差=$dd")
            assertTrue("develop render(migrate)≠process: $dd 像素", dd == 0)
        } finally {
            mDev.release()
        }
        report.put("split_equality", details)
        saveReport()
    }

    /**
     * 保护强度参数化回归：`protect` 只在**出图期**生效（不进迁移缓存），
     * 因此 ① render 的 protect 必须与整图 `region.apply(...,protect,...)` 逐位
     * 一致；② 同一个 Migration 连续换 protect 仍逐位（锁"不进 key"的正确性）。
     */
    @Test
    fun protectStrengthEqualsMonolithic() {
        val w = 1024
        val h = 768
        val bmp = makeBitmap(w, h)
        val rgb = bitmapToRgbF32(bmp)
        val region = RegionEngine(engine, selfie, 1.0f)
        val details = JSONObject()
        for (protect in listOf(0.0f, 0.5f, 1.0f)) {
            val mono = region.apply(rgb.first, w, h, protect, 0.5f)
            val monoU8 = u8Of(f32ToBitmap(mono, w, h))
            val pipe = TiledPipeline(engine, selfie)
            val m = pipe.migrate(bmp, ProtectMode.REGION)
            try {
                val gotU8 = u8Of(pipe.render(m, 0.5f, null, protect))
                var d = 0
                for (i in monoU8.indices) if (monoU8[i] != gotU8[i]) d++
                details.put("protect_$protect", JSONObject().put("px_diff", d))
                println("[protect] $protect: region u8 像素差=$d")
                assertTrue("protect=$protect tiled≠monolithic: $d 像素", d == 0)
            } finally {
                m.release()
            }
        }

        // 同一 Migration 换 protect：与各自的一次性 process 逐位一致
        val pipe = TiledPipeline(engine, selfie)
        val m = pipe.migrate(bmp, ProtectMode.REGION)
        try {
            var reuse = 0
            for (protect in listOf(1.0f, 0.3f)) {
                val ref = u8Of(TiledPipeline(engine, selfie)
                    .process(bmp, ProtectMode.REGION, 0.5f, null, protect))
                val got = u8Of(pipe.render(m, 0.5f, null, protect))
                for (i in ref.indices) if (ref[i] != got[i]) reuse++
            }
            details.put("reuse_protect", JSONObject().put("px_diff", reuse))
            println("[protect] reuse: u8 像素差=$reuse")
            assertTrue("同 Migration 换 protect 不一致: $reuse 像素", reuse == 0)
        } finally {
            m.release()
        }

        // chroma 档同样参数化（否则 chroma 的保护强度滑块无效）
        val pipeC = TiledPipeline(engine, selfie)
        val mC = pipeC.migrate(bmp, ProtectMode.CHROMA)
        try {
            val ref = u8Of(f32ToBitmap(
                engine.applyChroma(rgb.first, w, h, 0.4f, 0.5f), w, h))
            val got = u8Of(pipeC.render(mC, 0.5f, null, 0.4f))
            var d = 0
            for (i in ref.indices) if (ref[i] != got[i]) d++
            details.put("chroma_protect", JSONObject().put("px_diff", d))
            println("[protect] chroma 0.4: u8 像素差=$d")
            assertTrue("chroma protect 不一致: $d 像素", d == 0)
        } finally {
            mC.release()
        }
        report.put("protect_strength", details)
        saveReport()
    }

    /**
     * 真实人像上的保护**可见性**关卡（P2.11 用户反馈"拖滑块没区别"后的量化回归
     * 护栏）：等价性关卡只锁 tiled ≡ monolithic，锁不住"protect 有没有视觉
     * 效果"。在 `protect_probe_portrait.jpg`（开源授权人像样片 768 缩版，已在库）
     * 的高权重皮肤像素上断言 protect 满拉（1→0）的输出差超过可见阈值。量级结论
     * （2026-10-08 实测，样片对 = oc14 × t13，模拟器 x86_64）：
     *  - REGION ws≥0.7（75154px）strength=1 时 mean≈16.3 / max≈33 —— 断言 ≥8 / ≥16
     *    （样片对色调差比旧素材温和，绝对差小于旧标定 43，但相对安全带同口径 ~0.5×）；
     *  - CHROMA（SkinWeight 路径，与分割无关）——阈值见下方断言；
     *  - strength=0.5（默认）下效果减半 ——**感知温和是物理事实**（迁移先被强度
     *    砍半 + 高权重皮肤只有脸），不是 bug。
     * 合成图（平坦色块椭圆）不能当皮肤探针——分割网对它给出的 wSkin≈0。
     * 注意：阈值随样片对（样片×内容）的迁移幅度走，换素材需重新实测标定。
     */
    @Test
    fun protectVisibleOnRealPortrait() {
        val bytes = TestIO.assetBytes("goldens/protect_probe_portrait.jpg")
        val bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        assertTrue("人像探针解码失败", bmp != null && bmp.width > 256)
        val w = bmp!!.width; val h = bmp.height
        val n3 = w * h * 3
        val details = JSONObject().put("w", w).put("h", h)

        // ---- REGION ----
        val pipe = TiledPipeline(engine, selfie)
        val m = pipe.migrate(bmp, ProtectMode.REGION)
        try {
            // wSkin 分位：看模型到底给了多少皮肤
            val n = w * h
            val ws = FloatArray(n)
            m.tiled!!.wSkin.get(0, 0, ws)
            val sorted = ws.copyOf(); sorted.sort()
            fun pct(p: Double) = sorted[((n - 1) * p).toInt()]
            details.put("ws_p50", pct(0.50).toDouble())
            details.put("ws_p90", pct(0.90).toDouble())
            details.put("ws_p99", pct(0.99).toDouble())
            println("[pv] wSkin 分位 p50=${pct(0.50)} p90=${pct(0.90)} p99=${pct(0.99)}")

            // 顶部 1/4 wSkin 的像素集合（蒙版显示最绿的地方）
            val th = sorted[((n - 1) * 0.75).toInt()]
            val skinIdx = HashSet<Int>()
            for (i in 0 until n) if (ws[i] >= th) skinIdx.add(i)
            details.put("top_quartile_threshold", th.toDouble())

            val p1 = u8Of(pipe.render(m, 0.5f, null, 1f))
            val p0 = u8Of(pipe.render(m, 0.5f, null, 0f))
            var sum = 0.0; var mx = 0
            for (i in skinIdx) {
                val va = p1[i]; val vb = p0[i]
                val d = maxOf(kotlin.math.abs((va shr 16 and 0xFF) - (vb shr 16 and 0xFF)),
                              kotlin.math.abs((va shr 8 and 0xFF) - (vb shr 8 and 0xFF)),
                              kotlin.math.abs((va and 0xFF) - (vb and 0xFF)))
                sum += d; if (d > mx) mx = d
            }
            val mean = sum / skinIdx.size
            println("[pv] 真实人像 REGION 顶部皮肤 protect 1 vs 0: mean=$mean max=$mx")
            details.put("region_mean", mean); details.put("region_max", mx.toDouble())

            // 高权重真皮肤（ws>=0.7，蒙版深绿处）+ strength=1（最大迁移）下的量级
            val strong = ArrayList<Int>()
            for (i in 0 until n) if (ws[i] >= 0.7f) strong.add(i)
            details.put("strong_skin_px", strong.size)
            val p1s = u8Of(pipe.render(m, 1f, null, 1f))
            val p0s = u8Of(pipe.render(m, 1f, null, 0f))
            var sum2 = 0.0; var mx2 = 0
            for (i in strong) {
                val va = p1s[i]; val vb = p0s[i]
                val d = maxOf(kotlin.math.abs((va shr 16 and 0xFF) - (vb shr 16 and 0xFF)),
                              kotlin.math.abs((va shr 8 and 0xFF) - (vb shr 8 and 0xFF)),
                              kotlin.math.abs((va and 0xFF) - (vb and 0xFF)))
                sum2 += d; if (d > mx2) mx2 = d
            }
            val meanS = if (strong.isEmpty()) 0.0 else sum2 / strong.size
            println("[pv] 真实人像 REGION ws>=0.7 (${strong.size}px) strength=1 protect 1 vs 0: " +
                            "mean=$meanS max=$mx2")
            details.put("region_strong_s1_mean", meanS)
            details.put("region_strong_s1_max", mx2.toDouble())
            assertTrue("REGION protect 在真皮肤上无视觉差（s=1 mean=$meanS max=$mx2）" +
                               "——保护数学失效", strong.isNotEmpty() && meanS >= 8 && mx2 >= 16)
        } finally { m.release() }

        // ---- CHROMA（SkinWeight.of 路径，与分割无关）----
        val mC = TiledPipeline(engine, selfie).migrate(bmp, ProtectMode.CHROMA)
        try {
            val p1 = u8Of(TiledPipeline(engine, selfie).render(mC, 0.5f, null, 1f))
            val p0 = u8Of(TiledPipeline(engine, selfie).render(mC, 0.5f, null, 0f))
            // chroma 没有显式 mask，用整图 |Δ| 的 p95 看皮肤锚定的量级
            val diffs = IntArray(w * h)
            for (i in p1.indices) {
                val va = p1[i]; val vb = p0[i]
                diffs[i] = maxOf(kotlin.math.abs((va shr 16 and 0xFF) - (vb shr 16 and 0xFF)),
                                 kotlin.math.abs((va shr 8 and 0xFF) - (vb shr 8 and 0xFF)),
                                 kotlin.math.abs((va and 0xFF) - (vb and 0xFF)))
            }
            val sd = diffs.copyOf(); sd.sort()
            val p95 = sd[((w * h - 1) * 0.95).toInt()]
            var sum = 0.0
            for (d in diffs) sum += d
            val mean = sum / diffs.size
            println("[pv] 真实人像 CHROMA protect 1 vs 0: mean=$mean p95=$p95")
            details.put("chroma_mean", mean); details.put("chroma_p95", p95.toDouble())

            // 高肤色隶属度（SkinWeight>=0.7）像素处的量级 + strength=1 版本
            // labIn 需要源图：从迁移里拿 src
            val srcF = FloatArray(n3)
            mC.src.get(0, 0, srcF)
            val wgt = com.colortrace.engine.SkinWeight.of(
                com.colortrace.engine.LabConv.rgbToLab(srcF))
            val strong = ArrayList<Int>()
            for (i in 0 until w * h) if (wgt[i] >= 0.7f) strong.add(i)
            details.put("chroma_strong_px", strong.size)
            var sum3 = 0.0; var mx3 = 0
            for (i in strong) {
                val va = p1[i]; val vb = p0[i]
                val d = maxOf(kotlin.math.abs((va shr 16 and 0xFF) - (vb shr 16 and 0xFF)),
                              kotlin.math.abs((va shr 8 and 0xFF) - (vb shr 8 and 0xFF)),
                              kotlin.math.abs((va and 0xFF) - (vb and 0xFF)))
                sum3 += d; if (d > mx3) mx3 = d
            }
            val meanC = if (strong.isEmpty()) 0.0 else sum3 / strong.size
            println("[pv] 真实人像 CHROMA w>=0.7 (${strong.size}px) s=0.5 protect 1 vs 0: " +
                            "mean=$meanC max=$mx3")
            details.put("chroma_strong_mean", meanC)
            details.put("chroma_strong_max", mx3.toDouble())
            val p1s = u8Of(TiledPipeline(engine, selfie).render(mC, 1f, null, 1f))
            val p0s = u8Of(TiledPipeline(engine, selfie).render(mC, 1f, null, 0f))
            var sum4 = 0.0; var mx4 = 0
            for (i in strong) {
                val va = p1s[i]; val vb = p0s[i]
                val d = maxOf(kotlin.math.abs((va shr 16 and 0xFF) - (vb shr 16 and 0xFF)),
                              kotlin.math.abs((va shr 8 and 0xFF) - (vb shr 8 and 0xFF)),
                              kotlin.math.abs((va and 0xFF) - (vb and 0xFF)))
                sum4 += d; if (d > mx4) mx4 = d
            }
            val meanC1 = if (strong.isEmpty()) 0.0 else sum4 / strong.size
            println("[pv] 真实人像 CHROMA w>=0.7 strength=1 protect 1 vs 0: mean=$meanC1 max=$mx4")
            details.put("chroma_strong_s1_mean", meanC1)
            details.put("chroma_strong_s1_max", mx4.toDouble())
            assertTrue("CHROMA protect 在高隶属皮肤上无视觉差（s=1 mean=$meanC1 max=$mx4）" +
                               "——保护数学失效", strong.isNotEmpty() && meanC1 >= 8 && mx4 >= 20)
        } finally { mC.release() }
        report.put("protect_probe_portrait", details)
        saveReport()
    }

    /**
     * 流式 JPEG 导出等价关卡（P2.12）：`processToJpeg`（段直写 BGR Mat →
     * imencode）必须与旧链路（process 出 Bitmap → bitmapToMat → cvtColor →
     * imencode）产物**逐字节一致**。旧链路正被 ExportEncodeTest 锁 4:4:4，
     * 本关卡通过 ⇒ 流式产物同为 4:4:4 同字节。
     */
    @Test
    fun jpegStreamEqualsBitmapPath() {
        val w = 512
        val h = 384            // 非整段对齐：抓最后不完整段的缓冲处理
        val devParams = mapOf("exposure" to 0.75f, "contrast" to 28f,
                              "shadows" to 42f, "temp" to 32f, "vibrance" to 38f)
        val cases = listOf(
            JpegCase("plain", ProtectMode.OFF, 0.5f, null),
            JpegCase("region", ProtectMode.REGION, 0.8f, null),
            JpegCase("develop", ProtectMode.OFF, 0.5f, devParams),
        )
        val details = JSONObject()
        for (c in cases) {
            val refBmp = makeBitmap(w, h)
            val ref = encodeJpeg444(
                TiledPipeline(engine, selfie).process(refBmp, c.mode, c.s, c.dev))
            refBmp.recycle()
            val streamBmp = makeBitmap(w, h)   // processToJpeg 接管并回收它
            val got = TiledPipeline(engine, selfie)
                .processToJpeg(streamBmp, c.mode, c.s, c.dev).bytes
            var same = ref.size == got.size
            var firstDiff = -1
            if (same) for (i in ref.indices) { if (ref[i] != got[i]) { same = false; firstDiff = i; break } }
            details.put(c.name, JSONObject().put("bytes_equal", same)
                .put("ref_len", ref.size).put("got_len", got.size))
            println("[jpeg] ${c.name}: 流式 vs Bitmap 路径 字节一致=$same " +
                    "ref=${ref.size}B got=${got.size}B firstDiff=$firstDiff")
            // 解码回读对比：像素级一致 ⇒ 差异只在 JPEG 头/编码器状态
            val dref = BitmapFactory.decodeByteArray(ref, 0, ref.size)
            val dgot = BitmapFactory.decodeByteArray(got, 0, got.size)
            var pxDiff = -1
            if (dref != null && dgot != null && dref.width == dgot.width &&
                dref.height == dgot.height) {
                val pa = IntArray(dref.width * dref.height)
                val pb = IntArray(dgot.width * dgot.height)
                dref.getPixels(pa, 0, dref.width, 0, 0, dref.width, dref.height)
                dgot.getPixels(pb, 0, dgot.width, 0, 0, dgot.width, dgot.height)
                for (i in pa.indices) { if (pa[i] != pb[i]) { pxDiff = i; break } }
            }
            details.put(c.name, details.getJSONObject(c.name)
                .put("decoded_pxDiff", pxDiff))
            println("[jpeg] ${c.name}: 解码回读首个像素差=$pxDiff")
            dref?.recycle(); dgot?.recycle()
            assertTrue("${c.name} 流式 JPEG 与 Bitmap 中转产物不一致 " +
                    "(ref=${ref.size} got=${got.size} firstByte=$firstDiff decodedPx=$pxDiff)", same)
        }
        report.put("jpeg_stream_equality", details)
        saveReport()
    }

    private data class JpegCase(
        val name: String, val mode: ProtectMode, val s: Float,
        val dev: Map<String, Float>?)

    @Test
    fun tiledLargeSmoke() {
        // 12MP smoke：验证段循环在大图上完整跑通、进度回调与输出尺寸正确
        val w = 4032
        val h = 3024
        val bmp = makeBitmap(w, h)
        var progressCalls = 0
        var lastTotal = 0
        val t0 = System.currentTimeMillis()
        val out = TiledPipeline(engine, selfie).process(
            bmp, ProtectMode.REGION, 0.5f) { _, total ->
            progressCalls++
            lastTotal = total
        }
        val ms = System.currentTimeMillis() - t0
        println("[tiled] 12MP smoke: ${out.width}x${out.height} in ${ms}ms " +
                "($progressCalls 次进度, 总段 $lastTotal)")
        assertTrue("输出尺寸错误", out.width == w && out.height == h)
        assertTrue("进度回调缺失", progressCalls >= 2 && lastTotal >= 2)
        report.put("smoke_12mp_ms", ms.toDouble())
        report.put("smoke_bands", lastTotal)
        saveReport()
    }

    private fun saveReport() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        ctx.filesDir.mkdirs()
        ctx.openFileOutput("tiled_report.json", android.content.Context.MODE_PRIVATE)
            .use { it.write(report.toString().toByteArray()) }
    }
}
