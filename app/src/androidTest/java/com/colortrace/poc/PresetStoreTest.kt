package com.colortrace.poc

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.colortrace.engine.ContentModel
import com.colortrace.engine.StatPresets
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import java.io.File

/**
 * 预设库存储与反向序列化（P2.14）。
 *
 * 关卡：
 *  - 统计类预设入库 → 列表/取档/解析往返（数值逐位，无 round 漂移）；
 *  - **encoder 反向序列化**：桌面预设置入库再读回，apply 输出**逐位一致**
 *    （style_thumb PNG 量化往返 + 构造期 8-bit 量化对齐）；
 *  - **设备端 fit 的 encoder 往返**：fit → toPresetJson → 载入 → 输出逐位
 *    （锁 fromSample 的量化：不量化时读回会有亚 1/255 漂移）；
 *  - 损坏容错：bundle 坏了只当该条无效（索引不受影响）；索引坏了按空库处理，不抛；
 *  - 同名覆盖更新 + 删除（条目与文件一起消失）。
 *
 * 测试写的是**真机 app 私有目录**（targetContext.filesDir/presets），
 * 故全部条目用 `T-` 前缀 + 时间戳命名，@After 清理；索引损坏关卡先备份后还原。
 */
@RunWith(AndroidJUnit4::class)
class PresetStoreTest {

    private val ctx get() = InstrumentationRegistry.getInstrumentation().targetContext
    private val created = mutableListOf<String>()

    @Before
    fun setUp() {
        assertTrue("OpenCV native 加载失败", OpenCVLoader.initLocal())
    }

    @After
    fun tearDown() {
        for (id in created) runCatching { PresetStore.delete(ctx, id) }
        created.clear()
    }

    private fun saveUnique(name: String, kind: String, json: String,
                           thumb: ByteArray? = null): PresetStore.Entry =
        PresetStore.save(ctx, name, mapOf(kind to JSONObject(json)), thumb)
            .also { created.add(it.id) }

    @Test
    fun statPresetRoundTrip() {
        val name = "T-stat往返-${System.nanoTime()}"
        val src = TestIO.assetText("goldens/lab_stats_preset.json")
        val e = saveUnique(name, "lab_stats", src)

        val listed = PresetStore.list(ctx).first { it.id == e.id }
        assertEquals(name, listed.name)
        assertEquals(listOf("lab_stats"), listed.methods)
        assertNull("统计类预设不含图像", PresetStore.thumbBitmap(ctx, e.id))

        val parsed = StatPresets.parse(PresetStore.methodJson(ctx, e.id, "lab_stats")!!)
        val orig = StatPresets.parse(src) as ContentModel.LabStats
        assertTrue(parsed is ContentModel.LabStats)
        parsed as ContentModel.LabStats
        for (i in 0 until 3) {
            assertEquals("ref_mean[$i]", orig.refMean[i], parsed.refMean[i], 0.0)
            assertEquals("ref_std[$i]", orig.refStd[i], parsed.refStd[i], 0.0)
        }
        println("[preset-store] stat 往返 ok: ${listed.name}")
    }

    @Test
    fun encoderPresetRoundTripBitExact() {
        val repo = EngineRepository(ctx)
        val src = TestIO.assetText("goldens/encoder_preset.json")
        val session = repo.loadPreset(src)
        val name = "T-encoder往返-${System.nanoTime()}"
        val e = saveUnique(name, "encoder", src, repo.sampleThumbPng(session))

        assertNotNull("encoder 条目应有样片缩略图", PresetStore.thumbBitmap(ctx, e.id))

        val content = TestIO.assetBytes("goldens/p1_content.png")
        val bmp = BitmapFactory.decodeByteArray(content, 0, content.size)
        val before = session.apply(bmp, ProtectMode.OFF, 1f)
        val after = repo.loadPreset(PresetStore.methodJson(ctx, e.id, "encoder")!!)
            .apply(bmp, ProtectMode.OFF, 1f)
        val d = pixelDiff(before, after)
        println("[preset-store] encoder 入库往返 diff=$d")
        assertEquals("encoder 预设入库往返应逐位一致", 0, d)
        before.recycle(); after.recycle(); bmp.recycle()
    }

    @Test
    fun deviceFitEncoderRoundTripBitExact() {
        val repo = EngineRepository(ctx)
        val bytes = TestIO.assetBytes("goldens/protect_probe_portrait.jpg")
        val bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        val s = repo.fitSample(bmp, "encoder")
        val reloaded = repo.loadPreset(s.toPresetJson())
        val out1 = s.apply(bmp, ProtectMode.OFF, 1f)
        val out2 = reloaded.apply(bmp, ProtectMode.OFF, 1f)
        val d = pixelDiff(out1, out2)
        println("[preset-store] 设备端 fit(encoder) → 预设 → 读回 diff=$d")
        assertEquals("设备端 fit 的 encoder 序列化往返应逐位一致", 0, d)
        out1.recycle(); out2.recycle(); bmp.recycle()
    }

    @Test
    fun deviceFitStatRoundTripBitExact() {
        val repo = EngineRepository(ctx)
        val bytes = TestIO.assetBytes("goldens/protect_probe_portrait.jpg")
        val bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        for (m in listOf("reinhard", "ot")) {
            val s = repo.fitSample(bmp, m)
            val reloaded = repo.loadPreset(s.toPresetJson())
            val out1 = s.apply(bmp, ProtectMode.OFF, 1f)
            val out2 = reloaded.apply(bmp, ProtectMode.OFF, 1f)
            val d = pixelDiff(out1, out2)
            println("[preset-store] 设备端 fit($m) → 预设 → 读回 diff=$d")
            assertTrue("$m 往返差 $d（round 量化后应 ≤1）", d <= 1)
            out1.recycle(); out2.recycle()
        }
        bmp.recycle()
    }

    @Test
    fun corruptFilesTolerated() {
        val name = "T-损坏-${System.nanoTime()}"
        val e = saveUnique(name, "lab_stats", TestIO.assetText("goldens/lab_stats_preset.json"))

        // bundle 损坏：该条不可用，但索引仍列出（条目元数据不受影响）
        val bf = PresetStore.bundleFile(ctx, e.id)
        val bundleBackup = bf.readText()
        try {
            bf.writeText("{ 这不是合法 JSON")
            assertNull(PresetStore.load(ctx, e.id))
            assertNull(PresetStore.methodJson(ctx, e.id, "lab_stats"))
            assertTrue(PresetStore.list(ctx).any { it.id == e.id })
        } finally {
            bf.writeText(bundleBackup)
        }

        // 索引损坏：按空库处理，不抛
        val idx = File(ctx.filesDir, "presets/index.json")
        val idxBackup = idx.readText()
        try {
            idx.writeText("不是 JSON")
            assertEquals("索引损坏应按空库处理", 0, PresetStore.list(ctx).size)
        } finally {
            idx.writeText(idxBackup)
            assertTrue(PresetStore.list(ctx).any { it.id == e.id })
        }
        println("[preset-store] 损坏容错 ok")
    }

    @Test
    fun sameNameOverwritesAndDelete() {
        val name = "T-覆盖-${System.nanoTime()}"
        val a = saveUnique(name, "lab_stats", TestIO.assetText("goldens/lab_stats_preset.json"))
        val b = saveUnique(name, "ot_linear", TestIO.assetText("goldens/ot_preset.json"))
        assertEquals("同名应更新同一条目", a.id, b.id)
        assertEquals(1, PresetStore.list(ctx).count { it.name == name })
        assertEquals(listOf("ot_linear"),
                     PresetStore.list(ctx).first { it.id == a.id }.methods)
        assertNull("覆盖后旧档不应残留", PresetStore.methodJson(ctx, a.id, "lab_stats"))
        assertNotNull(PresetStore.methodJson(ctx, a.id, "ot_linear"))

        PresetStore.delete(ctx, a.id)
        created.remove(a.id)
        assertFalse(PresetStore.list(ctx).any { it.id == a.id })
        assertFalse("bundle 文件应随条目删除", PresetStore.bundleFile(ctx, a.id).exists())
        println("[preset-store] 同名覆盖 + 删除 ok")
    }

    @Test
    fun fitAllPresetsShapeAndKinds() {
        val repo = EngineRepository(ctx)
        val bytes = TestIO.assetBytes("goldens/protect_probe_portrait.jpg")
        val bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        val all = repo.fitAllSamplePresets(bmp)
        assertEquals("应产出 3 档（kind 键）",
                     setOf("encoder", "lab_stats", "ot_linear"), all.keys)
        assertEquals("encoder", all["encoder"]!!.methodKind)
        assertFalse("encoder 存 plain（region×encoder 是路线 0 的壳）", all["encoder"]!!.isRegion)
        assertEquals("lab_stats", all["lab_stats"]!!.methodKind)
        assertTrue("reinhard 档存 region 变体（⊇ plain）", all["lab_stats"]!!.isRegion)
        assertEquals("ot_linear", all["ot_linear"]!!.methodKind)
        assertTrue("ot 档存 region 变体（⊇ plain）", all["ot_linear"]!!.isRegion)
        bmp.recycle()
    }

    @Test
    fun fitAllSaveApplyRoundTrip() {
        val repo = EngineRepository(ctx)
        val bytes = TestIO.assetBytes("goldens/protect_probe_portrait.jpg")
        val bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        val all = repo.fitAllSamplePresets(bmp)
        val methods = all.mapValues { JSONObject(it.value.toPresetJson()) }
        val name = "T-fitall-${System.nanoTime()}"
        val entry = PresetStore.save(ctx, name, methods, null).also { created.add(it.id) }

        // 库条目形状：3 档、kind 键（defaultMethodOf 取 encoder 为默认档）
        assertEquals(listOf("encoder", "lab_stats", "ot_linear"), entry.methods)
        assertEquals("encoder", entry.methods.first())

        // 读回每档都能 loadPreset 且 kind 正确（预设会话换档的取档路径）
        for (kind in methods.keys) {
            val text = PresetStore.methodJson(ctx, entry.id, kind)!!
            assertEquals(kind, repo.loadPreset(text).methodKind)
        }

        // encoder 档输出与直接 fit 会话逐位一致
        val reloaded = repo.loadPreset(PresetStore.methodJson(ctx, entry.id, "encoder")!!)
        val out1 = all["encoder"]!!.apply(bmp, ProtectMode.OFF, 1f)
        val out2 = reloaded.apply(bmp, ProtectMode.OFF, 1f)
        val d = pixelDiff(out1, out2)
        println("[preset-store] fitAll→存库→读回 encoder diff=$d")
        assertEquals("encoder 档往返应逐位一致", 0, d)
        out1.recycle(); out2.recycle(); bmp.recycle()
    }

    @Test
    fun savedRegionVariantOffEqualsPlain() {
        // 「分区档 ⊇ plain」穿过序列化路径的闸门：存库读回后的 region 变体在 OFF 档
        // 与 plain 预设出图应逐位一致（global 是同一份统计、同一 round 口径；
        // RegionStat 会话 OFF 走 model.global 的 plain 分派路径）。
        val repo = EngineRepository(ctx)
        val bytes = TestIO.assetBytes("goldens/protect_probe_portrait.jpg")
        val sample = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        val tbytes = TestIO.assetBytes("goldens/stats_content.png")
        val target = BitmapFactory.decodeByteArray(tbytes, 0, tbytes.size)
        val all = repo.fitAllSamplePresets(sample)
        for ((m, kind) in mapOf("reinhard" to "lab_stats", "ot" to "ot_linear")) {
            val regionSaved = repo.loadPreset(all[kind]!!.toPresetJson())
            val plainSaved = repo.loadPreset(repo.fitSample(sample, m).toPresetJson())
            val a = TiledPipeline(plainSaved.model, repo.selfieNet)
                .process(target, ProtectMode.OFF, 1f)
            val b = TiledPipeline(regionSaved.model, repo.selfieNet)
                .process(target, ProtectMode.OFF, 1f)
            val d = pixelDiff(a, b)
            println("[preset-store] $m 存库 region@OFF vs plain@OFF maxCh=$d")
            assertEquals("$m 存库 region 变体 OFF 出图≠plain（maxCh=$d）", 0, d)
            a.recycle(); b.recycle()
        }
        sample.recycle(); target.recycle()
    }

    @Test
    fun exportImportBundleRoundTrip() {
        // 导出 = PresetStore.load 的 bundle 原样文本；导入 = parseImportableBundle。
        // 往返后 name/全部档/缩略图一致，且 encoder 档输出与原会话逐位一致。
        val repo = EngineRepository(ctx)
        val bytes = TestIO.assetBytes("goldens/protect_probe_portrait.jpg")
        val bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        val all = repo.fitAllSamplePresets(bmp)
        val thumb = byteArrayOf(0x89.toByte(), 0x50, 0x4E, 0x47, 0x01, 0x02, 0x03)
        val entry = PresetStore.save(ctx, "T-导出-${System.nanoTime()}",
                                     all.mapValues { JSONObject(it.value.toPresetJson()) },
                                     thumb).also { created.add(it.id) }

        val exported = PresetStore.load(ctx, entry.id)!!.toString()   // ← 落盘文件内容
        val imported = repo.parseImportableBundle(exported, "fallback 名")
        assertEquals("bundle 自带名应优先于 fallback", entry.name, imported.name)
        assertEquals(all.keys, imported.methods.keys)
        assertTrue("缩略图应从 bundle 解回", imported.sampleThumb.contentEquals(thumb))
        assertEquals("手机间互传不应有跳档", emptyList<String>(), imported.skippedKinds)

        // 导入的档可直接 loadPreset 出图，encoder 档与原会话逐位一致
        val reloaded = repo.loadPreset(imported.methods["encoder"]!!.toString())
        val out1 = all["encoder"]!!.apply(bmp, ProtectMode.OFF, 1f)
        val out2 = reloaded.apply(bmp, ProtectMode.OFF, 1f)
        val d = pixelDiff(out1, out2)
        println("[preset-store] 导出→导入 encoder diff=$d")
        assertEquals("导出→导入往返应逐位一致", 0, d)
        out1.recycle(); out2.recycle(); bmp.recycle()
    }

    @Test
    fun importBundleSkipsUnsupportedAndSingleStillWorks() {
        val repo = EngineRepository(ctx)
        val lab = JSONObject(TestIO.assetText("goldens/lab_stats_preset.json"))

        // bundle 含一个可用档 + 一个本机不支持的 kind → 跳过并回报，不静默
        val mixed = JSONObject().put("version", 1).put("name", "T-混合")
            .put("methods", JSONObject().put("lab_stats", lab).put("swot_fake", JSONObject()))
        val imported = repo.parseImportableBundle(mixed.toString(), "fallback")
        assertEquals(setOf("lab_stats"), imported.methods.keys)
        assertEquals(listOf("swot_fake"), imported.skippedKinds)
        assertEquals("T-混合", imported.name)

        // 全部档都不可用 → 报错（不入库）
        val allBad = JSONObject().put("name", "T-全坏")
            .put("methods", JSONObject().put("swot_fake", JSONObject()))
        var threw = false
        try { repo.parseImportableBundle(allBad.toString(), "fallback") }
        catch (e: IllegalArgumentException) { threw = true }
        assertTrue("全不可用档应抛错", threw)

        // 单档预设 JSON（桌面 fit 产物）照常导入——包成单档、用 fallback 名
        val single = repo.parseImportableBundle(lab.toString(), "桌面经典统计")
        assertEquals(mapOf("lab_stats" to lab).keys, single.methods.keys)
        assertEquals("桌面经典统计", single.name)
        assertTrue("单档导入不应有跳档", single.skippedKinds.isEmpty())
    }

    private fun pixelDiff(a: Bitmap, b: Bitmap): Int {
        val n = a.width * a.height
        require(b.width * b.height == n) { "尺寸不一致" }
        val pa = IntArray(n); val pb = IntArray(n)
        a.getPixels(pa, 0, a.width, 0, 0, a.width, a.height)
        b.getPixels(pb, 0, b.width, 0, 0, b.width, b.height)
        var maxCh = 0
        for (i in pa.indices) {
            if (pa[i] == pb[i]) continue
            maxCh = maxOf(maxCh,
                kotlin.math.abs((pa[i] shr 16 and 0xFF) - (pb[i] shr 16 and 0xFF)),
                kotlin.math.abs((pa[i] shr 8 and 0xFF) - (pb[i] shr 8 and 0xFF)),
                kotlin.math.abs((pa[i] and 0xFF) - (pb[i] and 0xFF)))
        }
        return maxCh
    }
}