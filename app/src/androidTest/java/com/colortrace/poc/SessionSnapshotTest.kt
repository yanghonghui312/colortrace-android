package com.colortrace.poc

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.colortrace.engine.DevelopSpec
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertArrayEquals
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
 * 会话快照与恢复（P2.20）：覆盖安装 / 被杀 / 系统回收后「继续上次编辑」靠
 * `files/session_snapshot.json` + [SessionRestore]。
 *
 * 关卡（这是本功能的**核心断言**——恢复出来的会话必须与原来那个是同一个）：
 *  1. 快照全字段 save/load 往返一致；损坏 / 无来源（无从重建）必须落回 null 并清除；
 *  2. **参考图会话**：同一张参考图 refit 出来的会话，`toPresetJson()` **逐字节相同**
 *     ——并且用分块管线各出一张图，像素**逐位相同**（不是"看起来一样"）；
 *  3. **分区档**：快照 protection=REGION 的统计档会话，恢复后 `isRegion=true`
 *     （否则恢复出来就切不回分区保护）；同一张样片、保护档 OFF 时则**不**升档；
 *  4. **预设会话**：按库条目 id 把全部档带回来（方法 pill 可换）；条目被删时
 *     单档照常恢复（不静默丢整段现场）；
 *  5. 照片 uri 读不到时**整项跳过**，不留孤儿索引（activeIndex 收到有效范围）。
 *
 * ⚠️ 测试写的是真机 app 私有目录，会覆盖用户真实会话快照——这里先备份、跑完还原。
 */
@RunWith(AndroidJUnit4::class)
class SessionSnapshotTest {

    private val ctx get() = InstrumentationRegistry.getInstrumentation().targetContext
    private lateinit var repo: EngineRepository
    private var backup: ByteArray? = null
    private val createdPresets = mutableListOf<String>()

    @Before
    fun setUp() {
        assertTrue("OpenCV native 加载失败", OpenCVLoader.initLocal())
        repo = EngineRepository(ctx)
        val f = File(ctx.filesDir, "session_snapshot.json")
        backup = if (f.exists()) f.readBytes() else null
        SessionSnapshotStore.clear(ctx)
    }

    @After
    fun tearDown() {
        SessionSnapshotStore.clear(ctx)
        backup?.let { File(ctx.filesDir, "session_snapshot.json").writeBytes(it) }
        File(ctx.filesDir, "T-snapshot-sample.png").delete()   // 测试自己造的样片，自己收
        for (id in createdPresets) runCatching { PresetStore.delete(ctx, id) }
        createdPresets.clear()
    }

    // ---- 工具 ----

    /** 把金标 PNG 落成一张真文件，并返回可被 ContentResolver 读的 file:// uri。 */
    private fun sampleUri(name: String = "goldens/p1_content.png"): String {
        val bytes = TestIO.assetBytes(name)
        val f = File(ctx.filesDir, "T-snapshot-sample.png")
        f.writeBytes(bytes)
        return Uri.fromFile(f).toString()
    }

    private fun assetBitmap(name: String = "goldens/p1_content.png"): Bitmap {
        val bytes = TestIO.assetBytes(name)
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }

    private fun snap(
        sampleUri: String?,
        presetJson: String? = null,
        presetId: String? = null,
        presetName: String? = null,
        method: String = "encoder",
        photoUris: List<String> = emptyList(),
        activeIndex: Int = -1,
        strength: Float = 0.5f,
        protect: String = ProtectMode.OFF.name,
        protectStrength: Float = 1f,
        developEnabled: Boolean = false,
        devValues: Map<String, Float> = DevelopSpec.DEFAULTS,
    ) = SessionSnapshotStore.Snapshot(
        sampleUri = sampleUri, presetId = presetId, presetName = presetName,
        presetJson = presetJson, method = method, photoUris = photoUris,
        activeIndex = activeIndex, strength = strength, protect = protect,
        protectStrength = protectStrength, developEnabled = developEnabled,
        devValues = devValues)

    private fun pixels(b: Bitmap): IntArray =
        IntArray(b.width * b.height).also { b.getPixels(it, 0, b.width, 0, 0, b.width, b.height) }

    // ---- 1. 存储往返 ----

    @Test
    fun roundTripAllFieldsAndCorruptFallback() {
        assertNull("初始不应有快照", SessionSnapshotStore.load(ctx))

        val s = snap(sampleUri = "content://media/picker/test/1", method = "reinhard",
                     photoUris = listOf("u://0", "u://1", "u://2"), activeIndex = 2,
                     strength = 0.75f, protect = ProtectMode.REGION.name,
                     protectStrength = 0.4f, developEnabled = true,
                     devValues = DevelopSpec.DEFAULTS + mapOf("exposure" to 0.33f))
        SessionSnapshotStore.save(ctx, s)
        assertEquals("全字段往返必须一致", s, SessionSnapshotStore.load(ctx))

        // 预设来源（sampleUri 与 presetJson 二选一）
        val p = s.copy(sampleUri = null, presetJson = "{\"kind\":\"lab_stats\"}",
                       presetId = "abc", presetName = "T-预设")
        SessionSnapshotStore.save(ctx, p)
        assertEquals(p, SessionSnapshotStore.load(ctx))

        SessionSnapshotStore.clear(ctx)
        assertNull(SessionSnapshotStore.load(ctx))
        assertFalse(File(ctx.filesDir, "session_snapshot.json").exists())

        // 损坏：返回 null 且顺手删除（不能让坏状态卡在启动路径上）
        File(ctx.filesDir, "session_snapshot.json").writeText("{oops")
        assertNull(SessionSnapshotStore.load(ctx))
        assertFalse("损坏文件应被清除", File(ctx.filesDir, "session_snapshot.json").exists())

        // 无来源：字段齐全但既没参考图也没预设 JSON —— 无从重建，视为无效
        SessionSnapshotStore.save(ctx, snap(sampleUri = null, presetJson = null))
        assertNull("无来源的快照不算快照", SessionSnapshotStore.load(ctx))
        assertFalse(File(ctx.filesDir, "session_snapshot.json").exists())
    }

    // ---- 2. 参考图会话：恢复出来的必须与原来那个是同一个 ----

    @Test
    fun restoredSampleSessionIsBitIdentical() = runBlocking {
        val uri = sampleUri()
        val img = assetBitmap()
        val original = repo.fitSample(img, "encoder")

        SessionSnapshotStore.save(ctx, snap(sampleUri = uri, photoUris = listOf(uri),
                                            activeIndex = 0, method = "encoder"))
        val loaded = SessionSnapshotStore.load(ctx)!!
        assertEquals("先确认存进去的东西是它", uri, loaded.sampleUri)

        val r = SessionRestore.rebuild(repo, ctx, loaded)
        assertEquals("encoder", r.session.methodId)
        assertEquals("照片项必须原样回来", 1, r.photos.size)
        assertEquals(0, r.activeIndex)
        assertNotNull("参考图会话要有来源缩略图（来源面板显示用）", r.sampleThumbPng)

        // ① 预设序列化逐字节相同（Θs + 样片缩略图都在里面）
        assertEquals("refit 出来的预设必须与原来逐字节相同",
                     original.toPresetJson(), r.session.toPresetJson())

        // ② 真出一张图：像素逐位相同（这才是"恢复成功"的定义）
        val a = TiledPipeline(original.model, null).process(img, ProtectMode.OFF, 0.5f)
        val b = TiledPipeline(r.session.model, null).process(img, ProtectMode.OFF, 0.5f)
        assertEquals(a.width, b.width)
        assertArrayEquals("恢复前后出图必须逐位相同", pixels(a), pixels(b))
        println("[snapshot] 参考图会话恢复：预设 JSON 逐字节相同、出图逐位相同 " +
                "(${a.width}x${a.height})")
    }

    // ---- 3. 分区档：保护档决定要不要带分区数据 ----

    @Test
    fun regionSnapshotRestoresWithRegionData() = runBlocking {
        val uri = sampleUri()
        // REGION + 统计档：必须 region=true 升档，否则恢复后 isRegion=false、切不回分区保护
        SessionSnapshotStore.save(ctx, snap(sampleUri = uri, method = "reinhard",
                                            protect = ProtectMode.REGION.name))
        val region = SessionRestore.rebuild(repo, ctx, SessionSnapshotStore.load(ctx)!!)
        assertTrue("REGION 快照必须恢复出分区档", region.session.isRegion)
        assertEquals("分区档的方法 id 仍是基方法", "reinhard", region.session.methodId)

        // 同一张样片、保护档 OFF：不带分区数据（plain 档，与首次选图一致）
        SessionSnapshotStore.save(ctx, snap(sampleUri = uri, method = "reinhard",
                                            protect = ProtectMode.OFF.name))
        val plain = SessionRestore.rebuild(repo, ctx, SessionSnapshotStore.load(ctx)!!)
        assertFalse("OFF 快照不该白付一次分割", plain.session.isRegion)
        println("[snapshot] 分区档随保护档恢复：REGION→isRegion=true、OFF→plain")
    }

    // ---- 4. 预设会话：全部档随条目带回；条目没了也能恢复 ----

    @Test
    fun presetSessionRestoresVariantsAndSurvivesMissingEntry() = runBlocking {
        val name = "T-快照-${System.nanoTime()}"
        val entry = PresetStore.save(ctx, name, mapOf(
            "lab_stats" to JSONObject(TestIO.assetText("goldens/lab_stats_preset.json")),
            "ot_linear" to JSONObject(TestIO.assetText("goldens/ot_preset.json")),
        ), null).also { createdPresets.add(it.id) }

        SessionSnapshotStore.save(ctx, snap(
            sampleUri = null, presetId = entry.id, presetName = name,
            presetJson = PresetStore.methodJson(ctx, entry.id, "lab_stats"),
            method = "reinhard"))
        val r = SessionRestore.rebuild(repo, ctx, SessionSnapshotStore.load(ctx)!!)
        assertEquals("reinhard", r.session.methodId)
        assertEquals("条目名要带回来（来源面板显示）", name, r.presetName)
        assertEquals("全部档都要在 ⇒ 方法 pill 可换",
            setOf("reinhard", "ot"), r.session.bundleVariants.keys)
        assertNull("预设来源不该有参考图 uri", r.sampleUri)

        // 条目被删（用户清理过预设库）：单档照常恢复，方法 pill 置灰——不静默丢现场
        PresetStore.delete(ctx, entry.id)
        val solo = SessionRestore.rebuild(repo, ctx, SessionSnapshotStore.load(ctx)!!)
        assertEquals("reinhard", solo.session.methodId)
        assertTrue("条目没了就没有可换的档", solo.session.bundleVariants.isEmpty())
        println("[snapshot] 预设会话恢复：多档带回 / 条目删除后单档恢复")
    }

    // ---- 5. 照片读不到：整项跳过，不留孤儿索引 ----

    @Test
    fun unreadablePhotosAreSkippedNotOrphaned() = runBlocking {
        val uri = sampleUri()
        SessionSnapshotStore.save(ctx, snap(
            sampleUri = uri, method = "encoder",
            photoUris = listOf(uri, "content://media/picker/does-not-exist/9"),
            activeIndex = 1))
        val r = SessionRestore.rebuild(repo, ctx, SessionSnapshotStore.load(ctx)!!)
        assertEquals("读不到的那张整项跳过", 1, r.photos.size)
        assertEquals("跳过数必须如实计数（恢复消息据此点名）", 1, r.skippedPhotos)
        assertEquals("activeIndex 必须落在有效范围内", 0, r.activeIndex)

        // 全部照片都读不到 ⇒ 空照片条 + activeIndex=-1（不是崩，也不是错位）
        SessionSnapshotStore.save(ctx, snap(
            sampleUri = uri, photoUris = listOf("content://media/picker/gone/1"),
            activeIndex = 0))
        val empty = SessionRestore.rebuild(repo, ctx, SessionSnapshotStore.load(ctx)!!)
        assertTrue(empty.photos.isEmpty())
        assertEquals("全跳过时计数也要如实", 1, empty.skippedPhotos)
        assertEquals(-1, empty.activeIndex)
        println("[snapshot] 坏 uri 处理：跳过 + activeIndex 收敛（不留孤儿索引）")
    }
}
