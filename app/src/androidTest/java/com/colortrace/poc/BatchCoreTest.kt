package com.colortrace.poc

import android.graphics.Bitmap
import android.net.Uri
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.colortrace.DebugLog
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader

/**
 * 批量导出核心 [runBatchCore] 的服务无关回归（P2.18 前台服务化的三案）：
 *  1. 全部成功 → 每张 markDone、状态文件自清、回调序完整；
 *  2. 含失败 → 有效张保留标记，失败项留在 pending（下次启动可续传）；
 *  3. 取消（协程取消）→ 已完成的下标已在状态文件里，pending 只剩未跑的。
 *
 * 测试照片走真实 MediaStore（saveBitmap 写入 + @After 删除），快照文件用
 * `T-` 前缀命名目录隔离——与 PresetStoreTest 同一套纪律。注意：批量核心写的是
 * **真实相册**（saveJpegBytes）与真实 `batch_resume.json`——跑前备份/跑后还原。
 */
@RunWith(AndroidJUnit4::class)
class BatchCoreTest {

    private val ctx get() = InstrumentationRegistry.getInstrumentation().targetContext
    private val createdUris = mutableListOf<Uri>()
    private lateinit var repo: EngineRepository
    private lateinit var sample: Bitmap

    @Before
    fun setUp() {
        assertTrue("OpenCV native 加载失败", OpenCVLoader.initLocal())
        repo = EngineRepository(ctx)
        sample = Bitmap.createBitmap(96, 128, Bitmap.Config.ARGB_8888)
        sample.eraseColor(0xFF3366CC.toInt())
        runBlocking {
            // 批量核心操作真实 batch_resume.json——先清场（测试独占它）
            BatchResumeStore.clear(ctx)
            repeat(2) { i ->
                val saved = repo.saveBitmap(
                    sample, "colortrace_batchcore_${System.nanoTime()}_$i")
                assertNotNull("测试照片落盘失败", saved)
                createdUris.add(saved!!.first)
            }
        }
    }

    @After
    fun tearDown() {
        for (u in createdUris) runCatching { ctx.contentResolver.delete(u, null, null) }
        runCatching { BatchResumeStore.clear(ctx) }
        sample.recycle()
    }

    private fun makeJob(): BatchResumeStore.BatchJob {
        // 会话：lab_stats 对 96x128 自 fit（OFF 模式出图，无需分割/模型）
        val bmp = repo.decodeCapped(createdUris[0], EngineRepository.PROC_MAX)
        val s = repo.fitSample(bmp, "reinhard")
        bmp.recycle()
        return BatchResumeStore.BatchJob(
            sampleUri = null,
            presetJson = s.toPresetJson(),
            mode = "OFF",
            strength = 1f,
            protectStrength = 1f,
            developEnabled = false,
            devValues = emptyMap(),
            uris = createdUris.map { it.toString() },
            method = "reinhard",
            done = emptySet())
    }

    @Test
    fun batchCoreRunsAndClearsState() = runBlocking {
        val job = makeJob()
        var started = 0; var stepped = 0; var done = 0; var status = ""
        val final = runBatchCore(ctx, repo, EngineBox.engineDispatcher,
                                 repo.loadPreset(job.presetJson!!), job,
                                 job.uris.indices.toList(), resume = false,
                                 cb = object : BatchCallbacks {
            override fun onPhotoStart(idx: Int, total: Int, uri: Uri) { started++ }
            override fun onStep(idx: Int, total: Int, d: Int, n: Int) { stepped++ }
            override fun onPhotoDone(idx: Int, total: Int, w: Int, h: Int, ms: Long) { done++ }
            override fun onStatus(text: String) { status = text }
        })
        assertEquals("两张都应 markDone", setOf(0, 1), final.done)
        assertEquals(2, started); assertEquals(2, done)
        assertTrue("分块步进回调应发生", stepped > 0)
        assertTrue("状态文件应自清",
                   null == BatchResumeStore.load(ctx)?.takeIf { it.pending.isNotEmpty() })
        assertTrue("汇总文案应报成功", status.contains("批量导出完成"))
    }

    @Test
    fun batchCoreFailureKeepsPendingForResume() = runBlocking {
        val job = makeJob().copy(
            uris = createdUris.map { it.toString() } + "content://media/none/exist.jpg")
        // 快照按 3 张落盘：第 3 张 uri 无效 → 失败保留，前两张完成
        BatchResumeStore.save(ctx, job)
        val final = runBatchCore(ctx, repo, EngineBox.engineDispatcher,
                                 repo.loadPreset(job.presetJson!!), job,
                                 job.pending, resume = false)
        assertEquals("前两张应完成", setOf(0, 1), final.done)
        val reloaded = BatchResumeStore.load(ctx)
        assertNotNull("有失败应保留状态文件", reloaded)
        assertEquals("失败项应留在 pending", listOf(2), reloaded!!.pending)
    }

    @Test
    fun batchCoreCancelKeepsDoneMarks() = runBlocking {
        val job = makeJob()
        BatchResumeStore.save(ctx, job)
        val deferred = CoroutineScope(Dispatchers.Default).async {
            runBatchCore(ctx, repo, EngineBox.engineDispatcher,
                         repo.loadPreset(job.presetJson!!), job,
                         job.pending, resume = false,
                         cb = object : BatchCallbacks {
                override fun onPhotoDone(idx: Int, total: Int, w: Int, h: Int, ms: Long) {
                    // 第一张完成后立刻取消——模拟服务被停/用户点「停止」
                    throw CancellationException("stop requested")
                }
            })
        }
        var cancelled = false
        try { deferred.await() } catch (e: CancellationException) { cancelled = true }
        assertTrue("应以取消收场", cancelled)
        val reloaded = BatchResumeStore.load(ctx)
        assertNotNull("取消后状态文件应保留", reloaded)
        assertEquals("已完成的第 1 张应已标记", setOf(0), reloaded!!.done)
        assertEquals("待跑只剩第 2 张", listOf(1), reloaded.pending)
    }
}
