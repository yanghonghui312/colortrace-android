package com.colortrace.poc

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * 断点续传状态存取关卡（P2.10）：批量导出的崩溃恢复靠 `files/batch_resume.json`。
 * 关卡：全字段 save/load 往返一致、markDone 立即落盘、样片/预设两种会话来源、
 * 损坏文件兜底为 null 且被清除、clear 干净。
 */
@RunWith(AndroidJUnit4::class)
class BatchResumeTest {

    private val ctx get() = InstrumentationRegistry.getInstrumentation().targetContext

    private fun job(done: Set<Int> = emptySet()) = BatchResumeStore.BatchJob(
        sampleUri = "content://media/picker/test/1", presetJson = null,
        mode = "REGION", strength = 0.5f, protectStrength = 0.8f,
        developEnabled = true,
        devValues = mapOf("exposure" to 0.75f, "gamma" to 1.15f),
        uris = listOf("u://0", "u://1", "u://2"), done = done)

    @Test
    fun roundTripMarkDoneAndClear() {
        BatchResumeStore.clear(ctx)
        assertNull("初始不应有状态", BatchResumeStore.load(ctx))

        BatchResumeStore.save(ctx, job())
        val loaded = BatchResumeStore.load(ctx)!!
        assertEquals("全字段往返必须一致", job(), loaded)
        assertEquals(listOf(0, 1, 2), loaded.pending)

        // markDone 落盘：重新 load 要能看到（崩溃恢复的最低要求）
        val j1 = BatchResumeStore.markDone(ctx, loaded, 1)
        assertEquals(setOf(1), j1.done)
        assertEquals(listOf(0, 2), BatchResumeStore.load(ctx)!!.pending)

        // 桌面预设来源的往返（presetJson 与 sampleUri 二选一）
        val p = job().copy(sampleUri = null, presetJson = "{\"ver\":1}")
        BatchResumeStore.save(ctx, p)
        assertEquals(p, BatchResumeStore.load(ctx))

        BatchResumeStore.clear(ctx)
        assertNull(BatchResumeStore.load(ctx))
        assertEquals(false, File(ctx.filesDir, "batch_resume.json").exists())
    }

    @Test
    fun corruptFileFallsBackToNull() {
        BatchResumeStore.clear(ctx)
        File(ctx.filesDir, "batch_resume.json").writeText("{oops")
        assertNull("损坏状态必须返回 null（不能让续传崩在启动路径上）",
            BatchResumeStore.load(ctx))
        assertNull("损坏文件应被顺手删除", BatchResumeStore.load(ctx))
    }
}
