package com.colortrace.poc

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import com.colortrace.DebugLog
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * 会话恢复（P2.20）：按 [SessionSnapshotStore.Snapshot] 把编辑现场重建出来。
 *
 * 为什么单独成文件而不是写在 `AppRoot` 里：这是**唯一**要跨进程重放的逻辑，
 * 也是"恢复出来的东西和原来是不是同一个"的唯一战场——放在 Composable 里就只能
 * 靠手点验证。这里做成普通 suspend 函数，插桩测试可以拿真解释器/真模型跑一遍
 * （`SessionSnapshotTest`：同一张参考图 fit 两次，出图**逐字节相同**）。
 *
 * 两条重建路径与首次选图完全同源，不新增数学：
 *  - **参考图会话**：重解码样片 → `fitSample(method, region = 保护档是 REGION)`。
 *    region 标志必须跟保护档走：REGION × 统计类若不升档，恢复出来就切不回分区保护
 *    （`applyProtectMode` 只对**有样片像素**的会话现场升档，快照恢复正好在手）。
 *  - **预设会话**：`loadPreset(存的当前档 JSON, 条目全部档)`——存的 JSON 自包含，
 *    条目被删也照样恢复（只是方法 pill 置灰，不静默丢整段现场）。
 *
 * 照片一律**整项跳过**解不开的 uri（uri 与缩略图绑成一项）：宁可少一张，
 * 也不能留下"索引指向不存在的缩略图"错位（选图路径同口径）。
 */
object SessionRestore {

    /** 恢复产物：会话 + 与选图路径同型的照片项 + 来源信息。 */
    class Result(
        val session: Session,
        val sampleUri: String?,
        val presetJson: String?,
        val presetId: String?,
        val presetName: String?,
        val sampleThumbPng: ByteArray?,
        val photos: List<Pair<Uri, Bitmap>>,
        val activeIndex: Int,
        /** 原图已不可读而被整项跳过的照片数——恢复消息要点名，不静默（2026-10-03）。 */
        val skippedPhotos: Int = 0,
    )

    /**
     * @param io 解码/缩略图线程（磁盘 + 解码）
     * @param engine 引擎线程（fit / loadPreset）
     * @throws Exception 来源不可用（uri 读不到、预设损坏）——调用方负责清快照并说明
     */
    suspend fun rebuild(
        repo: EngineRepository,
        ctx: Context,
        snap: SessionSnapshotStore.Snapshot,
        io: CoroutineDispatcher = Dispatchers.IO,
        engine: CoroutineDispatcher = Dispatchers.Default,
    ): Result {
        val session: Session
        val thumb: ByteArray?
        if (snap.sampleUri != null) {
            val bmp = withContext(io) {
                repo.decodeCapped(Uri.parse(snap.sampleUri), EngineRepository.PROC_MAX)
            }
            thumb = withContext(io) { PresetStore.encodeThumbPng(bmp) }
            session = withContext(engine) {
                repo.fitSample(bmp, snap.method,
                               region = snap.protect == ProtectMode.REGION.name)
            }
        } else {
            val json = snap.presetJson ?: throw IllegalArgumentException("快照没有会话来源")
            val variants = variantsOf(ctx, snap.presetId, io)
            session = withContext(engine) { repo.loadPreset(json, variants) }
            thumb = repo.sampleThumbPng(session)
            DebugLog.i("恢复预设会话 ${snap.presetName ?: "（无条目名）"} " +
                    "档=${variants.keys.ifEmpty { setOf("（条目已删，单档恢复）") }}")
        }

        var skipped = 0
        val photos = withContext(io) {
            snap.photoUris.mapNotNull { u ->
                runCatching {
                    val uri = Uri.parse(u)
                    uri to repo.decodeCapped(uri, EngineRepository.THUMB_MAX)
                }.onFailure {
                    skipped++
                    DebugLog.e("恢复会话：缩略图解码失败 $u（该项已跳过）", it)
                }.getOrNull()
            }
        }

        // 参考图会话带分区数据（region 升档）⇒ 恢复后保护档能原样立住
        return Result(
            session = session,
            sampleUri = snap.sampleUri,
            presetJson = if (snap.sampleUri == null) snap.presetJson else null,
            presetId = if (snap.sampleUri == null) snap.presetId else null,
            presetName = if (snap.sampleUri == null) snap.presetName else null,
            sampleThumbPng = thumb,
            photos = photos,
            activeIndex = snap.activeIndex.coerceIn(-1, photos.size - 1),
            skippedPhotos = skipped,
        )
    }

    /**
     * 库条目的**全部档**（methodId → 预设 JSON），构造与「应用预设」一致。
     * 条目已删 / id 为空 → 空 map：会话照常恢复、方法 pill 置灰，
     * 不因为一个**可选**的库文件把整段编辑现场丢掉。
     */
    suspend fun variantsOf(ctx: Context, id: String?,
                           io: CoroutineDispatcher = Dispatchers.IO): Map<String, String> {
        if (id == null) return emptyMap()
        val out = LinkedHashMap<String, String>()
        for (k in withContext(io) { PresetStore.methodsOf(ctx, id) }) {
            val text = withContext(io) { PresetStore.methodJson(ctx, id, k) } ?: continue
            out[methodIdOfKind(k)] = text
        }
        return out
    }
}
