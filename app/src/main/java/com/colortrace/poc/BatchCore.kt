package com.colortrace.poc

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import com.colortrace.DebugLog
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.withContext

/** 批量进度回调：UI 实现写文案状态，服务实现刷通知（数值型进度）。 */
interface BatchCallbacks {
    /** 某张开始（含 uri 便于 UI 显示）。 */
    fun onPhotoStart(idx: Int, total: Int, uri: Uri) {}

    /** 某张内部的分块步进（migrate/render 各算 bands 步，P2.10 口径）。 */
    fun onStep(idx: Int, total: Int, d: Int, n: Int) {}

    /** 某张成功落盘。 */
    fun onPhotoDone(idx: Int, total: Int, w: Int, h: Int, ms: Long) {}

    /** 整批结束的汇总文案（成功/部分失败）。 */
    fun onStatus(text: String) {}
}

/**
 * 批量导出可测核心（P2.18 自 MainActivity.runBatch 抽出；执行权在
 * [BatchExportService]——切后台照常跑 + 通知栏进度）。
 *
 * 只跑 [indices] 列出的下标（普通批量 = 全部；续传 = 未完成的）。
 * 每张成功即 `markDone` 落盘——**取消/崩溃后状态文件里没有的下标就是待重跑的**；
 * 全部成功才清状态，有失败则保留（下次启动可续传重试失败项）。
 *
 * 取消 = 调用方取消协程（CancellationException 原样上抛）——已完成下标已落盘，
 * 不算失败。日志行格式与 P2.10 一致（`批量导出开始/批量 [i/N]/批量导出结束`），
 * 是排障锚点，勿改。
 *
 * decode 预取（2026-10-02 性能轮，P2.22 ④）：处理第 N 张时后台预 decode 序列里
 * 的下一张（Dispatchers.IO——decode 不占引擎车道），与 migrate/render/encode
 * 整段重叠。**外部语义零变化**：预取失败（文件损坏等）＝无预取，该张现场重解码
 * （抛同样的错误 → 既有失败处理）；预取命中/未命中只影响耗时。内存安全来自
 * 时序天然收敛——预取图与当前张的 Bitmap 并存只发生在"预取比 migrate 还快完成"
 * 时（= 中小图，预压缩档 ~22MB），大图 decode 慢、完成时当前张已被
 * `processToJpeg` 回收。取消时正在跑的预取 decode 是阻塞调用，最多多等一张的
 * 解码时长（≤2s）才传播取消，可接受。
 */
suspend fun runBatchCore(
    context: Context,
    repo: EngineRepository,
    engineDispatcher: CoroutineDispatcher,
    session: Session,
    job0: BatchResumeStore.BatchJob,
    indices: List<Int>,
    resume: Boolean,
    cb: BatchCallbacks = object : BatchCallbacks {},
): BatchResumeStore.BatchJob = coroutineScope {
    var job = job0
    val total = job.uris.size
    val failures = mutableListOf<String>()
    DebugLog.i("批量导出开始 跑=${indices.size}/$total" +
            (if (resume) "（续传，跳过已完成 ${job.done.size}）" else "") +
            " mode=${job.mode} s=${job.strength} " +
            "protect=${job.protectStrength} dev=${job.developEnabled} " +
            "导出=${ExportSize.labelOfShortSide(job.shortSide)}")
    // 预取槽：Deferred<Bitmap>（成功）或 Result.failure（decode 失败→现场重试）
    var prefetch: Deferred<Result<Bitmap>>? = null
    var prefetchIdx = -1
    try {
        for ((pos, idx) in indices.withIndex()) {
            val uri = Uri.parse(job.uris[idx])
            DebugLog.i("批量 [${idx + 1}/$total] 开始 $uri" +
                    (if (resume) "（续传）" else ""))
            cb.onPhotoStart(idx, total, uri)
            val t0 = System.currentTimeMillis()
            try {
                // 原始 = 不缩放；预压缩 = 短边目标（P2.21）。缩放在**解码这一步**做——
                // LUT 是从内容图的 128² 缩略图算的（EncoderEngine.thumbFromMat），
                // 所以"先缩再走管线"与"全尺寸走完再缩"的画面几乎一致，但快数倍。
                val full: Bitmap = if (prefetchIdx == idx && prefetch != null) {
                    val wait0 = System.currentTimeMillis()
                    val pre = prefetch!!.await()
                    val waited = (System.currentTimeMillis() - wait0).toInt()
                    val hit = pre.getOrNull()
                    if (hit != null) {
                        DebugLog.i("批量 [${idx + 1}/$total] 预取命中 " +
                                "${hit.width}x${hit.height}" +
                                (if (waited > 5) "（等 ${waited}ms）" else ""))
                        hit
                    } else {
                        DebugLog.i("批量 [${idx + 1}/$total] 预取失败→现场解码")
                        withContext(Dispatchers.IO) {
                            repo.decodeForExport(uri, job.shortSide)
                        }
                    }
                } else {
                    withContext(Dispatchers.IO) {
                        repo.decodeForExport(uri, job.shortSide)
                    }
                }
                // 启动下一张的后台 decode（覆盖本张 migrate+render+encode 全窗口；
                // 阻塞 decode 无挂起点，取消不打断——见类注释）
                val nextIdx = indices.getOrNull(pos + 1)
                prefetch = nextIdx?.let { nxt ->
                    async(Dispatchers.IO) {
                        try {
                            Result.success(repo.decodeForExport(
                                Uri.parse(job.uris[nxt]), job.shortSide))
                        } catch (ce: CancellationException) {
                            throw ce
                        } catch (e: Exception) {
                            Result.failure(e)
                        }
                    }
                }
                prefetchIdx = nextIdx ?: -1
                // 流式导出（P2.12）：接管 full，直接产 JPEG 字节
                val ex = withContext(engineDispatcher) {
                    TiledPipeline(session.model, repo.selfieNet)
                        .processToJpeg(full, ProtectMode.valueOf(job.mode),
                                       job.strength,
                                       if (job.developEnabled) job.devValues else null,
                                       job.protectStrength,
                                       film = com.colortrace.engine.FilmLayer
                                           .normalize(job.filmValues)) { d, n ->
                            cb.onStep(idx, total, d, n)
                        }
                }
                val saved = withContext(Dispatchers.IO) {
                    repo.saveJpegBytes(
                        ex.bytes, "colortrace_${idx + 1}of$total" +
                            "_${System.currentTimeMillis()}")
                }
                if (saved == null) {
                    failures.add("第 ${idx + 1} 张")
                    DebugLog.e("批量 [${idx + 1}/$total] 失败：MediaStore 行创建失败")
                } else {
                    job = withContext(Dispatchers.IO) {
                        BatchResumeStore.markDone(context, job, idx)
                    }
                    val ms = System.currentTimeMillis() - t0
                    DebugLog.i("批量 [${idx + 1}/$total] 完成 ${ex.w}x${ex.h} " +
                            "耗时=${ms}ms 相册=${saved.second}" +
                            (if (!ex.filmApplied) " 胶片=跳过（处理失败）" else ""))
                    cb.onPhotoDone(idx, total, ex.w, ex.h, ms)
                }
            } catch (e: Exception) {
                // 协程取消不算"导出失败"——直接上抛退出循环，已完成的下标都在
                // 状态文件里，下次启动续传接着跑
                if (e is CancellationException) throw e
                DebugLog.e("批量导出第 ${idx + 1}/$total 张失败 $uri", e)
                failures.add("第 ${idx + 1} 张")
            }
        }
    } finally {
        prefetch?.cancel()      // 中途取消/异常时丢弃未消费的预取
    }
    // 汇总文案**不带尺寸档**：状态行是一行，加"（预压缩 · 短边 1920）"会被截成
    // "（预压缩 · 短边 1…"（真机实测）——尺寸已写进日志，且用户刚在菜单里选过。
    // resume 由调用方按"确实有已完成项"判定（P2.21 修正：服务曾写死 true，
    // 导致全新批次也说"本次…全部保存成功"）。
    cb.onStatus(if (failures.isEmpty()) {
        if (resume) "批量导出完成：本次 ${indices.size} 张全部保存成功"
        else "批量导出完成：$total 张已保存到相册"
    } else {
        "批量导出完成：成功 ${indices.size - failures.size}/${indices.size}；" +
                "${failures.first()}失败"
    })
    DebugLog.i("批量导出结束 本次成功=${indices.size - failures.size}/" +
            "${indices.size} 累计=${job.done.size}/$total " +
            "mode=${job.mode} s=${job.strength} dev=${job.developEnabled} " +
            "导出=${ExportSize.labelOfShortSide(job.shortSide)}")
    if (failures.isEmpty()) {
        withContext(Dispatchers.IO) { BatchResumeStore.clear(context) }
    }
    job
}

private fun waited(since: Long): Int = (System.currentTimeMillis() - since).toInt()
