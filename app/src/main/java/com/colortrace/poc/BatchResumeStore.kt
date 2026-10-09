package com.colortrace.poc

import android.content.Context
import com.colortrace.DebugLog
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * 批量导出断点续传状态（`files/batch_resume.json`）。
 *
 * 批量开始时把**当时的参数 + 会话来源**快照落盘，每张成功后原子更新 `done`；
 * 崩溃 / 进程被杀后重启，App 据此提示「接着导出未完成的」而不是从头跑。
 * 全部成功即清除；有失败则保留（下次启动可续传重试失败项）。
 *
 * 前提：样片/照片 uri 在选择器回调里做了 `takePersistableUriPermission`
 * （见 MainActivity），否则进程死后 uri 读不了、续传只能放弃并清状态。
 */
object BatchResumeStore {

    /** 一次批量导出的完整快照（足够在全新进程里重建会话并续跑）。 */
    data class BatchJob(
        val sampleUri: String?,      // 设备端 fit 的样片（与 presetJson 二选一）
        val presetJson: String?,     // 桌面预设 JSON 全文（与 sampleUri 二选一）
        val mode: String,            // ProtectMode.name
        val strength: Float,
        val protectStrength: Float,
        val developEnabled: Boolean,
        val devValues: Map<String, Float>,
        val uris: List<String>,      // 批量目标照片（按加入顺序）
        val done: Set<Int>,          // 已成功导出的下标
        val method: String = "encoder",  // 设备端 fit 方法档（P2.13；旧快照缺省 encoder）
        /** 导出短边像素（P2.21；0 = 原始尺寸）。旧快照缺省 0 ⇒ 续传口径与从前一致。 */
        val shortSide: Int = 0,
        /** 胶片感参数全表（含 *_on 开关；2026-10-02）。旧快照缺省 emptyMap =
         *  不启用（恢复时按 FilmLayer.DEFAULTS 补全）。不进预设（用户拍板）。 */
        val filmValues: Map<String, Float> = emptyMap(),
    ) {
        /** 还没导完的下标（续传就跑这些）。 */
        val pending: List<Int> get() = uris.indices.filter { it !in done }
    }

    private fun file(ctx: Context) = File(ctx.filesDir, "batch_resume.json")

    /** 原子写：先写 .tmp 再 rename 覆盖——崩溃最多丢旧状态，不会留半截文件。 */
    fun save(ctx: Context, job: BatchJob) {
        val o = JSONObject()
        o.put("sampleUri", job.sampleUri ?: JSONObject.NULL)
        o.put("presetJson", job.presetJson ?: JSONObject.NULL)
        o.put("mode", job.mode)
        o.put("strength", job.strength.toDouble())
        o.put("protectStrength", job.protectStrength.toDouble())
        o.put("developEnabled", job.developEnabled)
        o.put("devValues", JSONObject().apply {
            for ((k, v) in job.devValues) put(k, v.toDouble())
        })
        o.put("uris", JSONArray(job.uris))
        o.put("done", JSONArray(job.done.sorted()))
        o.put("method", job.method)
        o.put("shortSide", job.shortSide)   // P2.21；旧版本读不到会走 optInt 缺省 0
        o.put("filmValues", JSONObject().apply {
            for ((k, v) in job.filmValues) put(k, v.toDouble())
        })
        val tmp = File(ctx.filesDir, "batch_resume.json.tmp")
        tmp.writeText(o.toString())
        if (!tmp.renameTo(file(ctx))) {
            file(ctx).delete()
            check(tmp.renameTo(file(ctx))) { "batch_resume.json 写入失败" }
        }
    }

    /** 读取快照；不存在或损坏返回 null（损坏时顺手删掉，不让坏状态反复打扰）。 */
    fun load(ctx: Context): BatchJob? {
        val f = file(ctx)
        if (!f.exists()) return null
        return runCatching {
            val o = JSONObject(f.readText())
            val dev = mutableMapOf<String, Float>()
            o.optJSONObject("devValues")?.let { dv ->
                for (k in dv.keys()) dev[k] = dv.getDouble(k).toFloat()
            }
            val film = mutableMapOf<String, Float>()
            o.optJSONObject("filmValues")?.let { fv ->
                for (k in fv.keys()) film[k] = fv.getDouble(k).toFloat()
            }
            val uris = mutableListOf<String>()
            val ua = o.getJSONArray("uris")
            for (i in 0 until ua.length()) uris.add(ua.getString(i))
            val done = mutableSetOf<Int>()
            val da = o.getJSONArray("done")
            for (i in 0 until da.length()) done.add(da.getInt(i))
            BatchJob(
                sampleUri = if (o.isNull("sampleUri")) null else o.getString("sampleUri"),
                presetJson = if (o.isNull("presetJson")) null else o.getString("presetJson"),
                mode = o.getString("mode"),
                strength = o.getDouble("strength").toFloat(),
                protectStrength = o.getDouble("protectStrength").toFloat(),
                developEnabled = o.getBoolean("developEnabled"),
                devValues = dev,
                uris = uris,
                done = done,
                method = o.optString("method", "encoder"),   // 旧快照兼容
                shortSide = o.optInt("shortSide", 0),        // 旧快照兼容：原始尺寸
                filmValues = film,                           // 旧快照兼容：不启用
            )
        }.onFailure {
            DebugLog.e("断点续传状态损坏（忽略并清除）", it)
            f.delete()
        }.getOrNull()
    }

    /** 标记第 index 张完成（立即落盘并返回新 job）。 */
    fun markDone(ctx: Context, job: BatchJob, index: Int): BatchJob =
        job.copy(done = job.done + index).also { save(ctx, it) }

    fun clear(ctx: Context) {
        file(ctx).delete()
        File(ctx.filesDir, "batch_resume.json.tmp").delete()
    }
}
