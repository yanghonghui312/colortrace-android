package com.colortrace.poc

import android.content.Context
import com.colortrace.DebugLog
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * 编辑会话快照（`files/session_snapshot.json`，P2.20）。
 *
 * 为什么需要它：v1 没有 ViewModel，会话状态全在 `AppRoot` 的 `remember` 里——
 * **覆盖安装（`adb install -r`）、系统回收、被强杀**都会让"参考图 + 照片条 + 一路
 * 调好的参数"全部消失，只剩重启后重新选图重调。这里把**重建会话所需的最小信息**
 * 落盘，重启后欢迎屏出现「继续上次编辑」，一键回到原来的编辑现场。
 *
 * 存什么 / 不存什么（与 [BatchResumeStore] 同一取舍）：存**来源**（样片 uri 或
 * 预设 JSON + 库条目 id）、照片 uri 列表、当前选中项、以及全部参数（强度 / 保护档 /
 * 保护强度 / 精修开关与参数表）；**不存**像素与拟合结果——样片重新 refit 即可
 * （代价与首次选图相当，故恢复做成**用户点一下**才发生，不压冷启动）。
 *
 * 前提：样片/照片 uri 在选择器回调里做过 `takePersistableUriPermission`
 * （见 MainActivity），否则进程死后 uri 读不了——恢复时报错并清除快照，
 * 不留下"每次启动都失败"的空壳。
 *
 * 与批量续传的关系：两者是**独立的文件**（`batch_resume.json` 管"哪几张还没导完"，
 * 本文件管"编辑现场"）。批量在前台服务里跑时本文件照常写着当前会话。
 */
object SessionSnapshotStore {

    const val VERSION = 1

    /**
     * 一次编辑会话的完整快照。字段全部是**能跨进程重放**的原始值
     * （uri 字符串 / 预设 JSON 文本 / 参数），不含任何 Bitmap 或引擎对象。
     */
    data class Snapshot(
        val sampleUri: String?,      // 设备端 fit 的参考图（与 presetJson 二选一）
        val presetId: String?,       // 预设库条目 id（用于把**全部方法档**一起带回来）
        val presetName: String?,     // 条目名（来源面板显示用）
        val presetJson: String?,     // 当前档的预设 JSON 全文（自包含，条目没了也能恢复）
        val method: String,          // 当前方法档 id（encoder / reinhard / ot）
        val photoUris: List<String>, // 照片条（按加入顺序）
        val activeIndex: Int,        // 当前选中项（<0 = 无）
        val strength: Float,
        val protect: String,         // ProtectMode.name
        val protectStrength: Float,
        val developEnabled: Boolean,
        val devValues: Map<String, Float>,
        /** 胶片感参数全表（含三个 *_on 开关；2026-10-02）。旧快照缺省 =
         *  emptyMap → 恢复时按 [com.colortrace.engine.FilmLayer.DEFAULTS] 补全
         *  （开关默认关 = 不启用）。**不进预设库**（用户拍板：预设只互通追色）。 */
        val filmValues: Map<String, Float> = emptyMap(),
    ) {
        /** 来源二选一必须在——否则无从重建会话，视为无效快照。 */
        val usable: Boolean get() = sampleUri != null || presetJson != null
    }

    private fun file(ctx: Context) = File(ctx.filesDir, "session_snapshot.json")
    private fun tmpFile(ctx: Context) = File(ctx.filesDir, "session_snapshot.json.tmp")

    /** 原子写：先写 .tmp 再 rename 覆盖——崩溃最多丢新状态，不会留半截 JSON。 */
    fun save(ctx: Context, snap: Snapshot) {
        val o = JSONObject()
        o.put("version", VERSION)
        o.put("sampleUri", snap.sampleUri ?: JSONObject.NULL)
        o.put("presetId", snap.presetId ?: JSONObject.NULL)
        o.put("presetName", snap.presetName ?: JSONObject.NULL)
        o.put("presetJson", snap.presetJson ?: JSONObject.NULL)
        o.put("method", snap.method)
        o.put("photoUris", JSONArray(snap.photoUris))
        o.put("activeIndex", snap.activeIndex)
        o.put("strength", snap.strength.toDouble())
        o.put("protect", snap.protect)
        o.put("protectStrength", snap.protectStrength.toDouble())
        o.put("developEnabled", snap.developEnabled)
        o.put("devValues", JSONObject().apply {
            for ((k, v) in snap.devValues) put(k, v.toDouble())
        })
        o.put("filmValues", JSONObject().apply {
            for ((k, v) in snap.filmValues) put(k, v.toDouble())
        })
        val tmp = tmpFile(ctx)
        tmp.writeText(o.toString())
        if (!tmp.renameTo(file(ctx))) {
            file(ctx).delete()
            check(tmp.renameTo(file(ctx))) { "session_snapshot.json 写入失败" }
        }
    }

    /**
     * 读取快照；不存在、损坏、或**来源缺失**（无从重建）返回 null。
     * 后两种顺手删掉——不让坏状态每次启动都来打扰一次（同 BatchResumeStore 口径）。
     * 字段一律按 `opt*` 读：旧快照缺字段时取保守默认值，不因新增字段而整段作废。
     */
    fun load(ctx: Context): Snapshot? {
        val f = file(ctx)
        if (!f.exists()) return null
        val snap = runCatching {
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
            o.optJSONArray("photoUris")?.let { a ->
                for (i in 0 until a.length()) uris.add(a.getString(i))
            }
            Snapshot(
                sampleUri = if (o.isNull("sampleUri")) null else o.optString("sampleUri"),
                presetId = if (o.isNull("presetId")) null else o.optString("presetId"),
                presetName = if (o.isNull("presetName")) null else o.optString("presetName"),
                presetJson = if (o.isNull("presetJson")) null else o.optString("presetJson"),
                method = o.optString("method", "encoder"),
                photoUris = uris,
                activeIndex = o.optInt("activeIndex", -1),
                strength = o.optDouble("strength", 0.5).toFloat(),
                protect = o.optString("protect", ProtectMode.OFF.name),
                protectStrength = o.optDouble("protectStrength", 1.0).toFloat(),
                developEnabled = o.optBoolean("developEnabled", false),
                devValues = dev,
                filmValues = film,
            )
        }.onFailure {
            DebugLog.e("会话快照损坏（忽略并清除）", it)
        }.getOrNull()
        if (snap != null && !snap.usable) {
            DebugLog.i("会话快照无来源（无从重建）——已清除")
            clear(ctx)
            return null
        }
        if (snap == null) clear(ctx)
        return snap
    }

    fun clear(ctx: Context) {
        file(ctx).delete()
        tmpFile(ctx).delete()
    }
}
