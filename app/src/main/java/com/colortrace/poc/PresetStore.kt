package com.colortrace.poc

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import com.colortrace.DebugLog
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.Base64
import java.util.UUID

/**
 * 预设库存储（app 私有目录 `files/presets/`，P2.14）。
 *
 * 为什么不用 MediaStore：预设是 **app 资产**不是照片——进媒体库会被照片选择器
 * 索引进相册（调试截图踩过的同型坑），且删除语义混乱。
 *
 * 结构：
 * ```
 * files/presets/index.json         —— 条目索引：{version, entries:[{id,name,created,methods}]}
 * files/presets/<id>/bundle.json   —— 自包含条目：{version,name,created,sampleThumb,methods}
 * ```
 *
 * `methods` 的每个值 = 一份**完整预设对象**（`{version,kind,strength,params}`，与桌面
 * 预设 JSON 同构）——应用某档 = 取该子对象序列化后交 [EngineRepository.loadPreset]。
 * 一个条目可容纳多档（同一张样片 fit 出的 encoder / lab_stats / ot_linear…）；
 * P2.14 先写当前会话的一档，格式已为多档预留（`methods` 是 map）。
 *
 * 写入全部走 .tmp + rename（与 [BatchResumeStore] 同款）：崩溃最多丢新状态，
 * 不会留半截 JSON 把整库带坏。
 */
object PresetStore {

    const val VERSION = 1

    /** 库条目元数据（列表用；完整内容在 bundle.json 里按下标取）。 */
    data class Entry(
        val id: String,
        val name: String,
        val created: Long,
        val methods: List<String>,
    )

    private fun root(ctx: Context) = File(ctx.filesDir, "presets")
    private fun indexFile(ctx: Context) = File(root(ctx), "index.json")
    private fun dirOf(ctx: Context, id: String) = File(root(ctx), id)
    fun bundleFile(ctx: Context, id: String) = File(dirOf(ctx, id), "bundle.json")

    // ---- 读 ----

    /** 库条目（按创建时间倒序：新存的在最上面）。索引缺失/损坏按空库处理。 */
    fun list(ctx: Context): List<Entry> = readIndex(ctx).sortedByDescending { it.created }

    /** 某条目的完整 bundle；不存在/损坏返回 null。 */
    fun load(ctx: Context, id: String): JSONObject? {
        val f = bundleFile(ctx, id)
        if (!f.exists()) return null
        return runCatching { JSONObject(f.readText()) }
            .onFailure { DebugLog.e("预设条目损坏（$id）", it) }
            .getOrNull()
    }

    /** 某条目某档的完整预设 JSON 文本（供 loadPreset）；无该档返回 null。 */
    fun methodJson(ctx: Context, id: String, kind: String): String? =
        load(ctx, id)?.optJSONObject("methods")?.optJSONObject(kind)?.toString()

    /** 条目里的档位清单（bundle 为准，索引只是缓存）。 */
    fun methodsOf(ctx: Context, id: String): List<String> =
        load(ctx, id)?.optJSONObject("methods")?.let { m -> m.keys().asSequence().toList() }
            ?: emptyList()

    /** 列表缩略图（bundle 里的 sampleThumb base64 PNG）；无图返回 null。 */
    fun thumbBitmap(ctx: Context, id: String): Bitmap? {
        val b64 = load(ctx, id)?.let { if (it.isNull("sampleThumb")) null
                                       else it.optString("sampleThumb") } ?: return null
        if (b64.isEmpty()) return null
        return runCatching {
            val bytes = Base64.getDecoder().decode(b64)
            BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        }.onFailure { DebugLog.e("预设缩略图解码失败（$id）", it) }.getOrNull()
    }

    // ---- 写 ----

    /**
     * 存入预设库（同名条目 = 覆盖更新，保留其 id；否则新建）。
     *
     * @param methods 档位 → 完整预设对象（kind 由对象自身的 `kind` 字段决定）
     * @param sampleThumbPng 样片缩略图 PNG 字节（无样片图时传 null）
     */
    fun save(ctx: Context, name: String, methods: Map<String, JSONObject>,
             sampleThumbPng: ByteArray?): Entry {
        require(methods.isNotEmpty()) { "预设至少要有一档" }
        val entries = readIndex(ctx).toMutableList()
        val existing = entries.firstOrNull { it.name == name }
        val id = existing?.id ?: freshId(entries)
        val now = System.currentTimeMillis()

        val bundle = JSONObject()
        bundle.put("version", VERSION)
        bundle.put("name", name)
        bundle.put("created", now)
        bundle.put("sampleThumb",
            sampleThumbPng?.let { Base64.getEncoder().encodeToString(it) } ?: JSONObject.NULL)
        val m = JSONObject()
        for ((kind, preset) in methods) m.put(kind, preset)
        bundle.put("methods", m)
        writeAtomic(bundleFile(ctx, id), bundle.toString())

        val entry = Entry(id, name, now, methods.keys.sorted())
        entries.removeAll { it.id == id }
        entries.add(entry)
        writeIndex(ctx, entries)
        return entry
    }

    /** 删除条目：先摘索引再删目录（中途崩溃最多留个孤儿目录，列表不再引用）。 */
    fun delete(ctx: Context, id: String) {
        val entries = readIndex(ctx).toMutableList()
        entries.removeAll { it.id == id }
        writeIndex(ctx, entries)
        dirOf(ctx, id).deleteRecursively()
    }

    // ---- 缩略图编码 ----

    /** Bitmap → PNG 字节（库缩略图；side 为长边上限，等比缩放）。 */
    fun encodeThumbPng(bmp: Bitmap, side: Int = 128): ByteArray {
        val long = maxOf(bmp.width, bmp.height)
        val scaled = if (long <= side) bmp else {
            val s = side.toFloat() / long
            Bitmap.createScaledBitmap(bmp,
                maxOf(1, (bmp.width * s).toInt()), maxOf(1, (bmp.height * s).toInt()), true)
        }
        val out = ByteArrayOutputStream()
        scaled.compress(Bitmap.CompressFormat.PNG, 100, out)
        if (scaled !== bmp) scaled.recycle()
        return out.toByteArray()
    }

    // ---- 内部 ----

    private fun freshId(entries: List<Entry>): String {
        // 短 id（目录名友好）；理论上不撞，撞了补后缀
        var id = UUID.randomUUID().toString().replace("-", "").take(10)
        while (entries.any { it.id == id }) id += "_"
        return id
    }

    private fun readIndex(ctx: Context): MutableList<Entry> {
        val f = indexFile(ctx)
        if (!f.exists()) return mutableListOf()
        return runCatching {
            val arr = JSONObject(f.readText()).optJSONArray("entries") ?: JSONArray()
            val out = mutableListOf<Entry>()
            for (i in 0 until arr.length()) {
                val e = arr.getJSONObject(i)
                val ms = mutableListOf<String>()
                e.optJSONArray("methods")?.let { for (j in 0 until it.length()) ms.add(it.getString(j)) }
                out.add(Entry(e.getString("id"), e.getString("name"),
                              e.optLong("created"), ms))
            }
            out
        }.onFailure {
            DebugLog.e("预设索引损坏（按空库处理）", it)
        }.getOrElse { mutableListOf() }
    }

    private fun writeIndex(ctx: Context, entries: List<Entry>) {
        if (entries.isEmpty()) {           // 空库 = 删索引，不留空壳
            indexFile(ctx).delete()
            return
        }
        val arr = JSONArray()
        for (e in entries) {
            arr.put(JSONObject()
                .put("id", e.id)
                .put("name", e.name)
                .put("created", e.created)
                .put("methods", JSONArray(e.methods)))
        }
        writeAtomic(indexFile(ctx), JSONObject()
            .put("version", VERSION).put("entries", arr).toString())
    }

    private fun writeAtomic(f: File, text: String) {
        f.parentFile?.mkdirs()
        val tmp = File(f.parentFile, f.name + ".tmp")
        tmp.writeText(text)
        if (!tmp.renameTo(f)) {
            f.delete()
            check(tmp.renameTo(f)) { "预设写入失败: ${f.name}" }
        }
    }
}