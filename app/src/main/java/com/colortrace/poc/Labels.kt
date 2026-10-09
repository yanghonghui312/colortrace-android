package com.colortrace.poc

/**
 * 方法档 / 预设档 → 用户可见名称（2026-09-30 文案规范定稿）。
 *
 * 这条映射此前散在**四处**（EditScreen 的 pill、EditScreen 的方法选择器、MainActivity、
 * PresetLibrarySheet——最后一处还多一个 region 分支），已经出现"pill 与选择器文案不一致"。
 * 现在只此一份。
 *
 * 口径（2026-09-30 二次定稿：**收紧成纯中文**）：档名会出现在预设库条目的副标题里
 * （三档并列 + 日期），带英文括号时被迫折成三行，故去掉英文。
 * encoder = AI 追色、reinhard = 经典统计、ot = 影调保真；
 * 旧预设档 region = **分区统计**（与保护模式"分区保护"区分开）。
 * 设备端方法 id（encoder/reinhard/ot）与预设 kind（encoder/lab_stats/ot_linear/region）
 * 都在这里收口，两种写法都能过——导入回报里的 skippedKinds 就是 kind。
 */
fun methodLabelOf(id: String): String = when (id) {
    "reinhard", "lab_stats" -> "经典统计"
    "ot", "ot_linear" -> "影调保真"
    "region" -> "分区统计"
    else -> "AI 追色"                       // encoder
}

/**
 * 方法档 → 一行短标签（"切换追色方法"下拉里，显示在档名下方的小字）。
 *
 * 2026-09-30 二次修订：用户反馈"三种方法稍稍有点挤"——原先 30~40 字符的简介在
 * 对话框里必然折成两行、还断在词中间。这里收成**一行放得下**的短标签
 * （≤19 字符，格式统一为"算法名 · 特点"），**完整说明搬进使用帮助面板**
 * （`ui/HelpSheet.kt` 的「追色方法」一节）。
 *
 * 专业名词按要求保留：Encoder / Reinhard / OT。
 */
fun methodDescOf(id: String): String = when (id) {
    "reinhard", "lab_stats" -> "Reinhard 统计迁移 · 速度最快"
    "ot", "ot_linear" -> "高斯最优传输（OT）· 明暗保真"
    else -> "自训前馈网络（Encoder）· 适用最广"
}

/**
 * 保护模式快照值（ProtectMode.name 字符串）→ 中文标签。
 *
 * 批量续传快照里存的是 `OFF`/`CHROMA`/`REGION`（**格式不变**，旧快照照样能读），
 * 只在展示时翻译——此前直接把 job.mode 拼进续传对话框，用户看到的是英文枚举。
 */
fun protectModeLabelOf(mode: String): String =
    runCatching { ProtectMode.valueOf(mode).label }.getOrDefault(mode)

/**
 * 预设库存储键（kind）↔ 方法 pill 的方法 id：`Entry.methods` / 快照用 kind，
 * UI / 会话用 id。2026-10-01 从 MainActivity 挪到这里——会话恢复
 * （`SessionRestore`）也要用同一份映射，第二份就是不一致的温床。
 */
fun methodIdOfKind(kind: String): String = when (kind) {
    "lab_stats" -> "reinhard"
    "ot_linear" -> "ot"
    else -> "encoder"
}
