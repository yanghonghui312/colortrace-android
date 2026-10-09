package com.colortrace.poc

/**
 * 导出尺寸档（P2.21）。
 *
 * 为什么需要：原始尺寸导出对发朋友圈是**过量的**——平台侧拿到巨图后会自己
 * 重采样 + 重编码（实测口径：JPEG 会被
 * 等比缩到宽 ≤1280 / 高 ≤16383，并且**无论是否缩放都会被重新编码**）。
 * 先在本机缩到目标尺寸，好处是：① 上传更快、少一次平台侧的重采样；
 * ② 少一次"我们编码 → 平台解码 → 平台再编码"的世代损失。
 *
 * [shortSide] 是**短边像素**（0 = 不缩放，保持原始尺寸）。取短边而不是长边，是因为
 * 竖拍的常见长宽比差异很大（4:3 / 16:9 / 全景），按短边定档，横竖图的"视觉尺寸"
 * 才一致。
 *
 * ⚠️ 只在这里改尺寸口径——UI 文案里的数字也从这里取（`ExportSize.SOCIAL.shortSide`），
 * 不要在界面里写死 1920。
 */
enum class ExportSize(val shortSide: Int, val label: String) {
    /** 不缩放；与 P2.5/P2.12 以来的导出口径完全一致。 */
    ORIGINAL(0, "原始尺寸"),

    /** 预压缩：短边 1920（长边按比例），发朋友圈用。 */
    SOCIAL(1920, "预压缩 · 短边 1920");

    companion object {
        /**
         * 快照/参数里的口径：0 = 原始，其余 = 该短边像素。
         * 认不出的数值照实回报（不静默当原始——那会让"预压缩"悄悄失效）。
         */
        fun labelOfShortSide(shortSide: Int): String =
            entries.firstOrNull { it.shortSide == shortSide }?.label
                ?: if (shortSide <= 0) ORIGINAL.label else "短边 $shortSide"
    }
}
