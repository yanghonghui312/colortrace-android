package com.colortrace.engine

import com.colortrace.DebugLog

/**
 * native 热循环入口（`libcolortrace.so`，源码 `src/main/cpp/lab_conv.cpp`）。
 *
 * **降级纪律**：`.so` 加载失败（ABI 缺失/打包遗漏/16KB page size 不符）时
 * [available] 为 false，调用方回退纯 Kotlin 实现——一次加载失败绝不变成崩溃，
 * 也绝不静默换成别的数值口径。
 *
 * 数值口径见 `lab_conv.cpp` 顶部注释：与 Kotlin/桌面**逐位等价**（float32、
 * 同序乘加、超越函数 double 后截 float、编译期禁 FMA/fast-math）。
 */
object NativeLab {
    /** 是否可用；加载失败只记日志，不抛（`System.loadLibrary` 抛的是 Error）。 */
    val available: Boolean = runCatching {
        System.loadLibrary("colortrace")
        true
    }.onFailure {
        DebugLog.e("libcolortrace.so 加载失败→回退纯 Kotlin 引擎", it)
    }.getOrDefault(false)

    /** 启动横幅用：确认设备上真加载的是哪一版 native。 */
    val version: String = if (available) nativeVersion() else "n/a"

    private external fun nativeVersion(): String

    /** 与 [LabConv.rgbToLabKotlin] 逐位等价；分配失败/参数不合法时返回 null。 */
    external fun rgbToLab(rgb: FloatArray): FloatArray?

    /** 与 [LabConv.labToRgbKotlin] 逐位等价；分配失败/参数不合法时返回 null。 */
    external fun labToRgb(lab: FloatArray): FloatArray?
}