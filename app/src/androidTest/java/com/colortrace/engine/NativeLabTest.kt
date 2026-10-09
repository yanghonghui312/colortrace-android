package com.colortrace.engine

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.Random
import androidx.test.platform.app.InstrumentationRegistry

/**
 * native 热循环关卡（P2.8 里程碑：C++/NEON 移植第一步——LabConv）。
 *
 * **判据是逐位**：native（`src/main/cpp/lab_conv.cpp`）与纯 Kotlin 参考实现
 * 必须给出**完全相同的 float32 位模式**。这条比"u8 逐位"更强的口径是刻意选的：
 * u8 逐位只能发现 ≥1/255 的漂移，而 C++ 移植最容易出的问题正是 1 ulp 级
 * （FMA 收缩、pow 走了 float 版本、结合序被重排）——那些会在这里立刻暴露，
 * 而不是等到某张图偶然放大成可见色带。
 *
 * 允许的唯一例外是 **±0.0**：两者数值相等、u8 输出相同，位模式不同不构成漂移。
 *
 * 输入覆盖各分支：c≤0.04045（sRGB 线性段）、t≤EPS（fCie 线性段）、
 * L>KAPPA·EPS（labToRgb 的 fy³ 段）、invF 的 f3>EPS 两侧、越界裁剪。
 */
@RunWith(AndroidJUnit4::class)
class NativeLabTest {

    // 类级共享：tearDown 每个测试都写盘，逐方法各写一份会互相覆盖
    companion object {
        private val report = JSONObject()
    }

    private val ctx get() = InstrumentationRegistry.getInstrumentation().targetContext

    /** 位模式不同但都等于 0（±0.0）不算漂移。 */
    private fun mismatches(a: FloatArray, b: FloatArray): IntArray {
        val hits = ArrayList<Int>()
        for (i in a.indices) {
            if (a[i].toRawBits() == b[i].toRawBits()) continue
            if (a[i] == 0f && b[i] == 0f) continue
            hits.add(i)
        }
        return hits.toIntArray()
    }

    private fun maxAbsDiff(a: FloatArray, b: FloatArray): Float {
        var m = 0f
        for (i in a.indices) {
            val d = kotlin.math.abs(a[i] - b[i])
            if (d > m) m = d
        }
        return m
    }

    /** 覆盖各分支的 RGB 样本：细网格 + 阈值邻域 + 越界 + 随机。 */
    private fun rgbSamples(): FloatArray {
        val v = ArrayList<Float>()
        // 阈值邻域（sRGB 线性段边界 0.04045 / 0.0031308，以及暗部）
        for (base in floatArrayOf(0f, 0.0031308f, 0.04f, 0.04045f, 0.04046f, 1f)) {
            for (d in intArrayOf(-2, -1, 0, 1, 2)) {
                v.add(base + d * 1e-4f)
            }
        }
        // 网格 + 越界（覆盖 clamp01）
        var i = -20
        while (i <= 120) { v.add(i / 100f); i += 7 }
        val rnd = Random(20260929L)
        repeat(400) { v.add(rnd.nextFloat() * 1.6f - 0.3f) }
        // 铺成 RGB 三元组（每个采样值都当 R/G/B 轮换过一次，避免只测到单通道）
        val out = FloatArray(v.size * 3)
        for (k in v.indices) {
            out[3 * k] = v[k]
            out[3 * k + 1] = v[(k * 7 + 1) % v.size]
            out[3 * k + 2] = v[(k * 13 + 2) % v.size]
        }
        return out
    }

    /** 覆盖 LAB 各分支的样本（L 含负值/超 100；a,b 含大范围越界）。 */
    private fun labSamples(): FloatArray {
        val v = ArrayList<Float>()
        for (L in intArrayOf(-20, 0, 1, 8, 9, 90, 100, 120)) v.add(L.toFloat())
        var a = -150
        while (a <= 150) { v.add(a.toFloat()); a += 9 }
        val rnd = Random(7L)
        repeat(400) { v.add(rnd.nextFloat() * 300f - 150f) }
        val out = FloatArray(v.size * 3)
        for (k in v.indices) {
            out[3 * k] = v[k]                       // L
            out[3 * k + 1] = v[(k * 3 + 1) % v.size]
            out[3 * k + 2] = v[(k * 5 + 2) % v.size]
        }
        return out
    }

    @Test
    fun nativeLabIsBitExact() {
        report.put("native_available", NativeLab.available)
        report.put("native_version", NativeLab.version)
        report.put("useNative_default", LabConv.useNative)
        println("[native] available=${NativeLab.available} version=${NativeLab.version}")

        // .so 必须真的加载成功（ABI/打包/16KB page size 任一出错都要在这里红）
        assertTrue("libcolortrace.so 未加载——native 路径没被验证", NativeLab.available)
        assertTrue("LabConv.useNative 默认为 false", LabConv.useNative)

        val rgb = rgbSamples()
        val lab = labSamples()
        println("[native] samples rgb=${rgb.size / 3}px lab=${lab.size / 3}px")

        val labNative = NativeLab.rgbToLab(rgb)!!
        val labKotlin = LabConv.rgbToLabKotlin(rgb)
        val rgbNative = NativeLab.labToRgb(lab)!!
        val rgbKotlin = LabConv.labToRgbKotlin(lab)

        val mLab = mismatches(labNative, labKotlin)
        val mRgb = mismatches(rgbNative, rgbKotlin)
        report.put("rgb_to_lab_mismatch", mLab.size)
        report.put("lab_to_rgb_mismatch", mRgb.size)
        report.put("rgb_to_lab_max_abs", maxAbsDiff(labNative, labKotlin).toDouble())
        report.put("lab_to_rgb_max_abs", maxAbsDiff(rgbNative, rgbKotlin).toDouble())
        println("[native] rgb→lab 逐位失配=${mLab.size} max|Δ|=${maxAbsDiff(labNative, labKotlin)}")
        println("[native] lab→rgb 逐位失配=${mRgb.size} max|Δ|=${maxAbsDiff(rgbNative, rgbKotlin)}")

        if (mLab.isNotEmpty()) {
            val i = mLab[0]
            println("[native] 首个 rgb→lab 失配 p=${i / 3} native=${labNative[i]} kotlin=${labKotlin[i]}")
        }
        if (mRgb.isNotEmpty()) {
            val i = mRgb[0]
            println("[native] 首个 lab→rgb 失配 p=${i / 3} native=${rgbNative[i]} kotlin=${rgbKotlin[i]}")
        }

        assertEquals("rgbToLab 必须与 Kotlin 逐位一致", 0, mLab.size)
        assertEquals("labToRgb 必须与 Kotlin 逐位一致", 0, mRgb.size)
    }

    @Test
    fun dispatchGoesThroughNative() {
        // 公开入口必须真的走 native：同一组输入与 Kotlin 参考结果逐位相等，
        // 且 useNative=true 时结果与直接调 NativeLab 相同（防止"接了但没接上"）。
        val rgb = rgbSamples()
        val viaPublic = LabConv.rgbToLab(rgb)
        val viaNative = NativeLab.rgbToLab(rgb)!!
        val viaKotlin = LabConv.rgbToLabKotlin(rgb)
        assertEquals("公开入口与 NativeLab 不一致", 0, mismatches(viaPublic, viaNative).size)
        assertEquals("公开入口与 Kotlin 不一致", 0, mismatches(viaPublic, viaKotlin).size)

        val lab = labSamples()
        assertEquals("labToRgb 公开入口与 NativeLab 不一致", 0,
            mismatches(LabConv.labToRgb(lab), NativeLab.labToRgb(lab)!!).size)

        // 回退路径本身必须是可用的（native 挂了要能顶上）
        LabConv.useNative = false
        try {
            assertEquals("回退 Kotlin 路径结果变了", 0,
                mismatches(LabConv.rgbToLab(rgb), viaKotlin).size)
            assertEquals("回退 Kotlin 路径 labToRgb 结果变了", 0,
                mismatches(LabConv.labToRgb(lab), LabConv.labToRgbKotlin(lab)).size)
        } finally {
            LabConv.useNative = true
        }
        report.put("dispatch_ok", true)
    }

    /**
     * 同口径 A/B 计时（同一数组、同一进程、warmup 后取 3 次均值）。
     *
     * 断言只放"native 不能比 Kotlin 慢"这一条底线——具体倍数随机器/ABI 波动，
     * 写成硬指标会变成 flaky 关卡；倍数进报告供人看。
     */
    @Test
    fun nativeLabIsFaster() {
        val px = 1 shl 17                       // 131072 像素（Kotlin 单次 ~0.3s 量级）
        val rnd = Random(11L)
        val rgb = FloatArray(px * 3) { rnd.nextFloat() }
        val lab = LabConv.rgbToLabKotlin(rgb)

        fun bench(block: () -> Any?): Double {
            block()                              // warmup（JIT/缓存）
            val t = System.nanoTime()
            repeat(3) { block() }
            return (System.nanoTime() - t) / 3e6 / 3.0
        }

        val ktLab = bench { LabConv.rgbToLabKotlin(rgb) }
        val ntLab = bench { NativeLab.rgbToLab(rgb)!! }
        val ktRgb = bench { LabConv.labToRgbKotlin(lab) }
        val ntRgb = bench { NativeLab.labToRgb(lab)!! }

        val upLab = ktLab / ntLab
        val upRgb = ktRgb / ntRgb
        report.put("bench_px", px)
        report.put("rtolab_kotlin_ms", ktLab)
        report.put("rtolab_native_ms", ntLab)
        report.put("rtolab_speedup", upLab)
        report.put("labtorgb_kotlin_ms", ktRgb)
        report.put("labtorgb_native_ms", ntRgb)
        report.put("labtorgb_speedup", upRgb)
        println("[native] ${px}px rgbToLab  kotlin=${"%.1f".format(ktLab)}ms " +
                "native=${"%.1f".format(ntLab)}ms  ${"%.2f".format(upLab)}x")
        println("[native] ${px}px labToRgb  kotlin=${"%.1f".format(ktRgb)}ms " +
                "native=${"%.1f".format(ntRgb)}ms  ${"%.2f".format(upRgb)}x")

        assertTrue("native rgbToLab 比 Kotlin 慢（${ktLab} vs ${ntLab}）", ntLab <= ktLab)
        assertTrue("native labToRgb 比 Kotlin 慢（${ktRgb} vs ${ntRgb}）", ntRgb <= ktRgb)
    }

    @org.junit.After
    fun tearDown() {
        runCatching {
            File(ctx.filesDir, "native_lab_report.json").writeText(report.toString(2))
        }
        Log.i("NativeLabTest", "report=$report")
    }
}