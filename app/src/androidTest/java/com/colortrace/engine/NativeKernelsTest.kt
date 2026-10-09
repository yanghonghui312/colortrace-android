package com.colortrace.engine

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.Random

/**
 * native 内核第二批的逐位关卡（P2.9）：`kernels.cpp` 的 LUT 三线性查表与四区融合，
 * 必须与纯 Kotlin 参考实现给出**相同的 float32 位模式**。
 *
 * 覆盖口径：
 *  - LUT：33³ 随机表 + 输入含网格节点/中点/越界（0 与 1 的 ±），另跑一次 size=65；
 *  - 融合：protect ∈ {0, 0.5, 0.999, 1} × strength ∈ {0, 0.5, 1} 全组合
 *    （0 与 ≥1 会走 withStrength 的两个别名分支，0.999 走 skinIsRaw 分支）。
 *
 * 唯一放宽同 NativeLabTest：±0.0 不算漂移。
 */
@RunWith(AndroidJUnit4::class)
class NativeKernelsTest {

    companion object {
        private val report = JSONObject()
    }

    private val ctx get() = InstrumentationRegistry.getInstrumentation().targetContext

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

    /** 网格节点 + 中点 + 越界，专门打满 floor/clamp/插值权重各分支。 */
    private fun lutInputs(size: Int, n: Int, rnd: Random): FloatArray {
        val last = size - 1
        val out = FloatArray(n * 3)
        for (p in 0 until n) {
            for (c in 0 until 3) {
                val v: Float = when (p % 4) {
                    0 -> (p % size).toFloat() / last          // 正好落在网格节点
                    1 -> (p % size + 0.5f) / last             // 正中点（f=0.5）
                    2 -> rnd.nextFloat() * 1.3f - 0.15f       // 越界 ±
                    else -> if (p % 8 == 3) 0f else 1f        // 边界 0/1
                }
                out[3 * p + c] = v
            }
        }
        return out
    }

    @Test
    fun lutApplyIsBitExact() {
        assertTrue("libcolortrace.so 未加载", NativeKernels.available)
        val rnd = Random(20260929L)
        var totalMismatch = 0
        var worst = 0f
        for (size in intArrayOf(33, 65, 2)) {
            val table = DoubleArray(size * size * size * 3) { rnd.nextDouble() }
            val lut = Lut3D(size, table)                 // 构造内已按桌面口径 clip
            val rgb = lutInputs(size, 2000, rnd)
            val kt = lut.applyKotlin(rgb)
            val nt = FloatArray(rgb.size)
            assertTrue("native lutApply 返回 false (size=$size)",
                NativeKernels.lutApply(lut.table, size, rgb, nt))
            val m = mismatches(nt, kt)
            totalMismatch += m.size
            val d = maxAbsDiff(nt, kt)
            if (d > worst) worst = d
            println("[kernels] lutApply size=$size 逐位失配=${m.size}/${kt.size} max|Δ|=$d")
            if (m.isNotEmpty()) {
                val i = m[0]
                println("[kernels] 首个失配 i=$i native=${nt[i]} kotlin=${kt[i]}")
            }
        }
        report.put("lut_mismatch", totalMismatch)
        report.put("lut_max_abs", worst.toDouble())
        assertEquals("lutApply 必须与 Kotlin 逐位一致", 0, totalMismatch)
    }

    @Test
    fun fuseBandIsBitExact() {
        assertTrue("libcolortrace.so 未加载", NativeKernels.available)
        val rnd = Random(7L)
        val n = 3000
        val n3 = n * 3
        val rgb = FloatArray(n3) { rnd.nextFloat() * 1.4f - 0.2f }
        val mapped = FloatArray(n3) { rnd.nextFloat() }
        // 4 个权重场：逐像素归一（Σ=1），量级与真实分区权重一致
        val ws = Array(4) { FloatArray(n) }
        for (p in 0 until n) {
            var s = 0f
            for (r in 0 until 4) { ws[r][p] = rnd.nextFloat(); s += ws[r][p] }
            for (r in 0 until 4) ws[r][p] /= s
        }

        var totalMismatch = 0
        var worst = 0f
        for (protect in floatArrayOf(0f, 0.5f, 0.999f, 1f)) {
            for (strength in floatArrayOf(0f, 0.5f, 1f)) {
                val kt = FloatArray(n3)
                RegionEngine.applyBandMappedKotlin(rgb, mapped, ws[0], ws[1], ws[2], ws[3],
                                                   protect, strength, kt, n)
                val nt = FloatArray(n3) { -999f }        // 哨兵：native 没写就露馅
                val ok = NativeKernels.fuseBand(rgb, mapped, ws[0], ws[1], ws[2], ws[3],
                                                protect, strength, nt)
                assertTrue("native fuseBand 返回 false", ok)
                val m = mismatches(nt, kt)
                totalMismatch += m.size
                val d = maxAbsDiff(nt, kt)
                if (d > worst) worst = d
                println("[kernels] fuseBand protect=$protect strength=$strength " +
                        "逐位失配=${m.size}/$n3 max|Δ|=$d")
                if (m.isNotEmpty()) {
                    val i = m[0]
                    println("[kernels] 首个失配 i=$i native=${nt[i]} kotlin=${kt[i]}")
                }
            }
        }
        report.put("fuse_mismatch", totalMismatch)
        report.put("fuse_max_abs", worst.toDouble())
        assertEquals("fuseBand 必须与 Kotlin 逐位一致", 0, totalMismatch)
    }

    /**
     * rgbToBgrU8 逐位关卡（P2.12 流式导出）：float RGB → u8 BGR 字节流必须与
     * render/f32ToBitmap 的 Kotlin u8 式完全一致。边界覆盖：0/1/越界±/NaN/负数。
     */
    @Test
    fun rgbToBgrU8IsBitExact() {
        assertTrue("libcolortrace.so 未加载", NativeKernels.available)
        val rnd = Random(33L)
        val n = 4096
        val rgb = FloatArray(n * 3)
        for (i in rgb.indices) {
            rgb[i] = when (i % 7) {
                0 -> rnd.nextFloat() * 1.4f - 0.2f    // 常规 + 越界 ±
                1 -> 0f
                2 -> 1f
                3 -> -0.25f
                4 -> 1.25f
                5 -> Float.NaN                        // JVM toInt(NaN)=0
                else -> rnd.nextFloat()
            }
        }
        val nt = ByteArray(n * 3) { -1 }              // 哨兵：native 没写就露馅
        assertTrue("native rgbToBgrU8 返回 false", NativeKernels.rgbToBgrU8(rgb, nt))
        fun u8(v: Float): Byte = ((v.coerceIn(0f, 1f) * 255f + 0.5f).toInt()).toByte()
        var diffs = 0
        var firstBad = -1
        for (p in 0 until n) {
            val b = u8(rgb[3 * p + 2]); val g = u8(rgb[3 * p + 1]); val r = u8(rgb[3 * p])
            if (nt[3 * p] != b || nt[3 * p + 1] != g || nt[3 * p + 2] != r) {
                diffs++; if (firstBad < 0) firstBad = p
            }
        }
        println("[kernels] rgbToBgrU8 逐位失配=$diffs/$n")
        if (diffs > 0) {
            val p = firstBad
            println("[kernels] 首个失配 p=$p rgb=(${rgb[3 * p]},${rgb[3 * p + 1]},${rgb[3 * p + 2]}) " +
                    "native=(${nt[3 * p]},${nt[3 * p + 1]},${nt[3 * p + 2]})")
        }
        assertEquals("rgbToBgrU8 必须与 Kotlin u8 式逐位一致", 0, diffs)
    }

    /**
     * develop 整段 native 内核的逐位关卡（P2.16）：`kernels.cpp` 的 developCore
     * （RGB→LAB→mapLab→RGB 合成一趟）必须与纯 Kotlin 参考实现给出相同的
     * float32 位模式——这是"治本拖微调卡顿"的前提，绝不能以牺牲口径换速度。
     *
     * 用例覆盖：全 17 参数非默认（曝光/伽马走 pow、分离色调走三角函数、
     * 自然饱和度走 sqrt）、只动 L 的影调组、只动 a/b 的色彩组、分离色调组，
     * 输入含越界 ±（覆盖 clamp01 与 fCie/invF 的线性段）。
     */
    @Test
    fun developIsBitExact() {
        assertTrue("libcolortrace.so 未加载", NativeKernels.available)
        val rnd = Random(20260930L)
        val n = 4096
        val rgb = FloatArray(n * 3) { i ->
            when (i % 9) {
                0 -> 0f
                1 -> 1f
                2 -> -0.2f                       // 越界下
                3 -> 1.3f                        // 越界上
                else -> rnd.nextFloat()
            }
        }
        val cases = listOf(
            "all17" to mapOf(
                "exposure" to 1.25f, "contrast" to 42f, "highlights" to -35f,
                "shadows" to 55f, "whites" to 20f, "blacks" to -18f,
                "temp" to 33f, "tint" to -25f, "saturation" to 40f,
                "vibrance" to -60f, "hue_shift" to 75f,
                "split_high_hue" to 120f, "split_high_sat" to 45f,
                "split_low_hue" to 280f, "split_low_sat" to 30f,
                "split_balance" to 35f, "gamma" to 1.4f),
            "tone_only" to mapOf("exposure" to -2f, "contrast" to 80f,
                                 "shadows" to 100f, "gamma" to 0.7f),
            "color_only" to mapOf("temp" to -70f, "saturation" to -50f,
                                  "vibrance" to 80f, "hue_shift" to -140f),
            "split_only" to mapOf("split_high_hue" to 30f, "split_high_sat" to 100f,
                                  "split_low_hue" to 210f, "split_low_sat" to 100f,
                                  "split_balance" to -100f),
        )
        var totalMismatch = 0
        for ((name, params) in cases) {
            val dev = DevelopTransform(params)
            val kt = dev.applyKotlin(rgb)
            val nt = FloatArray(rgb.size) { -999f }   // 哨兵：native 没写就露馅
            assertTrue("native developApply 返回 false ($name)",
                NativeKernels.developApply(rgb, dev.paramVector, nt))
            val m = mismatches(nt, kt)
            totalMismatch += m.size
            val d = maxAbsDiff(nt, kt)
            println("[kernels] develop $name 逐位失配=${m.size}/${kt.size} max|Δ|=$d")
            if (m.isNotEmpty()) {
                val i = m[0]
                println("[kernels] 首个失配 i=$i native=${nt[i]} kotlin=${kt[i]}")
            }
        }
        report.put("develop_mismatch", totalMismatch)
        assertEquals("developApply 必须与 Kotlin 逐位一致", 0, totalMismatch)
    }

    /** 同口径 A/B 计时（断言只留"不能更慢"的底线，倍数进报告）。 */
    @Test
    fun kernelsAreFaster() {
        val rnd = Random(11L)
        val px = 1 shl 17
        val rgb = FloatArray(px * 3) { rnd.nextFloat() }
        val mapped = FloatArray(px * 3) { rnd.nextFloat() }
        val size = 33
        val table = DoubleArray(size * size * size * 3) { rnd.nextDouble() }
        val lut = Lut3D(size, table)
        val n = px
        val ws = Array(4) { FloatArray(n) { 0.25f } }
        val outKt = FloatArray(px * 3)
        val outNt = FloatArray(px * 3)

        fun bench(block: () -> Unit): Double {
            block()
            val t = System.nanoTime()
            repeat(3) { block() }
            return (System.nanoTime() - t) / 3e6 / 3.0
        }

        val ktLut = bench { lut.applyKotlin(rgb) }
        val ntLut = bench { NativeKernels.lutApply(table, size, rgb, outNt) }
        val ktFuse = bench {
            RegionEngine.applyBandMappedKotlin(rgb, mapped, ws[0], ws[1], ws[2], ws[3],
                                               1f, 0.5f, outKt, n)
        }
        val ntFuse = bench {
            NativeKernels.fuseBand(rgb, mapped, ws[0], ws[1], ws[2], ws[3],
                                   1f, 0.5f, outNt)
        }
        val upLut = ktLut / ntLut
        val upFuse = ktFuse / ntFuse
        report.put("lut_kotlin_ms", ktLut)
        report.put("lut_native_ms", ntLut)
        report.put("lut_speedup", upLut)
        report.put("fuse_kotlin_ms", ktFuse)
        report.put("fuse_native_ms", ntFuse)
        report.put("fuse_speedup", upFuse)
        println("[kernels] ${px}px lutApply kotlin=${"%.1f".format(ktLut)}ms " +
                "native=${"%.1f".format(ntLut)}ms ${"%.2f".format(upLut)}x")
        println("[kernels] ${px}px fuseBand kotlin=${"%.1f".format(ktFuse)}ms " +
                "native=${"%.1f".format(ntFuse)}ms ${"%.2f".format(upFuse)}x")

        assertTrue("native lutApply 比 Kotlin 慢（$ktLut vs $ntLut）", ntLut <= ktLut)
        assertTrue("native fuseBand 比 Kotlin 慢（$ktFuse vs $ntFuse）", ntFuse <= ktFuse)

        // develop：old = 接入 native 之前的真实链路（native LAB + Kotlin mapLab），
        // kt = 纯 Kotlin 参考（含纯 Kotlin LAB 往返），nt = 现在的 apply()。
        val dev = DevelopTransform(mapOf(
            "exposure" to 0.75f, "contrast" to 28f, "highlights" to -22f,
            "shadows" to 35f, "temp" to 18f, "saturation" to 25f,
            "vibrance" to 20f, "hue_shift" to 12f,
            "split_high_sat" to 15f, "split_low_sat" to 10f,
            "split_balance" to 8f, "gamma" to 1.1f))
        val oldDev = bench { LabConv.labToRgb(dev.mapLab(LabConv.rgbToLab(rgb))) }
        val ktDev = bench { dev.applyKotlin(rgb) }
        val ntDev = bench { dev.apply(rgb) }
        report.put("develop_old_ms", oldDev)
        report.put("develop_kotlin_ms", ktDev)
        report.put("develop_native_ms", ntDev)
        report.put("develop_speedup_vs_old", oldDev / ntDev)
        println("[kernels] ${px}px develop old=${"%.1f".format(oldDev)}ms " +
                "kotlin=${"%.1f".format(ktDev)}ms native=${"%.1f".format(ntDev)}ms " +
                "${"%.2f".format(oldDev / ntDev)}x vs old")
        assertTrue("native developApply 比接入前的老链路慢（$oldDev vs $ntDev）",
                   ntDev <= oldDev)

        // ---- 2026-10-02 性能轮第二轮：探针循环 / regionStatBand / chromaAnchor ----
        // 探针：33³ 全量 × hidden=128（批量每张重建 LUT 的真实规模）
        val nProbes = 33 * 33 * 33
        val probes = DoubleArray(nProbes * 3) { rnd.nextDouble() }
        val pkC = FlowWeights.fromFlat(DoubleArray(8 * 128 + 3) { rnd.nextDouble() * 2 - 1 }, 128)
        val pkS = FlowWeights.fromFlat(DoubleArray(8 * 128 + 3) { rnd.nextDouble() * 2 - 1 }, 128)
        val ktProbe = bench {
            val p = probes.copyOf()
            val x = DoubleArray(3)
            var qi = 0
            for (i in 0 until nProbes) {
                x[0] = p[qi]; x[1] = p[qi + 1]; x[2] = p[qi + 2]
                transfer(x, pkC, pkS, 1.0, 1)
                p[qi] = x[0]; p[qi + 1] = x[1]; p[qi + 2] = x[2]
                qi += 3
            }
        }
        val ntProbe = bench {
            val p = probes.copyOf()
            NativeKernels.flowProbes(p, pkC.pack(), pkS.pack(), 128, 1.0, 1)
        }
        report.put("probes_kotlin_ms", ktProbe)
        report.put("probes_native_ms", ntProbe)
        report.put("probes_speedup", ktProbe / ntProbe)
        println("[kernels] 33³ probes kotlin=${"%.1f".format(ktProbe)}ms " +
                "native=${"%.1f".format(ntProbe)}ms ${"%.2f".format(ktProbe / ntProbe)}x")
        assertTrue("native flowProbes 比 Kotlin 循环慢（$ktProbe vs $ntProbe）",
                   ntProbe <= ktProbe)

        // regionStatBand：lab 与 ot 两 plan 的 A/B
        for (isLab in booleanArrayOf(true, false)) {
            fun d3(): DoubleArray = DoubleArray(3) { rnd.nextDouble() * 60.0 + 20.0 }
            fun d3c(): DoubleArray = DoubleArray(3) { rnd.nextDouble() * 40.0 - 20.0 }
            val global: ContentModel =
                if (isLab) ContentModel.LabStats(d3(), d3(), 1f)
                else ContentModel.OtLinear(d3c(),
                    Array(3) { r -> DoubleArray(3) { c -> if (r == c) 0.04 else 0.0 } }, 1f)
            val model = ContentModel.RegionStat(global,
                mapOf("skin" to global, "hair" to null, "cloth" to global, "bg" to global),
                emptyMap(), "global", 1f)
            val srcCov = if (isLab) null
                else Array(4) { Array(3) { r -> DoubleArray(3) { c -> if (r == c) 0.04 else 0.0 } } }
            val plan = RegionStatEngine.buildPlan(model,
                Array(4) { d3c() }, if (isLab) Array(4) { d3() } else null, srcCov)
            val ktR = bench {
                RegionStatEngine.applyBandKotlin(plan, rgb, mapped, ws[0], ws[1], ws[2],
                                                 ws[3], 1f, 1f, outKt, n)
            }
            val ntR = bench {
                NativeKernels.regionStatBand(rgb, mapped, ws[0], ws[1], ws[2], ws[3],
                                             plan.packed, 1f, 1f, outNt)
            }
            report.put("regionstat${if (isLab) "_lab" else "_ot"}_kotlin_ms", ktR)
            report.put("regionstat${if (isLab) "_lab" else "_ot"}_native_ms", ntR)
            report.put("regionstat${if (isLab) "_lab" else "_ot"}_speedup", ktR / ntR)
            println("[kernels] ${n}px regionStatBand(${if (isLab) "lab" else "ot"}) " +
                    "kotlin=${"%.1f".format(ktR)}ms native=${"%.1f".format(ntR)}ms " +
                    "${"%.2f".format(ktR / ntR)}x")
            assertTrue("native regionStatBand 比 Kotlin 慢（$ktR vs $ntR）", ntR <= ktR)
        }

        // chromaAnchor：Kotlin 全流程（rgbToLab×2 + of + anchor + labToRgb）vs native
        val ktCh = bench {
            val labIn = LabConv.rgbToLab(rgb)
            val labOut = LabConv.rgbToLab(outKt)      // 当 globalS 用（对拍口径）
            val w = SkinWeight.of(labIn)
            LabConv.labToRgb(SkinWeight.anchorChroma(labIn, labOut, w, 1f))
        }
        val ntCh = bench {
            NativeKernels.chromaAnchor(rgb, outKt, 1f, outNt)
        }
        report.put("chroma_kotlin_ms", ktCh)
        report.put("chroma_native_ms", ntCh)
        report.put("chroma_speedup", ktCh / ntCh)
        println("[kernels] ${n}px chromaAnchor kotlin=${"%.1f".format(ktCh)}ms " +
                "native=${"%.1f".format(ntCh)}ms ${"%.2f".format(ktCh / ntCh)}x")
        assertTrue("native chromaAnchor 比 Kotlin 慢（$ktCh vs $ntCh）", ntCh <= ktCh)
    }

    /** 2026-10-01 性能轮：migrate 源段 ARGB→RGB f32(+u8) 的 native 化对拍。 */
    @Test
    fun argbToRgbF32IsBitExact() {
        val n = 200_001               // 非整块：覆盖并行区间的边界与不完整块
        val rnd = Random(42L)
        val argb = IntArray(n) { rnd.nextInt() }   // 全位随机（含 alpha≠FF 的负 Int）
        argb[0] = 0xFF000000.toInt(); argb[1] = 0; argb[2] = -1
        argb[3] = 0xFF7F0182.toInt(); argb[4] = 0x017F82FF

        val ktF = FloatArray(n * 3); val ktU = ByteArray(n * 3)
        for (p in 0 until n) {
            val v = argb[p]
            ktF[3 * p] = (v shr 16 and 0xFF) / 255f
            ktF[3 * p + 1] = (v shr 8 and 0xFF) / 255f
            ktF[3 * p + 2] = (v and 0xFF) / 255f
            ktU[3 * p] = (v shr 16 and 0xFF).toByte()
            ktU[3 * p + 1] = (v shr 8 and 0xFF).toByte()
            ktU[3 * p + 2] = (v and 0xFF).toByte()
        }
        val ntF = FloatArray(n * 3); val ntU = ByteArray(n * 3)
        assertTrue("argbToRgbF32 native 失败", NativeKernels.argbToRgbF32(argb, n, ntF, ntU))
        assertEquals("f32 逐位（含 /255f 的除法舍入）", 0, mismatches(ktF, ntF).size)
        assertTrue("u8 逐字节", ktU.contentEquals(ntU))

        // u8 侧可空：只出 f32
        val ntF2 = FloatArray(n * 3)
        assertTrue(NativeKernels.argbToRgbF32(argb, n, ntF2, null))
        assertEquals(0, mismatches(ktF, ntF2).size)
    }

    /** 2026-10-01 性能轮：OFF 档 withStrength 的 native 化对拍（(0,1) 开区间）。 */
    @Test
    fun strengthMixIsBitExact() {
        val n = 150_000
        val rnd = Random(7L)
        val rgb = FloatArray(n * 3) { rnd.nextFloat() }
        val mapped = FloatArray(n * 3) { rnd.nextFloat() }
        val nt = FloatArray(n * 3)
        for (s in floatArrayOf(0.001f, 0.25f, 0.5f, 0.7f, 0.999f)) {
            val kt = EncoderEngine.withStrength(rgb, mapped, s)
            assertTrue("s=$s native 失败", NativeKernels.strengthMix(rgb, mapped, s, nt))
            assertEquals("s=$s 逐位", 0, mismatches(kt, nt).size)
        }
    }

    /**
     * 2026-10-02 性能轮：33³ 探针循环的 native 化对拍。Kotlin 参考 =
     * `EncoderEngine.lutFor` 的逐探针 `transfer` 循环（double 全程）。
     * (strength, steps) 覆盖 encoder 实战档 (1.0, 1) 与 flow 通用档。
     */
    @Test
    fun flowProbesIsBitExact() {
        assertTrue("libcolortrace.so 未加载", NativeKernels.available)
        val rnd = Random(20261002L)
        val hidden = 128
        fun randomWeights(): FlowWeights =
            FlowWeights.fromFlat(DoubleArray(8 * hidden + 3) {
                rnd.nextDouble() * 2.0 - 1.0
            }, hidden)
        val wC = randomWeights()
        val wS = randomWeights()
        val n = 2001                                    // 非整块：覆盖并行区间边界
        val base = DoubleArray(n * 3) { rnd.nextDouble() }

        var totalMismatch = 0
        for ((strength, steps) in listOf(1.0 to 1, 0.5 to 8, 1.0 to 4)) {
            val kt = base.copyOf()
            val x = DoubleArray(3)
            var qi = 0
            for (i in 0 until n) {
                x[0] = kt[qi]; x[1] = kt[qi + 1]; x[2] = kt[qi + 2]
                transfer(x, wC, wS, strength, steps)
                kt[qi] = x[0]; kt[qi + 1] = x[1]; kt[qi + 2] = x[2]
                qi += 3
            }
            val nt = base.copyOf()
            assertTrue("strength=$strength steps=$steps native flowProbes 失败",
                NativeKernels.flowProbes(nt, wC.pack(), wS.pack(), hidden, strength, steps))
            var bad = 0
            var first = -1
            for (i in kt.indices) {
                if (kt[i].toRawBits() != nt[i].toRawBits()) {
                    bad++; if (first < 0) first = i
                }
            }
            println("[kernels] flowProbes s=$strength steps=$steps 逐位失配=$bad/${kt.size}")
            if (first >= 0) {
                println("[kernels] 首个失配 i=$first kotlin=${kt[first]} native=${nt[first]}")
            }
            totalMismatch += bad
        }
        assertEquals("flowProbes 必须与 Kotlin transfer 逐位一致", 0, totalMismatch)
    }

    /**
     * 2026-10-02 性能轮：分区统计档段级内核的 native 化对拍。lab / ot 两路径 ×
     * protect{0,0.5,0.999,1} × strength{0,0.5,1}，嵌套强度含 0.7（混合）、
     * 0（回原图）、1（直通）三分支；hair=null 覆盖"缺席区回退全局 ref"。
     */
    @Test
    fun regionStatBandIsBitExact() {
        assertTrue("libcolortrace.so 未加载", NativeKernels.available)
        val rnd = Random(1002L)
        val n = 3000
        val n3 = n * 3
        val rgb = FloatArray(n3) { rnd.nextFloat() * 1.2f - 0.1f }
        val mapped = FloatArray(n3) { rnd.nextFloat() }
        val ws = Array(4) { FloatArray(n) }
        for (p in 0 until n) {
            var s = 0f
            for (r in 0 until 4) { ws[r][p] = rnd.nextFloat(); s += ws[r][p] }
            for (r in 0 until 4) ws[r][p] /= s
        }

        fun d3(): DoubleArray = DoubleArray(3) { rnd.nextDouble() * 60.0 + 20.0 }
        fun d3c(): DoubleArray = DoubleArray(3) { rnd.nextDouble() * 40.0 - 20.0 }
        fun cov(): Array<DoubleArray> {
            val m = Array(3) { DoubleArray(3) }
            for (r in 0 until 3) for (c in 0 until 3) {
                m[r][c] = (rnd.nextDouble() * 0.04 - 0.02 + m[r][c]) / 2.0
            }
            for (r in 0 until 3) for (c in 0 until 3) m[r][c] = (m[r][c] + m[c][r]) / 2.0
            for (d in 0 until 3) m[d][d] += 0.05
            return m
        }

        var totalMismatch = 0
        var worst = 0f
        for (isLab in booleanArrayOf(true, false)) {
            val global: ContentModel =
                if (isLab) ContentModel.LabStats(d3(), d3(), 1f)
                else ContentModel.OtLinear(d3c(), cov(), 1f)
            val regions = mapOf(
                "skin" to global,
                "hair" to null,
                "cloth" to (if (isLab) ContentModel.LabStats(d3(), d3(), 0.7f)
                            else ContentModel.OtLinear(d3c(), cov(), 0.7f)),
                "bg" to (if (isLab) ContentModel.LabStats(d3(), d3(), 0f)
                         else ContentModel.OtLinear(d3c(), cov(), 0f)),
            )
            val model = ContentModel.RegionStat(global, regions, emptyMap(), "global", 1f)
            val srcMean = Array(4) { d3c() }
            val plan = RegionStatEngine.buildPlan(model, srcMean,
                if (isLab) Array(4) { d3() } else null,
                if (isLab) null else Array(4) { cov() })

            for (protect in floatArrayOf(0f, 0.5f, 0.999f, 1f)) {
                for (strength in floatArrayOf(0f, 0.5f, 1f)) {
                    val kt = FloatArray(n3)
                    RegionStatEngine.applyBandKotlin(plan, rgb, mapped, ws[0], ws[1],
                        ws[2], ws[3], protect, strength, kt, n)
                    val nt = FloatArray(n3) { -999f }    // 哨兵：native 没写就露馅
                    assertTrue("isLab=$isLab native regionStatBand 失败",
                        NativeKernels.regionStatBand(rgb, mapped, ws[0], ws[1], ws[2],
                            ws[3], plan.packed, protect, strength, nt))
                    val m = mismatches(nt, kt)
                    totalMismatch += m.size
                    val d = maxAbsDiff(nt, kt)
                    if (d > worst) worst = d
                    println("[kernels] regionStatBand isLab=$isLab protect=$protect " +
                            "strength=$strength 逐位失配=${m.size}/$n3 max|Δ|=$d")
                    if (m.isNotEmpty()) {
                        val i = m[0]
                        println("[kernels] 首个失配 i=$i native=${nt[i]} kotlin=${kt[i]}")
                    }
                }
            }
        }
        report.put("regionstat_mismatch", totalMismatch)
        report.put("regionstat_max_abs", worst.toDouble())
        assertEquals("regionStatBand 必须与 Kotlin 逐位一致", 0, totalMismatch)
    }

    /**
     * 2026-10-02 性能轮：chroma 锚定整段的 native 化对拍。覆盖 protect{0,0.25,
     * 0.5,1}（0 = 纯 LAB 往返）与输入越界 ±（LAB 裁剪分支）。
     */
    @Test
    fun chromaAnchorIsBitExact() {
        assertTrue("libcolortrace.so 未加载", NativeKernels.available)
        val rnd = Random(3L)
        val n = 4096
        val rgb = FloatArray(n * 3) { i ->
            when (i % 9) {
                0 -> 0f
                1 -> 1f
                2 -> -0.2f                       // 越界下
                3 -> 1.3f                        // 越界上
                else -> rnd.nextFloat()
            }
        }
        val globalS = FloatArray(n * 3) { rnd.nextFloat() * 1.2f - 0.1f }

        var totalMismatch = 0
        for (protect in floatArrayOf(0f, 0.25f, 0.5f, 1f)) {
            val kt = run {
                val labIn = LabConv.rgbToLab(rgb)
                val labOut = LabConv.rgbToLab(globalS)
                val weight = SkinWeight.of(labIn)
                LabConv.labToRgb(SkinWeight.anchorChroma(labIn, labOut, weight, protect))
            }
            val nt = FloatArray(rgb.size) { -999f }
            assertTrue("protect=$protect native chromaAnchor 失败",
                NativeKernels.chromaAnchor(rgb, globalS, protect, nt))
            val m = mismatches(nt, kt)
            totalMismatch += m.size
            val d = maxAbsDiff(nt, kt)
            println("[kernels] chromaAnchor protect=$protect 逐位失配=${m.size}/${kt.size} max|Δ|=$d")
            if (m.isNotEmpty()) {
                val i = m[0]
                println("[kernels] 首个失配 i=$i native=${nt[i]} kotlin=${kt[i]}")
            }
        }
        report.put("chroma_mismatch", totalMismatch)
        assertEquals("chromaAnchor 必须与 Kotlin 逐位一致", 0, totalMismatch)
    }

    /** 2026-10-02 性能轮：统计类内容归约的 native 化对拍（分块并行 vs Kotlin
     *  顺序累加参考；归约序不同 ⇒ 相对误差 ~1e-12，锁 <1e-9）。 */
    @Test
    fun statReductionMatchesKotlin() {
        val rnd = Random(20261002L)
        val n = 1 shl 18
        val rgb = FloatArray(n * 3) { rnd.nextFloat() }
        val mu = DoubleArray(3) { rnd.nextDouble() }

        fun ktSum(): DoubleArray {
            val s = DoubleArray(3)
            for (p in 0 until n) {
                s[0] += rgb[3 * p].toDouble()
                s[1] += rgb[3 * p + 1].toDouble()
                s[2] += rgb[3 * p + 2].toDouble()
            }
            return s
        }
        fun ktOuter(): DoubleArray {
            val o = DoubleArray(6)
            for (p in 0 until n) {
                val d0 = rgb[3 * p].toDouble() - mu[0]
                val d1 = rgb[3 * p + 1].toDouble() - mu[1]
                val d2 = rgb[3 * p + 2].toDouble() - mu[2]
                o[0] += d0 * d0; o[1] += d0 * d1; o[2] += d0 * d2
                o[3] += d1 * d1; o[4] += d1 * d2; o[5] += d2 * d2
            }
            return o
        }
        fun ktSumLab(): DoubleArray {
            val out = DoubleArray(6)
            val lab = LabConv.rgbToLab(rgb)
            for (p in 0 until n) {
                for (c in 0 until 3) {
                    val x = lab[3 * p + c].toDouble()
                    out[c] += x; out[3 + c] += x * x
                }
            }
            return out
        }

        fun assertClose(tag: String, kt: DoubleArray, nt: DoubleArray, scale: Double) {
            for (i in kt.indices) {
                val rel = kotlin.math.abs(kt[i] - nt[i]) /
                        kotlin.math.max(kotlin.math.abs(kt[i]), scale)
                println("[kernels] stat $tag[$i] kt=${kt[i]} nt=${nt[i]} rel=$rel")
                assertTrue("$tag[$i] 相对误差 $rel 必须 <1e-9", rel < 1e-9)
            }
        }

        val ntSum = DoubleArray(3)
        assertTrue("native statSumRgb 失败", NativeKernels.statSumRgb(rgb, ntSum))
        assertClose("sumRgb", ktSum(), ntSum, n * 0.5)

        val ntOuter = DoubleArray(6)
        assertTrue("native statOuterRgb 失败", NativeKernels.statOuterRgb(rgb, mu, ntOuter))
        assertClose("outerRgb", ktOuter(), ntOuter, n * 0.2)

        val ntLab = DoubleArray(6)
        assertTrue("native statSumLab 失败", NativeKernels.statSumLab(rgb, ntLab))
        assertClose("sumLab", ktSumLab(), ntLab, n * 50.0)

        // 确定性：同输入两次结果逐位一致（并行归约的固定块序保证）
        val ntSum2 = DoubleArray(3)
        NativeKernels.statSumRgb(rgb, ntSum2)
        assertEquals("归约必须端内确定", 0,
            (0 until 3).count { ntSum[it].toRawBits() != ntSum2[it].toRawBits() })
    }

    @org.junit.After
    fun tearDown() {
        runCatching {
            File(ctx.filesDir, "native_kernels_report.json").writeText(report.toString(2))
        }
        Log.i("NativeKernelsTest", "report=$report")
    }
}