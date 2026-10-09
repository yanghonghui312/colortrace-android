package com.colortrace.engine

import com.colortrace.DebugLog
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.security.MessageDigest
import kotlin.math.ceil
import kotlin.math.min

/**
 * 胶片质感层（柔光 / 光晕 / 颗粒）的安卓渲染（桌面 `methods/film.py` 移植，
 * 2026-10-02，用户点单）。
 *
 * 与桌面的口径关系（用户拍板）：
 *  - 柔光/光晕是纯确定性算法，与桌面**同式同序**（float32；模糊用同一 OpenCV
 *    家族的 `Imgproc.GaussianBlur`）——u8 出图按 ≤1/255 金标协议对拍
 *    （`FilmTest.softBloomMatchesDesktop`）；
 *  - 颗粒是随机纹理：**不与桌面逐位**（两端各自确定性、算法/观感同构）——
 *    同款"粗网格高斯噪声 INTER_LINEAR 上采样 + 封顶 35% 像素级细噪声 +
 *    全图 std 归一（幅度 = σ·数量）"，RNG 用端内自有 splitmix64+Box-Muller，
 *    种子由 [filmSeed] 从 (h, w, 粗糙, 大小) 派生 ⇒ **端内同图同参逐位可复现**。
 *
 * 接入点（性能设计，真机基线 5.5MP 单张 ~1.2s）：
 *  - film 关闭：渲染路径**零变化**（[active] 为 false 时不分配任何额外缓冲）；
 *  - film 开启：render 输出改写**全图 f32 Mat**（预览 ~9MB / 预压缩 ~66MB，
 *    迁移会话同期释放，60MP 原始档净内存反而更低），本对象按**效果分轮 +
 *    行段 apron** 处理——柔光/光晕的高斯模糊核窗跨段（apron = 4σ+2 行），
 *    apron 内必须是本效果**未处理**的数据，故三效果各跑一轮段循环（与桌面
 *    `_map` 的 soft→bloom→grain 顺序一致）；颗粒是点态，最后跟段顺带完成。
 *  - 模糊/上采样在 Kotlin 层走 OpenCV（与桌面同实现家族）；逐像素算子与
 *    颗粒在 native（`film_kernels.cpp`）。任一环节失败 → 整层跳过并记日志
 *    （风格化叠加不阻塞出图）。
 *  - **内存口径（2026-10-03 OOM 修复）**：段缓冲与颗粒噪声一律留 native——
 *    `Mat.dataAddr()` 数据指针直传 JNI、行窗偏移在 native 侧算，Java 堆零
 *    大分配。旧实现的 Java 堆段数组在 5011×3341 全尺寸导出必 OOM（堆上限
 *    256MB）→「整层跳过」静默出无效果图，真机三连复现后改为指针版内核
 *    （与数组版核心共用、逐位同算子同序，金标继续锁数组版）。
 */
object FilmLayer {

    // ---- 桌面 film.py 的同款常量 ----
    private const val SOFT_SIGMA0 = 0.002f
    private const val SOFT_SIGMA1 = 0.022f
    private const val SOFT_T_HI = 0.85f
    private const val SOFT_T_LO = 0.15f
    private const val SOFT_BAND = 0.18f
    private const val BLOOM_SIGMA_FRAC = 0.01f
    private const val GRAIN_SIGMA_L = 2.2f
    private const val GRAIN_FINE_MAX = 0.35f
    private const val GRAIN_SIZE_K = 3.0f
    private const val GRAIN_STRIP = 1024

    /** 滑杆全 11 键的默认值（与桌面 FILM_SPEC 的用户标定值一致，2026-10-02）。 */
    val DEFAULTS: Map<String, Float> = mapOf(
        "soft_on" to 0f, "soft_strength" to 30f, "soft_range" to 60f,
        "soft_radius" to 50f,
        "bloom_on" to 0f, "bloom_strength" to 40f, "bloom_tint" to 40f,
        "grain_on" to 0f, "grain_amount" to 35f, "grain_rough" to 45f,
        "grain_size" to 30f,
    )

    /** 滑杆规格（UI 与测试共用）：中文名 / 区间 / 默认 / 小数位（与桌面
     *  FILM_SPEC 的用户标定值一致）。*_on 三键是开关，不进滑杆表。 */
    data class Spec(val key: String, val cn: String, val min: Float,
                    val max: Float, val def: Float, val decimals: Int)

    val SPEC: List<Spec> = listOf(
        Spec("soft_strength", "柔光强度", 0f, 100f, 30f, 0),
        Spec("soft_range", "柔光范围", 0f, 100f, 60f, 0),
        Spec("soft_radius", "柔光半径", 0f, 100f, 50f, 0),
        Spec("bloom_strength", "光晕强度", 0f, 100f, 40f, 0),
        Spec("bloom_tint", "光晕色调", 0f, 100f, 40f, 0),
        Spec("grain_amount", "颗粒数量", 0f, 100f, 35f, 0),
        Spec("grain_rough", "颗粒粗糙程度", 0f, 100f, 45f, 0),
        Spec("grain_size", "颗粒大小", 0f, 100f, 30f, 0),
    )
    val SPEC_BY_KEY: Map<String, Spec> = SPEC.associateBy { it.key }

    /** 页面分组（组名 → 组内键，首键 = 该组启用开关）。 */
    val GROUPS: List<Pair<String, List<String>>> = listOf(
        "柔光" to listOf("soft_on", "soft_strength", "soft_range", "soft_radius"),
        "光晕" to listOf("bloom_on", "bloom_strength", "bloom_tint"),
        "颗粒" to listOf("grain_on", "grain_amount", "grain_rough", "grain_size"),
    )

    /** 缺省键补全（快照/批量旧数据只带部分键时对齐桌面 clamp 语义）。 */
    fun normalize(params: Map<String, Float>?): Map<String, Float> {
        if (params == null) return DEFAULTS
        val out = HashMap<String, Float>(DEFAULTS)
        for ((k, v) in params) if (k in out) out[k] = v
        return out
    }

    /** 三效果任一激活（开关开且对应量 >0；颗粒 = 数量或粗糙 >0）。 */
    fun active(params: Map<String, Float>?): Boolean {
        if (params == null) return false
        val p = normalize(params)
        fun on(k: String) = (p[k] ?: 0f) >= 0.5f
        fun v(k: String) = p[k] ?: 0f
        return (on("soft_on") && v("soft_strength") > 0f) ||
                (on("bloom_on") && v("bloom_strength") > 0f) ||
                (on("grain_on") && (v("grain_rough") > 0f || v("grain_amount") > 0f))
    }

    /** 端内确定性种子（与桌面字符串格式无关——颗粒两端各自随机）。 */
    fun filmSeed(h: Int, w: Int, rough: Float, size: Float): Long {
        val src = "colortrace-film|$h x$w|r=${Math.round(rough)}|s=${Math.round(size)}"
        val d = MessageDigest.getInstance("SHA-256").digest(src.toByteArray())
        var v = 0L
        for (i in 0 until 8) v = v or ((d[i].toLong() and 0xFF) shl (8 * i))
        return v
    }

    /**
     * 对全图 f32 RGB [mat]（CV_32FC3，[0,1]，就地修改）施加三效果。
     * 返回 false = 处理失败（mat 内容不变——失败发生在第一轮写回前；**注意**
     * 部分效果成功后失败时 mat 处于"已完成前几效果"状态，调用方应整层弃用）。
     */
    fun apply(mat: Mat, params: Map<String, Float>): Boolean =
        try {
            val p = normalize(params)
            val w = mat.cols()
            val h = mat.rows()
            val tAll0 = System.nanoTime()
            var softMs = 0.0; var bloomMs = 0.0; var grainMs = 0.0
            if ((p["soft_on"] ?: 0f) >= 0.5f && (p["soft_strength"] ?: 0f) > 0f) {
                val sigma = min(h, w) * (SOFT_SIGMA0 + SOFT_SIGMA1 * (p["soft_radius"] ?: 50f) / 100f)
                val t0 = SOFT_T_HI + (SOFT_T_LO - SOFT_T_HI) * ((p["soft_range"] ?: 60f) / 100f)
                val t = System.nanoTime()
                softPass(mat, sigma, t0, SOFT_BAND, (p["soft_strength"] ?: 0f) / 100f,
                         stepFor(h, w, sigma))
                softMs = (System.nanoTime() - t) / 1e6
            }
            if ((p["bloom_on"] ?: 0f) >= 0.5f && (p["bloom_strength"] ?: 0f) > 0f) {
                val t = System.nanoTime()
                bloomPass(mat, min(h, w) * BLOOM_SIGMA_FRAC,
                          (p["bloom_strength"] ?: 0f) / 100f,
                          (p["bloom_tint"] ?: 0f) / 100f,
                          stepFor(h, w, min(h, w) * BLOOM_SIGMA_FRAC))
                bloomMs = (System.nanoTime() - t) / 1e6
            }
            if ((p["grain_on"] ?: 0f) >= 0.5f &&
                ((p["grain_rough"] ?: 0f) > 0f || (p["grain_amount"] ?: 0f) > 0f)) {
                val t = System.nanoTime()
                grainPass(mat, p, filmSeed(h, w, p["grain_rough"] ?: 45f,
                                           p["grain_size"] ?: 30f),
                          stepFor(h, w, 0f))
                grainMs = (System.nanoTime() - t) / 1e6
            }
            DebugLog.i("film| total=${"%.0f".format((System.nanoTime() - tAll0) / 1e6)}ms " +
                    "soft=${"%.0f".format(softMs)} bloom=${"%.0f".format(bloomMs)} " +
                    "grain=${"%.0f".format(grainMs)} ${w}x${h}")
            true
        } catch (e: Throwable) {
            DebugLog.e("胶片质感层失败→整层跳过", e)
            false
        }

    /**
     * 半分辨率高斯模糊（2026-10-02 用户拍板，**两端同步口径**，与桌面
     * `film._gaussian_blur_half` 同式）：INTER_AREA 缩半 → blur(σ/2) →
     * INTER_LINEAR 放回。真机探针 blur 步骤 ~6×（374→60ms @2880×1920），
     * 与全分辨率差异 maxΔ 1.4/255（u8 亚阈）。金标
     * （film_out_*.png）为同口径重锚版——`FilmTest.softBloomMatchesDesktop`
     * 继续锁两端一致。
     */
    private fun blurHalf(src: Mat, dst: Mat, sigma: Double) {
        val sm = Mat()
        val bd = Mat()
        Imgproc.resize(src, sm,
                       Size((src.cols() * 0.5).coerceAtLeast(1.0),
                            (src.rows() * 0.5).coerceAtLeast(1.0)),
                       0.0, 0.0, Imgproc.INTER_AREA)
        Imgproc.GaussianBlur(sm, bd, Size(0.0, 0.0), sigma / 2, sigma / 2)
        Imgproc.resize(bd, dst,
                       Size(src.cols().toDouble(), src.rows().toDouble()),
                       0.0, 0.0, Imgproc.INTER_LINEAR)
        sm.release()
        bd.release()
    }

    /**
     * film 轮的段高：与 render 段高解耦——段越小 apron 重叠浪费越大（原实现
     * 13 段 × rows≈147 行时 apron 2×102 行 = **2.4×** blur 面积纯浪费）。取
     * max(4×apron, ~1.2M 像素行数)：重叠占比 ≤25%、段缓冲 ~14MB 级；整图小于
     * 该值时一段跑完（与全图一次模糊同效）。
     */
    private fun stepFor(h: Int, w: Int, sigma: Float): Int {
        val apron = (ceil(4.0 * sigma) + 2).toInt()
        return min(h, maxOf(apron * 4, (1_200_000 + w - 1) / w))
    }

    // ---- 柔光：段 + apron 高斯 → native screen/mask 混合（就地写回段） ----
    //
    // 全部中间数据留 native（2026-10-03 OOM 修复）：模糊输出是 Mat、逐像素核
    // 吃 `Mat.dataAddr()` 数据指针 + 行窗偏移——Java 堆零大分配。apron 子阵走
    // rowRange 连续视图（免拷贝，GaussianBlur 直接吃）。

    private fun softPass(mat: Mat, sigma: Float, t0: Float, band: Float,
                         soft: Float, step: Int) {
        val w = mat.cols()
        val h = mat.rows()
        val apron = (ceil(4.0 * sigma) + 2).toInt()
        for (y0 in 0 until h step step) {
            val r = min(step, h - y0)
            val yA = maxOf(0, y0 - apron)
            val yB = min(h, y0 + r + apron)
            val a = y0 - yA
            // apron 子阵（rowRange 连续视图，免拷贝直接吃）→ 全段高斯（与桌面
            // 整图同实现；apron ≥ 核半径 ⇒ 核窗完全落在真实数据内，段中央与
            // 全图模糊一致）
            val dst = Mat()
            blurHalf(mat.rowRange(yA, yB), dst, sigma.toDouble())
            NativeKernels.filmSoftBandMat(mat.dataAddr(), y0, dst.dataAddr(), a,
                                          r, w, soft, t0, band)
            dst.release()
        }
    }

    // ---- 光晕：高光提取（native 直写 Mat）→ apron 高斯 → tint + screen 混合 ----

    private fun bloomPass(mat: Mat, sigma: Float, bloom: Float, tint: Float,
                          step: Int) {
        val w = mat.cols()
        val h = mat.rows()
        val apron = (ceil(4.0 * sigma) + 2).toInt()
        for (y0 in 0 until h step step) {
            val r = min(step, h - y0)
            val yA = maxOf(0, y0 - apron)
            val yB = min(h, y0 + r + apron)
            val a = y0 - yA
            val rowsSub = yB - yA
            val masked = Mat(rowsSub, w, CvType.CV_32FC3)
            val mask = Mat(rowsSub, w, CvType.CV_32FC1)
            NativeKernels.filmBloomMaskMat(mat.dataAddr(), yA, masked.dataAddr(),
                                           mask.dataAddr(), rowsSub, w)
            val dst = Mat()
            blurHalf(masked, dst, sigma.toDouble())
            masked.release()
            mask.release()
            NativeKernels.filmBloomApplyMat(mat.dataAddr(), y0, dst.dataAddr(), a,
                                            r, w, bloom, tint)
            dst.release()
        }
    }

    // ---- 颗粒：base 网格（native 正态直写 Mat）→ INTER_LINEAR 上采样 →
    //      细噪声条带 → 全图 std 归一 → 逐段 L 通道加噪（LAB 往返在 native） ----
    // 噪声全程驻留 noiseMat（native，w×h）——旧的 FloatArray(w·h) + noiseMat
    // 双份是全尺寸导出 OOM 的另一大头。

    private fun grainPass(mat: Mat, p: Map<String, Float>, seed: Long, step: Int) {
        val w = mat.cols()
        val h = mat.rows()
        val amount = (p["grain_amount"] ?: 35f) / 100f
        val rough = (p["grain_rough"] ?: 45f) / 100f
        val size = (p["grain_size"] ?: 30f) / 100f
        val k = 1.0 + GRAIN_SIZE_K * size * size
        val hk = maxOf(2, ceil(h / k).toInt())
        val wk = maxOf(2, ceil(w / k).toInt())

        val baseMat = Mat(hk, wk, CvType.CV_32FC1)
        NativeKernels.filmGrainBaseMat(seed, baseMat.dataAddr(), hk * wk)
        val noiseMat = Mat()
        Imgproc.resize(baseMat, noiseMat, Size(w.toDouble(), h.toDouble()),
                       0.0, 0.0, Imgproc.INTER_LINEAR)
        baseMat.release()

        // 细噪声条带（端内顺序与桌面条带结构同构；RNG 用派生种子续流）。
        // native 条带并行：每条带独立子流（seed+si）⇒ 与串行版逐位一致。
        val fine = GRAIN_FINE_MAX * rough
        if (fine > 0f) {
            NativeKernels.filmGrainScaleMat(noiseMat.dataAddr(), w * h, 1f - fine)
            NativeKernels.filmGrainBandsMat(seed, (h + GRAIN_STRIP - 1) / GRAIN_STRIP,
                                            GRAIN_STRIP, w, h, fine,
                                            noiseMat.dataAddr())
        }
        NativeKernels.filmGrainNormalizeMat(noiseMat.dataAddr(), w * h,
                                            GRAIN_SIGMA_L * amount)

        // 逐段 L 通道加噪（点态、就地写回；噪声按同行窗直读 noiseMat）
        for (y0 in 0 until h step step) {
            val r = min(step, h - y0)
            NativeKernels.filmGrainBandMat(mat.dataAddr(), y0, noiseMat.dataAddr(),
                                           r, w)
        }
        noiseMat.release()
    }
}
