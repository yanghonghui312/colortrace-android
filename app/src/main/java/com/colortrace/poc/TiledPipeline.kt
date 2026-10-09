package com.colortrace.poc

import android.graphics.Bitmap
import com.colortrace.DebugLog
import com.colortrace.engine.ContentModel
import com.colortrace.engine.DevelopTransform
import com.colortrace.engine.FilmLayer
import com.colortrace.engine.EncoderEngine
import com.colortrace.engine.LabConv
import com.colortrace.engine.Lut3D
import com.colortrace.engine.NativeKernels
import com.colortrace.engine.RegionEngine
import com.colortrace.engine.RegionStatEngine
import com.colortrace.engine.SkinWeight
import com.colortrace.engine.StatEngine
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.dnn.Net

/**
 * 全分辨率分块管线：Bitmap（任意 MP）→ 追色 → Bitmap，Java 堆上只存在
 * 段级数组（~0.5M 像素/段 ≈ 12MB），全分辨率大缓冲都在 OpenCV 原生内存。
 *
 * **两阶段（P2.7）**：
 *  - [migrate]：在**缩略/一次性**开销上完成内容 LUT、语义分割、全分辨率权重，
 *    并把「LUT 全强度输出」物化进 `mapped` Mat——**与 strength/develop 无关**，
 *    因此可作为会话缓存复用（对应桌面 `_mapped_full` 缓存纪律）；
 *  - [render]：逐段读 `src`+`mapped`，按 mode 重放 withStrength→融合→LAB 锚定，
 *    再叠 develop、出 u8。**拖微调/拖强度只走这里**，不重跑分割/权重。
 *
 * 逐像素数学与整图路径完全一致（引擎的点态运算原样复用），逐位一致性由
 * TiledTest 的 0 残差关卡锁住（含 `render∘migrate ≡ process`）。内存口径
 * （20MP）：原生侧 src32 240MB + mapped 240MB + 权重 Mat 4×80MB + 输出
 * Bitmap 80MB；Java 堆峰值 ~30MB。
 */
class TiledPipeline(private val model: ContentModel,
                    private val selfieNet: Net?) {

    /** encoder 会话的既有调用点零改动（P2.13 起主构造吃 [ContentModel]）。 */
    constructor(engine: EncoderEngine, selfieNet: Net?) :
            this(ContentModel.Encoder(engine), selfieNet)

    companion object {
        // ~0.5M 像素/段：float 段 ≈6MB；applyBand 峰值同时活 ~10 个段数组，
        // 192MB 的应用堆要留足余量（12MP ≈ 24 段，循环开销可忽略）
        const val BAND_PIXELS = 1 shl 19
        // 超大图软顶（50MP+ 的传感器图）：超出按比例降到这里，避免原生内存失控
        const val MAX_PIXELS = 60_000_000L

        private const val TAG = "colortrace"
        // 迁移期构造 RegionEngine 时用的占位值：prepareTiled 不使用 protect
        // 字段（保护强度只作用于 apply 期，见 render 的 protect 参数）
        private const val PROTECT_UNUSED = 1f

        private fun fmt(ms: Double): String =
            String.format(java.util.Locale.US, "%.1f", ms)
    }

    /** 阶段计时（System.nanoTime；只在段/阶段边界调用，零逐像素开销）。 */
    private class Lap {
        private var t = System.nanoTime()
        fun ms(): Double {
            val n = System.nanoTime()
            val d = (n - t) / 1e6
            t = n
            return d
        }
    }

    /**
     * 迁移会话（可缓存）：源图 float Mat + LUT 全强度输出 float Mat（+REGION
     * 档的全分辨率权重 Mat）。strength / develop 都不参与迁移——缓存后拖微调、
     * 拖强度只需重跑 [render]。内含原生内存，用完必须 [release]。
     */
    class Migration internal constructor(
        val src: Mat,          // CV_32FC3 归一化源图（withStrength 的 base）
        val mapped: Mat,       // CV_32FC3 LUT 全强度输出（与 strength 无关）
        val tiled: RegionEngine.Tiled?,  // REGION 档的 4 区权重；其余档为 null
        val regionStat: RegionStatEngine.Plan?,  // 分区统计档的每区映射参数（P2.15）
        val w: Int, val h: Int,
        val mode: ProtectMode,
        val rows: Int, val bands: Int,
    ) {
        fun release() {
            src.release(); mapped.release(); tiled?.release()
        }
    }

    /**
     * 迁移阶段：Bitmap → src（全分辨率 float）+ 缩略内容 LUT / 语义权重 +
     * 物化全强度 LUT 输出到 `mapped`。`onProgress(done, total)` 只在 mapped
     * 填充循环上报（total=bands）。
     */
    fun migrate(bitmap: Bitmap, mode: ProtectMode,
                onProgress: (Int, Int) -> Unit = { _, _ -> }): Migration {
        val lapAll = Lap()
        var w = bitmap.width
        var h = bitmap.height
        val src: Bitmap = if (w.toLong() * h > MAX_PIXELS) {
            val scale = kotlin.math.sqrt(MAX_PIXELS.toDouble() / (w.toLong() * h))
            val nw = (w * scale).toInt().coerceAtLeast(1)
            val nh = (h * scale).toInt().coerceAtLeast(1)
            val scaled = Bitmap.createScaledBitmap(bitmap, nw, nh, true)
            w = nw; h = nh
            scaled
        } else bitmap

        val rows = maxOf(1, BAND_PIXELS / w)
        val bands = (h + rows - 1) / rows
        // P2.15：REGION 的语义随基档而不同——
        //   - encoder / 统计类（plain）：全局映射 + 语义皮肤保护（applyBandMapped）；
        //   - RegionStat（分区预设）：每区独立统计映射（桌面 `apply_region` 不忽略
        //     mask），走 RegionStatEngine。预设本色是 REGION（UI 载入时默认选中），
        //     但**档位仍可切**：CHROMA/OFF 退化为"全局映射 (+肤色锁定)"（不做分区
        //     追踪）——与 UI 选择一致，不静默忽略。P2.15 移除了旧的"统计类不支持
        //     REGION"require。
        val useRegion = mode == ProtectMode.REGION && selfieNet != null
        // 全局统计/全局映射的基档：RegionStat 取 global（lab_stats/ot_linear）
        val statModel: ContentModel? = when (model) {
            is ContentModel.LabStats, is ContentModel.OtLinear -> model
            is ContentModel.RegionStat -> model.global
            else -> null
        }

        val src32 = Mat(h, w, CvType.CV_32FC3)
        var src8: Mat? = null
        try {
            // ---- 源段：全分辨率像素 → src32（float）+ src8（u8，仅 REGION 需要） ----
            // 段缓冲提出循环：60MP 时每段 ~10MB、百余段的分配churn在内存压力下
            // 会被 GC 放大（P2.12）。完整段用 nMax 尺寸复用缓冲，最后不完整段
            // 用精确尺寸缓冲——Mat.get/put 的 float/int 版没有 offset 重载，
            // 数组必须与段严格等长。
            val nMax = w * rows
            val nLast = w * (h - (bands - 1) * rows)
            val intsF = IntArray(nMax); val fbF = FloatArray(nMax * 3)
            val intsT = IntArray(nLast); val fbT = FloatArray(nLast * 3)
            val bbF = if (useRegion) ByteArray(nMax * 3) else null
            val bbT = if (useRegion) ByteArray(nLast * 3) else null
            // 统计类在源段循环里顺带累加（免额外全图扫描；float64 归约）。
            // LabStats：单遍 Σx/Σx²（std=sqrt(max(Σx²/n−mean²,0))，与桌面两遍式
            // 在 double 下差 ≪ u8 量化，金标锁）；ot 的协方差需先有全图均值 ⇒
            // outer 在物化循环（第二遍读 src32）里累加。
            val labSum = if (statModel is ContentModel.LabStats) DoubleArray(3) else null
            val labSq = if (statModel is ContentModel.LabStats) DoubleArray(3) else null
            val rgbSum = if (statModel is ContentModel.OtLinear) DoubleArray(3) else null
            val nPxTotal = w.toLong() * h
            val labMean = DoubleArray(3)
            val labStd = DoubleArray(3)
            val lapSrc = Lap()
            for (b in 0 until bands) {
                val y0 = b * rows
                val r = minOf(rows, h - y0)
                val full = (r == rows)
                val n = w * r
                val ints = if (full) intsF else intsT
                val fb = if (full) fbF else fbT
                src.getPixels(ints, 0, w, 0, y0, w, r)
                val bb: ByteArray? = if (bbF != null) (if (full) bbF else bbT!!) else null
                if (NativeKernels.useNative &&
                    NativeKernels.argbToRgbF32(ints, n, fb, bb)) {
                    // native 快路径（2026-10-01 性能轮）：ARGB → RGB f32 (+RGB u8)
                    // 一趟 C 循环，逐位同式（通道 /255f + RGB 序）；失败原样走下式
                } else {
                    for (p in 0 until n) {
                        val v = ints[p]
                        fb[3 * p] = (v shr 16 and 0xFF) / 255f
                        fb[3 * p + 1] = (v shr 8 and 0xFF) / 255f
                        fb[3 * p + 2] = (v and 0xFF) / 255f
                    }
                    if (bb != null) {
                        for (p in 0 until n) {
                            val v = ints[p]
                            bb[3 * p] = (v shr 16 and 0xFF).toByte()
                            bb[3 * p + 1] = (v shr 8 and 0xFF).toByte()
                            bb[3 * p + 2] = (v and 0xFF).toByte()
                        }
                    }
                }
                src32.put(y0, 0, fb)
                if (labSum != null) {
                    // reinhard：LAB 统计（当前图自身的 src 侧统计量）。
                    // native 优先（2026-10-02 性能轮）：rgbToLab + Σ/Σ² 分块并行
                    // 一趟；失败回退 Kotlin 参考循环。
                    val sq = labSq!!                     // 与 labSum 同条件非空（编译器看不到）
                    val buf6 = DoubleArray(6)
                    if (NativeKernels.useNative && NativeKernels.statSumLab(fb, buf6)) {
                        for (c in 0 until 3) {
                            labSum[c] += buf6[c]
                            sq[c] += buf6[3 + c]
                        }
                    } else {
                        val lab = LabConv.rgbToLab(fb)
                        for (p in 0 until n) {
                            for (c in 0 until 3) {
                                val x = lab[3 * p + c].toDouble()
                                labSum[c] += x
                                sq[c] += x * x
                            }
                        }
                    }
                }
                if (rgbSum != null) {
                    // ot：RGB 均值（协方差在物化循环第二遍累加）。native 优先。
                    val buf3 = DoubleArray(3)
                    if (NativeKernels.useNative && NativeKernels.statSumRgb(fb, buf3)) {
                        for (c in 0 until 3) rgbSum[c] += buf3[c]
                    } else {
                        for (p in 0 until n) {
                            rgbSum[0] += fb[3 * p].toDouble()
                            rgbSum[1] += fb[3 * p + 1].toDouble()
                            rgbSum[2] += fb[3 * p + 2].toDouble()
                        }
                    }
                }
                if (bb != null) {
                    val m8 = src8 ?: Mat(h, w, CvType.CV_8UC3).also { src8 = it }
                    m8.put(y0, 0, bb, 0, n * 3)
                }
            }
            if (labSum != null) {
                val sq = labSq!!
                for (c in 0 until 3) {
                    labMean[c] = labSum[c] / nPxTotal
                    labStd[c] = maxOf(
                        kotlin.math.sqrt(sq[c] / nPxTotal - labMean[c] * labMean[c]),
                        1e-6)
                }
            }
            val srcMs = lapSrc.ms()
            if (src !== bitmap) src.recycle()

            // ---- 引擎准备（按模型分发）：encoder=LUT/分割；统计类=内容统计 → 映射参数 ----
            var thumbLutMs = 0.0; var segMs = 0.0; var weightsMs = 0.0; var toFullMs = 0.0
            var statMs = 0.0; var regionStatMs = 0.0
            val lut: Lut3D?
            val tiled: RegionEngine.Tiled?
            var otT: Array<DoubleArray>? = null
            var otMuSrc: DoubleArray? = null
            var otCov: Array<DoubleArray>? = null
            var regionPlan: RegionStatEngine.Plan? = null
            when (model) {
                is ContentModel.Encoder -> {
                    if (useRegion) {
                        val region = RegionEngine(model.engine, selfieNet!!, PROTECT_UNUSED)
                        tiled = region.prepareTiled(src32, src8!!) { name, ms ->
                            when (name) {
                                "thumb+lut" -> thumbLutMs = ms
                                "seg" -> segMs = ms
                                "weights" -> weightsMs = ms
                                "toFull" -> toFullMs = ms
                            }
                        }
                        lut = tiled.lut
                    } else {
                        if (mode == ProtectMode.REGION)   // 静默降级会伪装成"保护没生效"
                            DebugLog.e("region 档但分割网不可用→回退 chroma 点态锁定")
                        val lapT = Lap()
                        lut = model.engine.lutFor(model.engine.thumbFromMat(src32))
                        thumbLutMs = lapT.ms()
                        tiled = null
                    }
                }
                is ContentModel.RegionStat -> {
                    // P2.15：分区统计档——分割 + 权重 + 全分辨率类图（hard 掩码来源）
                    tiled = if (useRegion) RegionEngine(model, selfieNet!!, PROTECT_UNUSED)
                        .prepareTiled(src32, src8!!, withLut = false,
                                      withHardMasks = true) { name, ms ->
                            when (name) {
                                "seg" -> segMs = ms
                                "weights" -> weightsMs = ms
                                "toFull" -> toFullMs = ms
                            }
                        } else {
                        // 只有「要分区却拿不到分割」才是降级；OFF/CHROMA 本就只走全局
                        if (mode == ProtectMode.REGION)
                            DebugLog.e("分区统计档但分割网不可用→回退 chroma 点态锁定")
                        null
                    }
                    lut = null
                }
                is ContentModel.LabStats, is ContentModel.OtLinear -> {
                    if (useRegion) {
                        // 统计类分区保护：分割+权重照跑（withLut=false，不建 encoder LUT），
                        // mapped 由下方物化循环给统计类仿射——融合层与基方法无关
                        val region = RegionEngine(model, selfieNet!!, PROTECT_UNUSED)
                        tiled = region.prepareTiled(src32, src8!!, withLut = false) { name, ms ->
                            when (name) {
                                "seg" -> segMs = ms
                                "weights" -> weightsMs = ms
                                "toFull" -> toFullMs = ms
                            }
                        }
                    } else {
                        if (mode == ProtectMode.REGION)   // 静默降级会伪装成"保护没生效"
                            DebugLog.e("region 档但分割网不可用→回退 chroma 点态锁定")
                        tiled = null
                    }
                    lut = null
                }
            }
            // ot 全局协方差（第二遍扫描：有了全图均值才能累加；桌面 _mean_cov 两遍式）
            if (statModel is ContentModel.OtLinear) {
                val lapS = Lap()
                val mu = DoubleArray(3) { rgbSum!![it] / nPxTotal }
                val outer = Array(3) { DoubleArray(3) }
                val scanF = FloatArray(nMax * 3)
                val scanT = FloatArray(nLast * 3)
                val outer6 = DoubleArray(6)
                for (b in 0 until bands) {
                    val y0 = b * rows
                    val r = minOf(rows, h - y0)
                    val full = (r == rows)
                    val n = w * r
                    val seg = if (full) scanF else scanT
                    src32.get(y0, 0, seg)
                    // native 优先（2026-10-02 性能轮）：分块并行协方差累加
                    if (NativeKernels.useNative &&
                        NativeKernels.statOuterRgb(seg, mu, outer6)) {
                        // 快路径
                    } else {
                        for (p in 0 until n) {
                            val d0 = seg[3 * p].toDouble() - mu[0]
                            val d1 = seg[3 * p + 1].toDouble() - mu[1]
                            val d2 = seg[3 * p + 2].toDouble() - mu[2]
                            outer6[0] += d0 * d0; outer6[1] += d0 * d1
                            outer6[2] += d0 * d2; outer6[3] += d1 * d1
                            outer6[4] += d1 * d2; outer6[5] += d2 * d2
                        }
                    }
                }
                outer[0][0] = outer6[0]; outer[0][1] = outer6[1]
                outer[0][2] = outer6[2]; outer[1][1] = outer6[3]
                outer[1][2] = outer6[4]; outer[2][2] = outer6[5]
                val inv = 1.0 / maxOf(nPxTotal - 1, 1)
                val cov = Array(3) { r0 -> DoubleArray(3) { c0 ->
                    (if (r0 <= c0) outer[r0][c0] else outer[c0][r0]) * inv +
                            (if (r0 == c0) 1e-6 else 0.0)   // RIDGE（桌面同值）
                } }
                otT = StatEngine.otLinearMap(cov, statModel.refCov)
                otMuSrc = mu
                otCov = cov
                statMs = lapS.ms()
            }
            src8?.release(); src8 = null

            // ---- P2.15：分区统计档的内容各区统计 → 渲染计划（每区映射参数） ----
            if (model is ContentModel.RegionStat) {
                val clsFull = tiled?.clsFull
                if (clsFull != null) {
                    val lapR = Lap()
                    val acc = StatEngine.RegionStatAccumulator(statModel is ContentModel.LabStats)
                    if (statModel is ContentModel.LabStats) {
                        RegionEngine.regionStats(clsFull, src32, acc, rows)
                        val (srcMean, srcStd) = acc.labStats(labMean, labStd)
                        regionPlan = RegionStatEngine.buildPlan(model, srcMean, srcStd, null)
                    } else {
                        // ot：第一遍各区均值 → 第二遍各区协方差（桌面 _mean_cov 两遍式）
                        RegionEngine.regionStats(clsFull, src32, acc, rows)
                        val means = acc.means(otMuSrc!!)
                        RegionEngine.regionStats(clsFull, src32, acc, rows, means)
                        val (srcMu, srcCov) = acc.otStats(otMuSrc!!, otCov!!)
                        regionPlan = RegionStatEngine.buildPlan(model, srcMu, null, srcCov)
                    }
                    regionStatMs = lapR.ms()
                }
                tiled?.releaseCls()   // 类图只在统计期用，别让它驻留整个会话
            }

            // ---- 物化全强度输出（与 strength 无关；render 复用，避免重算） ----
            val lapMap = Lap()
            val mapped = Mat(h, w, CvType.CV_32FC3)
            try {
                val segF = FloatArray(nMax * 3)
                val segT = FloatArray(nLast * 3)
                for (b in 0 until bands) {
                    val y0 = b * rows
                    val r = minOf(rows, h - y0)
                    val seg = if (r == rows) segF else segT
                    src32.get(y0, 0, seg)
                    // 全局映射（与 strength 无关）：statModel==null ⇒ encoder 档走 LUT
                    val outSeg: FloatArray = when (statModel) {
                        is ContentModel.LabStats -> StatEngine.mapLabStats(
                            seg, statModel.refMean, statModel.refStd, labMean, labStd)
                        is ContentModel.OtLinear -> StatEngine.mapOtLinear(
                            seg, otT!!, otMuSrc!!, statModel.refMean)
                        else -> lut!!.apply(seg)          // LUT apply 返回新数组（保持原式）
                    }
                    mapped.put(y0, 0, outSeg)
                    onProgress(b + 1, bands)
                }
            } catch (e: Throwable) {
                mapped.release()
                throw e
            }
            val mapMs = lapMap.ms()
            val modelName = when (model) {
                is ContentModel.Encoder -> "encoder"
                is ContentModel.LabStats -> "lab_stats"
                is ContentModel.OtLinear -> "ot_linear"
                is ContentModel.RegionStat -> "region:${model.baseKind}"
            }
            DebugLog.i("migrate| model=$modelName total=${fmt(lapAll.ms())}ms " +
                    "src=${fmt(srcMs)} thumb+lut=${fmt(thumbLutMs)} seg=${fmt(segMs)} " +
                    "weights=${fmt(weightsMs)} toFull=${fmt(toFullMs)} " +
                    "stat=${fmt(statMs)} regionStat=${fmt(regionStatMs)} " +
                    "mapped=${fmt(mapMs)} mode=$mode " +
                    "${w}x${h} bands=$bands")
            return Migration(src32, mapped, tiled, regionPlan, w, h, mode, rows, bands)
        } catch (e: Throwable) {
            src32.release()
            src8?.release()
            throw e
        }
    }

    /** 渲染产物：位图 + film 层是否真正应用（false = 处理失败被整层跳过，日志有因；
     *  预览路径据此提示，效果不再"悄悄消失"）。 */
    class RenderResult internal constructor(val bitmap: Bitmap, val filmApplied: Boolean)

    fun render(m: Migration, strength: Float,
               develop: Map<String, Float>? = null,
               protect: Float = 1f,
               film: Map<String, Float>? = null,
               onProgress: (Int, Int) -> Unit = { _, _ -> }): Bitmap =
        renderResult(m, strength, develop, protect, film, onProgress).bitmap

    /**
     * 出图阶段：从迁移会话逐段重放「withStrength → 融合 → LAB 锚定」，
     * 叠加 develop，再出 u8 Bitmap。不触碰 LUT/分割/权重计算。
     * `onProgress(done, total)` 按段上报（total=bands）。
     *
     * 与 [render] 同实现，另带 film 应用状态（2026-10-03：预览路径失败不再
     * 静默——导出路径的对应物是 [JpegExport.filmApplied]）。
     *
     * @param protect 保护强度 0~1（默认 1＝皮肤恒原色）。**只作用于本阶段**——
     *        `migrate` 不使用它（`RegionEngine.prepareTiled` 不读 protect），
     *        因此拖保护强度不会让迁移缓存失效，与 strength 同级开销。
     */
    fun renderResult(m: Migration, strength: Float,
                     develop: Map<String, Float>? = null,
                     protect: Float = 1f,
                     film: Map<String, Float>? = null,
                     onProgress: (Int, Int) -> Unit = { _, _ -> }): RenderResult {
        val lapAll = Lap()
        val dev = if (develop != null && !DevelopTransform(develop).isIdentity())
            DevelopTransform(develop) else null
        val filmOn = FilmLayer.active(film)
        val out = Bitmap.createBitmap(m.w, m.h, Bitmap.Config.ARGB_8888)
        val outInts = IntArray(m.w * m.rows)
        var applyMs = 0.0; var devMs = 0.0; var u8Ms = 0.0; var filmMs = 0.0
        var filmOk = true        // film 开启时由 apply 回填；跳过 = false 传给调用方
        val buf = BandBuf(m.w * m.rows)
        // film 开启：逐段出图改写全图 f32 Mat（film 的模糊核窗跨段，不能逐段
        // u8），render 完成后 FilmLayer 分段处理再 u8 回 Bitmap。预览图小
        // （≤2MP），全图 f32 驻留 ~9-24MB。film 关闭时此 Mat 不分配（零变化）。
        val outF32 = if (filmOn) Mat(m.h, m.w, CvType.CV_32FC3) else null
        try {
            for (b in 0 until m.bands) {
                val y0 = b * m.rows
                val r = minOf(m.rows, m.h - y0)
                val n = m.w * r
                // 完整段复用缓冲；最后不完整段用精确尺寸临时数组——
                // native fuseBand 校验所有数组等长，超配会假失败（P2.12）
                val use = if (r == m.rows) buf else BandBuf(n)
                m.src.get(y0, 0, use.rgb)
                m.mapped.get(y0, 0, use.mapped)

                var t0 = System.nanoTime()
                applyBandCore(m, y0, r, n, strength, protect, use)
                var t1 = System.nanoTime(); applyMs += (t1 - t0) / 1e6

                var resultF: FloatArray = use.outF
                if (dev != null) resultF = dev.apply(use.outF)   // 微调叠在自动追色之后
                var t2 = System.nanoTime(); devMs += (t2 - t1) / 1e6

                if (outF32 != null) {
                    outF32.put(y0, 0, resultF)
                    var t3 = System.nanoTime(); u8Ms += (t3 - t2) / 1e6
                    onProgress(b + 1, m.bands)
                    continue
                }
                // float → u8（与 f32ToBitmap 同式）→ 写回输出位图
                for (p in 0 until n) {
                    fun u8(v: Float): Int = ((v.coerceIn(0f, 1f) * 255f + 0.5f).toInt())
                    outInts[p] = (0xFF shl 24) or (u8(resultF[3 * p]) shl 16) or
                            (u8(resultF[3 * p + 1]) shl 8) or u8(resultF[3 * p + 2])
                }
                out.setPixels(outInts, 0, m.w, 0, y0, m.w, r)
                var t3 = System.nanoTime(); u8Ms += (t3 - t2) / 1e6
                onProgress(b + 1, m.bands)
            }
            if (outF32 != null) {
                val tf = System.nanoTime()
                filmOk = FilmLayer.apply(outF32, film!!)
                filmMs = (System.nanoTime() - tf) / 1e6
                // 预览图小（≤2MP）：全图读回一次 u8 化
                val f = FloatArray(m.w * m.h * 3)
                outF32.get(0, 0, f)
                fun u8(v: Float): Int = ((v.coerceIn(0f, 1f) * 255f + 0.5f).toInt())
                var idx = 0
                for (b in 0 until m.bands) {
                    val y0 = b * m.rows
                    val r = minOf(m.rows, m.h - y0)
                    val n = m.w * r
                    for (p in 0 until n) {
                        outInts[p] = (0xFF shl 24) or (u8(f[idx]) shl 16) or
                                (u8(f[idx + 1]) shl 8) or u8(f[idx + 2])
                        idx += 3
                    }
                    out.setPixels(outInts, 0, m.w, 0, y0, m.w, r)
                }
            }
            DebugLog.i("render | total=${fmt(lapAll.ms())}ms apply=${fmt(applyMs)} " +
                    "dev=${fmt(devMs)} film=${fmt(filmMs)} u8=${fmt(u8Ms)} " +
                    "bands=${m.bands} s=${fmt(strength.toDouble())} " +
                    "protect=${fmt(protect.toDouble())} devActive=${dev != null} " +
                    "filmActive=$filmOn")
            return RenderResult(out, !filmOn || filmOk)
        } finally {
            outF32?.release()
        }
    }

    /**
     * 流式 JPEG 导出（P2.12）：迁移 → 逐段出图直写 CV_8UC3 BGR Mat →
     * **先释放迁移会话再 imencode**。与旧链路（render 出全尺寸 Bitmap →
     * bitmapToMat → cvtColor → imencode）产物**逐字节一致**（u8 式与 BGR 字节序
     * 由 NativeKernelsTest / TiledTest 的流式关卡锁死），但砍掉 240MB 级
     * Bitmap 分配、setPixels/bitmapToMat 两次全图拷贝与一次全图 cvtColor——
     * 60MP 的 u8 段 17.7s 的主因就是这些拷贝在 2GB+ 原生占用触发的内存
     * 回收下被放大。迁移会话（60MP 时 src32+mapped ≈1.4GB）在编码前释放。
     *
     * **所有权转移**：调用即接管 `bitmap`——迁移完成后就地回收（成功与失败
     * 都不可再用）。调用方不要再 recycle 它。
     *
     * `onProgress(done, total)` 语义与 [process] 相同：迁移 [0,bands)、
     * 出图 [bands,2*bands)。
     */
    class JpegExport internal constructor(val bytes: ByteArray, val w: Int, val h: Int,
                                          /** false = 胶片层处理失败被跳过（日志有因；
                                           *  导出照常成功，调用方应提示用户不静默）。 */
                                          val filmApplied: Boolean = true)

    fun processToJpeg(bitmap: Bitmap, mode: ProtectMode, strength: Float,
                      develop: Map<String, Float>? = null,
                      protect: Float = 1f, quality: Int = 100,
                      film: Map<String, Float>? = null,
                      onProgress: (Int, Int) -> Unit = { _, _ -> }): JpegExport {
        var bands = 1
        val m: Migration = try {
            migrate(bitmap, mode) { d, t ->
                bands = t
                onProgress(d, t * 2)
            }
        } catch (e: Throwable) {
            bitmap.recycle()
            throw e
        }
        bitmap.recycle()                 // 迁移不再读它；回收省一份全图驻留
        val outBgr = Mat(m.h, m.w, CvType.CV_8UC3)
        var mReleased = false
        try {
            val dev = if (develop != null && !DevelopTransform(develop).isIdentity())
                DevelopTransform(develop) else null
            val filmOn = FilmLayer.active(film)
            // film 开启：render 输出改写全图 f32 Mat（模糊核窗跨段），循环后
            // FilmLayer 分段处理、再 u8 直写 outBgr。film 的内存代价由
            // 「迁移会话提前释放」抵消——60MP 时 src+mapped ~1.4GB 的释放远大于
            // 全图 f32（预压缩 66MB / 60MP 720MB）的驻留。
            val outF32 = if (filmOn) Mat(m.h, m.w, CvType.CV_32FC3) else null
            val buf = BandBuf(m.w * m.rows)
            val bgr = ByteArray(m.w * m.rows * 3)
            var applyMs = 0.0; var devMs = 0.0; var u8Ms = 0.0; var filmMs = 0.0
            var filmOk = true        // film 开启时由 apply 回填；跳过 = false 传给调用方
            val lapAll = Lap()
            for (b in 0 until m.bands) {
                val y0 = b * m.rows
                val r = minOf(m.rows, m.h - y0)
                val n = m.w * r
                val use = if (r == m.rows) buf else BandBuf(n)
                val useBgr = if (r == m.rows) bgr else ByteArray(n * 3)
                m.src.get(y0, 0, use.rgb)
                m.mapped.get(y0, 0, use.mapped)

                var t0 = System.nanoTime()
                applyBandCore(m, y0, r, n, strength, protect, use)
                var t1 = System.nanoTime(); applyMs += (t1 - t0) / 1e6

                var resultF: FloatArray = use.outF
                if (dev != null) resultF = dev.apply(use.outF)
                var t2 = System.nanoTime(); devMs += (t2 - t1) / 1e6

                if (outF32 != null) {
                    outF32.put(y0, 0, resultF)
                    onProgress(m.bands + b + 1, m.bands * 2)
                    continue
                }
                // float → u8 BGR：native 优先，失败回退 Kotlin（同式）
                if (!NativeKernels.useNative ||
                    !NativeKernels.rgbToBgrU8(resultF, useBgr)) {
                    kotlinRgbToBgr(resultF, useBgr, n)
                }
                outBgr.put(y0, 0, useBgr, 0, n * 3)   // byte[] 有 offset 重载
                var t3 = System.nanoTime(); u8Ms += (t3 - t2) / 1e6
                onProgress(m.bands + b + 1, m.bands * 2)
            }
            if (outF32 != null) {
                // 迁移会话提前释放（film 轮不再读它），再跑三效果分段 + u8
                m.release(); mReleased = true
                val tf = System.nanoTime()
                filmOk = FilmLayer.apply(outF32, film!!)
                filmMs = (System.nanoTime() - tf) / 1e6
                fun toBgr(f: FloatArray, dst: ByteArray, cnt: Int) {
                    if (!NativeKernels.useNative ||
                        !NativeKernels.rgbToBgrU8(f, dst)) {
                        kotlinRgbToBgr(f, dst, cnt)
                    }
                }
                val seg = FloatArray(m.w * m.rows * 3)
                for (b in 0 until m.bands) {
                    val y0 = b * m.rows
                    val r = minOf(m.rows, m.h - y0)
                    val n = m.w * r
                    val useBgr = if (r == m.rows) bgr else ByteArray(n * 3)
                    if (r == m.rows) {
                        outF32.rowRange(y0, y0 + r).get(0, 0, seg)
                        toBgr(seg, useBgr, n)
                    } else {
                        val tmp = FloatArray(n * 3)
                        outF32.rowRange(y0, y0 + r).get(0, 0, tmp)
                        toBgr(tmp, useBgr, n)
                    }
                    outBgr.put(y0, 0, useBgr, 0, n * 3)
                }
                outF32.release()
                // 标签必须如实：film 跳过时这行若是"(film on)"会把失败伪装成正常
                DebugLog.i("filmBgr| ${fmt(filmMs)}ms (film ${if (filmOk) "on" else "跳过"})")
            }
            DebugLog.i("renderBgr| total=${fmt(lapAll.ms())}ms apply=${fmt(applyMs)} " +
                    "dev=${fmt(devMs)} film=${fmt(filmMs)} u8=${fmt(u8Ms)} " +
                    "bands=${m.bands} s=${fmt(strength.toDouble())} " +
                    "protect=${fmt(protect.toDouble())} devActive=${dev != null}")

            // 编码前先释放迁移会话：60MP 时 src32+mapped ≈1.4GB，编码期只留
            // 180MB BGR + JPEG 工作集，不再双倍压内存
            m.release(); mReleased = true
            val t4 = System.nanoTime()
            val bytes = encodeJpeg444FromBgr(outBgr, quality)
            DebugLog.i("encodeJpeg| ${fmt((System.nanoTime() - t4) / 1e6)}ms q=$quality " +
                    "${m.w}x${m.h} -> ${bytes.size}B")
            return JpegExport(bytes, m.w, m.h, !filmOn || filmOk)
        } finally {
            if (!mReleased) m.release()
            outBgr.release()
        }
    }

    /** Kotlin 回退：与 render 的 u8 同式、BGR 字节序（native 不可用时）。 */
    private fun kotlinRgbToBgr(rgb: FloatArray, out: ByteArray, n: Int) {
        fun u8(v: Float): Byte = ((v.coerceIn(0f, 1f) * 255f + 0.5f).toInt()).toByte()
        for (p in 0 until n) {
            out[3 * p] = u8(rgb[3 * p + 2])
            out[3 * p + 1] = u8(rgb[3 * p + 1])
            out[3 * p + 2] = u8(rgb[3 * p])
        }
    }

    /**
     * 段级出图核心（render / processToJpeg 共用）：src/mapped 段已由调用方
     * 读入 `use`，按档位重放 withStrength/融合/chroma 锚定 → 写 `use.outF`
     * （不含 develop——计时口径与调用方拆分保持不变）。
     */
    private class BandBuf(n: Int) {
        val rgb = FloatArray(n * 3)
        val mapped = FloatArray(n * 3)
        val outF = FloatArray(n * 3)
    }

    private fun applyBandCore(m: Migration, y0: Int, r: Int, n: Int,
                              strength: Float, protect: Float, use: BandBuf) {
        val tiled = m.tiled
        when {
            // P2.15 分区统计档：每区独立仿射 → RGB 加权融合 → L 替换 → 外层 strength
            tiled != null && m.regionStat != null -> RegionStatEngine.applyBand(
                tiled, m.regionStat, use.rgb, use.mapped, y0, r, protect, strength, use.outF)
            tiled != null -> RegionEngine.applyBandMapped(
                tiled, use.rgb, use.mapped, y0, r, protect, strength, use.outF)
            m.mode == ProtectMode.CHROMA || m.mode == ProtectMode.REGION -> {
                // chroma（及 region 的分割不可用回退）：全局追色 + a/b 锚回
                // ——引擎 applyChroma 的逐像素式（protect 由滑块给，默认 1）。
                // 2026-10-02 性能轮：SkinWeight+锚定+LAB 往返合成一趟 native
                // 内核（逐位对拍锁），失败回退下面的纯 Kotlin。
                val globalS = EncoderEngine.withStrength(use.rgb, use.mapped, strength)
                if (NativeKernels.chromaAnchorApply(use.rgb, globalS, protect,
                                                    use.outF)) {
                    return
                }
                val labIn = LabConv.rgbToLab(use.rgb)
                val labOut = LabConv.rgbToLab(globalS)
                val weight = SkinWeight.of(labIn)
                val anchored = LabConv.labToRgb(
                    SkinWeight.anchorChroma(labIn, labOut, weight, protect))
                System.arraycopy(anchored, 0, use.outF, 0, n * 3)
            }
            else -> {
                // OFF 档：withStrength 直写 outF（2026-10-01 性能轮）——原式每段
                // 建一个临时数组再 arraycopy；native 优先，逐位同式（inv 预计算）。
                val s = strength
                if (NativeKernels.useNative &&
                    NativeKernels.strengthMix(use.rgb, use.mapped, s, use.outF)) {
                    // native 快路径
                } else if (s <= 0f) {
                    System.arraycopy(use.rgb, 0, use.outF, 0, n * 3)
                } else if (s >= 1f) {
                    System.arraycopy(use.mapped, 0, use.outF, 0, n * 3)
                } else {
                    val inv = 1f - s
                    for (i in 0 until n * 3) {
                        use.outF[i] = use.rgb[i] * inv + use.mapped[i] * s
                    }
                }
            }
        }
    }

    /**
     * 保护区域蒙版（0~1，长度 = `m.w * m.h`）——与**实际生效的保护同源**：
     *  - REGION（`tiled != null`）：直接读 `tiled.wSkin`（Σ 归一 skin 权重，
     *    与 `applyBandMapped` 用的是同一张权重图，~10ms，不重跑分割）；
     *  - CHROMA / REGION 回退（`tiled == null`）：肤色隶属度 × protect，
     *    与 `render` 的 chroma 分支同式；
     *  - OFF：无保护 ⇒ 全零（诚实显示"没有保护在生效"）。
     *
     * 对应桌面 `transform.protection()`（methods/region_transfer.py:190 /
     * skin_protect.py:184）——界面只认这一个入口，不必知道是哪套机制。
     */
    fun protectionMask(m: Migration, protect: Float): FloatArray {
        val n = m.w * m.h
        if (m.mode == ProtectMode.OFF) return FloatArray(n)
        val tiled = m.tiled
        val mask = if (tiled != null) {
            FloatArray(n).also { tiled.wSkin.get(0, 0, it) }
        } else {
            val rgb = FloatArray(n * 3)
            m.src.get(0, 0, rgb)
            SkinWeight.of(LabConv.rgbToLab(rgb))
        }
        if (protect != 1f) for (i in 0 until n) mask[i] *= protect
        return mask
    }

    /**
     * REGION 档是否真的检测到皮肤：`tiled.wSkin` 的**峰值**（Σ 归一权重里
     * 皮肤像素 ≈0.7~1、无人图整张 ≈0）。UI 用它给"未检测到人物 → 建议改用
     * 肤色锁定"的提示（P2.11）；`tiled == null`（回退档）返回 0，但提示只在
     * REGION 生效路径上出现，不受影响。整个 Mat 读一遍 ~3MB（预览口径），
     * 只在迁移后调用一次，不在拖动路径上。
     */
    fun regionSkinPeak(m: Migration): Float {
        val t = m.tiled ?: return 0f
        val n = t.wSkin.cols() * t.wSkin.rows()
        if (n <= 0) return 0f
        val a = FloatArray(n)
        t.wSkin.get(0, 0, a)
        var mx = 0f
        for (v in a) if (v > mx) mx = v
        return mx
    }

    /**
     * 一次性全流程（保存/批量/测试用）：migrate + render，结束后释放迁移会话。
     * 缓存路径见 MainActivity 的 Effect A/B。
     */
    fun process(bitmap: Bitmap, mode: ProtectMode, strength: Float,
                develop: Map<String, Float>? = null,
                protect: Float = 1f,
                onProgress: (Int, Int) -> Unit = { _, _ -> }): Bitmap {
        // 进度语义与旧版一致：迁移映射到 [0, bands)、出图映射到 [bands, 2*bands)
        var bands = 1
        val m = migrate(bitmap, mode) { d, t ->
            bands = t
            onProgress(d, t * 2)
        }
        try {
            return render(m, strength, develop, protect) { d, _ ->
                onProgress(bands + d, bands * 2)
            }
        } finally {
            m.release()
        }
    }
}