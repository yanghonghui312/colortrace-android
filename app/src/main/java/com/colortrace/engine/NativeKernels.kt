package com.colortrace.engine

/**
 * native 逐像素内核第二批（`src/main/cpp/kernels.cpp`）：
 * 33³ LUT 三线性查表（[Lut3D]）与四区融合整段（[RegionEngine] 的 applyBandMapped）。
 *
 * 与 [NativeLab] 同一个 `libcolortrace.so`——加载入口只在 [NativeLab] 一处，
 * 这里直接复用它的结果，不重复 loadLibrary。
 *
 * 两个 external 都返回 **Boolean**：false ＝ 参数不合法/分配失败/异常，
 * 调用方**必须**回退纯 Kotlin 路径，绝不静默产出错值。
 */
object NativeKernels {
    /** 与 [NativeLab.available] 同源（.so 加载失败即整体回退）。 */
    val available: Boolean get() = NativeLab.available

    /** 对拍/回退开关（`NativeKernelsTest` 用它 A/B 两条路径）。 */
    @Volatile
    var useNative: Boolean = NativeLab.available

    /** 33³ 三线性查表：table 需 size³×3；out 与 rgb 等长且不重叠。 */
    external fun lutApply(table: DoubleArray, size: Int, rgb: FloatArray,
                          out: FloatArray): Boolean

    /** 四区融合整段：out 与 rgb 等长；四个权重数组长度 = rgb.size/3。 */
    external fun fuseBand(rgb: FloatArray, mapped: FloatArray,
                          wSkin: FloatArray, wHair: FloatArray,
                          wCloth: FloatArray, wBg: FloatArray,
                          protect: Float, strength: Float, out: FloatArray): Boolean

    /** float RGB [0,1] → u8 BGR 字节流（流式 JPEG 直编，P2.12）；out.length == rgb.length。 */
    external fun rgbToBgrU8(rgb: FloatArray, out: ByteArray): Boolean

    /**
     * develop（手动微调）整段：RGB → LAB → mapLab → RGB 一趟走完（P2.16）。
     * `params` 长度必须为 17（顺序见 [DevelopTransform.PARAM_VECTOR_KEYS]）；
     * `out` 与 `rgb` 等长。与 [DevelopTransform.applyKotlin] 逐位一致。
     */
    external fun developApply(rgb: FloatArray, params: FloatArray, out: FloatArray): Boolean

    /**
     * Bitmap 段 ARGB int → RGB float32（+ 可选 RGB u8，null = 不出 u8）。
     * migrate 源段的 Kotlin 循环 native 化（2026-10-01 性能轮）。逐位口径：
     * `(通道 and 0xFF) / 255f`（float 除法）+ RGB 序 u8；alpha 不参与。
     * [argb] 可长于 [n]（只处理前 n 个）；out 长度 ≥ n×3。
     */
    external fun argbToRgbF32(argb: IntArray, n: Int, rgb: FloatArray,
                              rgbU8: ByteArray?): Boolean

    /**
     * [EncoderEngine.withStrength] 的 native 版（OFF 档出图段，0<s<1 的混合）。
     * 与 Kotlin 逐位一致（inv 预计算——(1f-s) 是同一个值）；s≤0 / s≥1 走
     * arraycopy 快路径，不进这里。
     */
    external fun strengthMix(rgb: FloatArray, mapped: FloatArray,
                             strength: Float, out: FloatArray): Boolean

    /**
     * 33³ 探针循环的 native 版（2026-10-02 性能轮）：对每探针做
     * `FlowCore.transfer(x, wContent, wStyle, strength, steps)`（原地写回 probes）。
     * `wc`/`ws` = [FlowWeights.pack]（W1|b1|W2|b2 打平，长度 8·hidden+3）。
     * Kotlin 线程并行此循环实测负优化（P2.22），正解即本内核；double 全程逐位。
     */
    external fun flowProbes(probes: DoubleArray, wc: DoubleArray, ws: DoubleArray,
                            hidden: Int, strength: Double, steps: Int): Boolean

    /**
     * 分区统计档段级内核（2026-10-02 性能轮）：[RegionStatEngine.applyBandKotlin]
     * 的整段等价（每区映射 → 嵌套强度 → 皮肤分支 → 四区融合 → L 替换 → 外层
     * strength）。`prm` = [RegionStatEngine.Plan.packed]（buildPlan 预构建）。
     */
    external fun regionStatBand(rgb: FloatArray, mapped: FloatArray,
                                wSkin: FloatArray, wHair: FloatArray,
                                wCloth: FloatArray, wBg: FloatArray,
                                prm: FloatArray, protect: Float, strength: Float,
                                out: FloatArray): Boolean

    /**
     * chroma 锚定整段（2026-10-02 性能轮）：rgbToLab×2 + SkinWeight.of +
     * anchorChroma + labToRgb 合成一趟（[TiledPipeline] 的 chroma 分支同式）。
     * **无 protect≤0 早退**——调用方有该语义的（applyChroma 等）自行在
     * protect>0 时才进来；out 与 rgb/globalS 必须是不同数组。
     */
    external fun chromaAnchor(rgb: FloatArray, globalS: FloatArray,
                              protect: Float, out: FloatArray): Boolean

    /**
     * chroma 锚定统一入口：native 优先，失败关 useNative 并返回 false，调用方
     * 走纯 Kotlin（与 [RegionEngine.applyBandMapped] 的回退纪律一致）。
     */
    fun chromaAnchorApply(rgb: FloatArray, globalS: FloatArray, protect: Float,
                          out: FloatArray): Boolean {
        if (!useNative) return false
        if (chromaAnchor(rgb, globalS, protect, out)) return true
        useNative = false
        com.colortrace.DebugLog.e("native chromaAnchor 失败→回退纯 Kotlin（后续不再尝试）")
        return false
    }

    // ---- 胶片质感层（桌面 methods/film.py 移植，2026-10-02；film_kernels.cpp）----
    // 柔光/光晕与桌面同式（u8 ≤1/255 金标协议）；颗粒不与桌面逐位（用户拍板：
    // 两端各自确定性、算法同构）。模糊与上采样在 Kotlin 层走 OpenCV Imgproc，
    // 这里只收逐像素算子。这些内核失败直接抛异常向上传（FilmLayer 捕获后
    // 整层跳过并记日志——胶片是风格化叠加，不阻塞出图）。

    /** 柔光段：out = rgb + (screen(rgb,blurred)−rgb)·(soft·smoothstep(t0,t0+band,luma))。 */
    external fun filmSoftBand(rgb: FloatArray, blurred: FloatArray,
                              soft: Float, t0: Float, band: Float,
                              out: FloatArray)

    /** 光晕·高光提取：mask = smoothstep(0.7,1.0,luma)，masked = rgb·mask。 */
    external fun filmBloomMask(rgb: FloatArray, masked: FloatArray, mask: FloatArray)

    /** 光晕·混合：tint>0 时 glow 乘 (1+(WARM−1)·tint)，out = rgb+(screen−rgb)·bloom。 */
    external fun filmBloomApply(rgb: FloatArray, glow: FloatArray,
                                bloom: Float, tint: Float, out: FloatArray)

    /** 颗粒·标准正态序列（端内确定性：splitmix64 + Box-Muller，seed 任意）。 */
    external fun filmGrainBase(seed: Long, n: Int, out: FloatArray)

    /** 颗粒·全图 std 归一：noise *= σ·amount/std(noise)（double 顺序累加）。 */
    external fun filmGrainNormalize(noise: FloatArray, sigmaAmount: Float)

    /** 颗粒·段应用：LAB 往返只加 L（clip [0,100]），out = labToRgb。 */
    external fun filmGrainBand(rgb: FloatArray, noise: FloatArray, out: FloatArray)

    // ---- 统计类方法的内容统计归约（2026-10-02 性能轮；ot 档真机 src+stat
    // ~2.5s/张的 Kotlin 单线程累加 native 化）。分块并行：固定块边界、块内
    // 顺序、块间按序合并 ⇒ 端内确定；与旧顺序版差 ~1e-12，远低于统计类
    // 金标 u8 ≤2/255 容差。失败返回 false 由调用方走 Kotlin 参考。

    /** ot 第一遍：Σx（out 长 3）。 */
    external fun statSumRgb(rgb: FloatArray, out: DoubleArray): Boolean

    /** ot 第二遍：d = x − mu 的上三角 Σ d⊗d（out 长 6 = [00,01,02,11,12,22]）。 */
    external fun statOuterRgb(rgb: FloatArray, mu: DoubleArray, out: DoubleArray): Boolean

    /** reinhard：rgbToLab + Σ 与 Σ²（out 长 6 = [ΣL,Σa,Σb,ΣL²,Σa²,Σb²]）。 */
    external fun statSumLab(rgb: FloatArray, out: DoubleArray): Boolean

    /**
     * 颗粒·细噪声条带**并行**生成：每条带独立子流（seed+si）⇒ 输出与 Kotlin
     * 串行逐条带**逐位一致**（端内确定性不变）；`fine·strip` 直接累加进
     * `noise`（w×h 全图数组，免条带 JNI 往返）。
     */
    external fun filmGrainBands(seed: Long, bands: Int, rowsPerBand: Int,
                                w: Int, h: Int, fine: Float, noise: FloatArray)

    // ---- 上面的 **Mat 数据指针版**（2026-10-03 生产路径；数组版保留作金标基准）----
    // 动机：5011×3341 全尺寸导出时段缓冲/颗粒噪声的 Java 堆数组必 OOM（堆上限
    // 256MB）→「胶片质感层失败→整层跳过」静默出无效果图。指针版把全部中间数据
    // 留在 native（`Mat.dataAddr()` 直传），Java 堆零大分配。核心与数组版共用
    // （逐位同算子同序）；addr 一律是**数据**指针，行窗按 (y0/a, rows) 偏移。
    // 金标继续锁数组版，两端等价由 `FilmTest.filmMatWrappersMatchArray` 锁死。

    /** 柔光段（就地写回）：读 mat[y0, y0+rows)、blurred[a, a+rows)。 */
    external fun filmSoftBandMat(matAddr: Long, y0: Int, blurredAddr: Long,
                                 a: Int, rows: Int, w: Int,
                                 soft: Float, t0: Float, band: Float)

    /** 光晕·高光提取（就地无关）：读 mat[yA, yA+rowsSub) → masked/mask。 */
    external fun filmBloomMaskMat(matAddr: Long, yA: Int, maskedAddr: Long,
                                  maskAddr: Long, rowsSub: Int, w: Int)

    /** 光晕·混合（就地写回）：读 mat[y0, y0+rows)、glow[a, a+rows)。 */
    external fun filmBloomApplyMat(matAddr: Long, y0: Int, glowAddr: Long,
                                   a: Int, rows: Int, w: Int,
                                   bloom: Float, tint: Float)

    /** 颗粒·标准正态序列直写 base 网格（hk×wk）。 */
    external fun filmGrainBaseMat(seed: Long, baseAddr: Long, n: Int)

    /** 颗粒·噪声逐元素 *= inv（与顺序无关 ⇒ 与串行逐位一致）。 */
    external fun filmGrainScaleMat(noiseAddr: Long, n: Int, inv: Float)

    /** 颗粒·细噪声条带并行（Mat 版，累加进 noiseMat 内存）。 */
    external fun filmGrainBandsMat(seed: Long, bands: Int, rowsPerBand: Int,
                                   w: Int, h: Int, fine: Float, noiseAddr: Long)

    /** 颗粒·全图 std 归一（Mat 版）。 */
    external fun filmGrainNormalizeMat(noiseAddr: Long, n: Int, sigmaAmount: Float)

    /** 颗粒·段应用（就地写回）：mat[y0, y0+rows) LAB 往返 + noise 同行窗。 */
    external fun filmGrainBandMat(matAddr: Long, y0: Int, noiseAddr: Long,
                                  rows: Int, w: Int)
}