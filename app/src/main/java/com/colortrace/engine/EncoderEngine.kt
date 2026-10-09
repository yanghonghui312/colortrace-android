package com.colortrace.engine

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import org.json.JSONObject
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.Size
import org.opencv.dnn.Dnn
import org.opencv.dnn.Net
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.util.Base64
import kotlin.math.round

/**
 * 零样本编码器引擎的 Kotlin 移植（桌面 `methods/encoder_transform.py`，P1）。
 *
 * 链路：预设 JSON（桌面产，原样）→ style_thumb PNG → Θs；
 * 内容图 → 128² 缩略图（INTER_AREA）→ Θc → 33³ LUT（单步 Euler flow）→ 查表。
 * 精度约定：Θ 推理 fp32（cv2.dnn，P0 已证安卓↔桌面 8.9e-08）；flat→权重 f32→f64
 * 精确提升；flow/LUT 全 Double；图像与强度混合 Float（与 numpy float32 语义同）。
 *
 * P2 追加（设备端 fit，不影响上述对拍路径）：
 * - [fromSample]：样片图直接 128² 缩略前向得 Θs——与桌面 GUI `analyze_sample`
 *   同语义（样片统计用 float 缩略图本身，量化 PNG 只在预设序列化时出现），
 *   无需桌面先出预设 JSON；
 * - `apply/applyChroma` 的 strength 重载：桌面「fit(strength=s) 后 apply」等价于
 *   「全强度映射 → RGB 线性混合 s」（`_with_strength` 语义），GUI 预览快路径
 *   同式；chroma/region 的 LAB 锚定作用在**混合后的全局结果**上，与桌面把
 *   strength 烘进 transform 再锚定同序。
 */
class EncoderPreset(jsonText: String) {
    val strength: Float
    val hidden: Int
    val imageSize: Int
    val lutSize: Int
    val onnxName: String
    val styleThumb: FloatArray   // 128² 交错 RGB（值 = k/255f，与桌面量化语义一致）
    val thumbSide: Int

    init {
        val root = JSONObject(jsonText)
        strength = root.getDouble("strength").toFloat()
        val p = root.getJSONObject("params")
        require(p.getString("arch") == "colorfm-l-cnn128") {
            "未知的编码器结构: ${p.getString("arch")}"
        }
        hidden = p.getInt("hidden")
        imageSize = p.getInt("image_size")
        lutSize = p.getInt("lut_size")
        onnxName = p.getString("onnx")
        val png = Base64.getDecoder().decode(p.getString("style_thumb"))
        val bmp = BitmapFactory.decodeByteArray(png, 0, png.size)
        require(bmp != null && bmp.width == bmp.height) { "缩略图解码失败或非正方形" }
        val side = bmp.width
        val argb = bmp.copy(Bitmap.Config.ARGB_8888, false)
        val px = IntArray(side * side)
        argb.getPixels(px, 0, side, 0, 0, side, side)
        styleThumb = FloatArray(px.size * 3)
        for (i in px.indices) {
            val v = px[i]
            styleThumb[3 * i] = (v shr 16 and 0xFF) / 255f
            styleThumb[3 * i + 1] = (v shr 8 and 0xFF) / 255f
            styleThumb[3 * i + 2] = (v and 0xFF) / 255f
        }
        thumbSide = side
        argb.recycle()
    }
}

class EncoderEngine private constructor(onnxBytes: ByteArray,
                                        styleThumb: FloatArray,
                                        styleSide: Int,
                                        private val imageSize: Int,
                                        private val lutSize: Int,
                                        val strength: Float,
                                        private val hidden: Int) {

    /** 桌面预设路径（P1 对拍锁定的入口，语义零改动）。 */
    constructor(preset: EncoderPreset, onnxBytes: ByteArray) : this(
        onnxBytes, preset.styleThumb, preset.thumbSide,
        preset.imageSize, preset.lutSize, preset.strength, preset.hidden)

    private val net: Net = sharedNet(onnxBytes)
    private val wStyle: FlowWeights =
        forwardWeights(net, styleThumb, styleSide, hidden).second

    /** 序列化用（P2.14）：style_thumb 的量化值与边长（toPresetJson 写回用）。 */
    private val styleThumbRef: FloatArray = styleThumb
    private val styleSideRef: Int = styleSide

    // LUT 按内容缓存单条（桌面 _lut_for_content 同策略）
    private var lut: Lut3D? = null
    private var lutKey: Long = 0

    /** 缩略图前向 → flat f64。flat 与桌面同序（Θc | Θs）。 */
    private fun forwardFlat(thumb: FloatArray, side: Int): DoubleArray {
        // ⚠ 桌面 `_weights` 喂网络前做 HWC→CHW 转置（thumb.transpose(2,0,1)）——
        // 必须同样平面化，否则 Θ 全错（实测：布局错位 → 权重垃圾 → LUT 近似恒等）
        val px = side * side
        val chw = FloatArray(thumb.size)
        for (p in 0 until px) {
            chw[p] = thumb[3 * p]
            chw[px + p] = thumb[3 * p + 1]
            chw[2 * px + p] = thumb[3 * p + 2]
        }
        val flatIn = Mat(1, chw.size, CvType.CV_32F)
        flatIn.put(0, 0, chw)
        net.setInput(flatIn.reshape(1, intArrayOf(1, 3, side, side)))
        val out = net.forward().reshape(1, 1)
        val flat = FloatArray(out.cols())
        out.get(0, 0, flat)
        return DoubleArray(flat.size) { flat[it].toDouble() }   // f32→f64 精确
    }

    /** 内容图 → 128² 缩略图（float32 INTER_AREA，与桌面 _thumb_of 同参）。 */
    fun thumbOf(image: FloatArray, w: Int, h: Int): FloatArray =
        resizeThumb(image, w, h, imageSize)

    /** 分块管线的缩略图入口：直接吃全分辨率 CV_32FC3 Mat（免整图 FloatArray）。 */
    fun thumbFromMat(src32: Mat): FloatArray {
        val dst = Mat()
        Imgproc.resize(src32, dst,
                       Size(imageSize.toDouble(), imageSize.toDouble()),
                       0.0, 0.0, Imgproc.INTER_AREA)
        val out = FloatArray(imageSize * imageSize * 3)
        dst.get(0, 0, out)
        return out
    }

    /** 该内容的 33³ LUT（构建后缓存，键 = 缩略图内容哈希）。 */
    fun lutFor(contentThumb: FloatArray): Lut3D {
        var key = 1125899906842597L
        for (v in contentThumb) key = key * 31 + v.hashCode()
        val cached = lut
        if (cached != null && key == lutKey) return cached
        val flatD = forwardFlat(contentThumb, imageSize)
        val half = flatD.size / 2
        val wC = FlowWeights.fromFlat(flatD.copyOfRange(0, half), hidden)
        // 33³ 探针网格：linspace(0,1,size) 双精度，indexing="ij"（R 变化最慢）
        val n = lutSize
        val probes = DoubleArray(n * n * n * 3)
        val ax = DoubleArray(n) { it.toDouble() / (n - 1) }
        var q = 0
        for (ri in 0 until n) for (gi in 0 until n) for (bi in 0 until n) {
            probes[q++] = ax[ri]; probes[q++] = ax[gi]; probes[q++] = ax[bi]
        }
        // 逐探针迁移（transfer 是单点 API——桌面版是 (N,3) 全数组；
        // 直接传整条 probes 只会迁移第 0 个点，LUT 退化成单位表）
        // 历史教训（2026-10-01 性能轮，V2324A）：Kotlin 线程并行此循环无收益
        // （2 线程 = 串行、8 线程 2.5× 负优化，疑 SoC 功耗墙/GC 叠加），当时
        // 保持串行；2026-10-02 正解落地 = 探针循环 native 化（parallelFor 并行
        // 骨架，double 逐位口径不变，NativeKernelsTest.flowProbesIsBitExact 锁），
        // 批量每张重建 LUT 的 ~1.4s 就此消失。
        if (NativeKernels.useNative) {
            if (NativeKernels.flowProbes(probes, wC.pack(), wStyle.pack(),
                                         hidden, 1.0, 1)) {
                lut = Lut3D(n, probes); lutKey = key
                return lut!!
            }
            NativeKernels.useNative = false
            com.colortrace.DebugLog.e(
                "native flowProbes 失败→回退纯 Kotlin 探针循环（后续不再尝试）")
        }
        val x = DoubleArray(3)
        var qi = 0
        for (i in 0 until n * n * n) {
            x[0] = probes[qi]; x[1] = probes[qi + 1]; x[2] = probes[qi + 2]
            transfer(x, wC, wStyle, 1.0, 1)
            probes[qi] = x[0]; probes[qi + 1] = x[1]; probes[qi + 2] = x[2]
            qi += 3
        }
        val lutNew = Lut3D(n, probes)
        lut = lutNew; lutKey = key
        return lutNew
    }

    /** 纯 encoder 全局追色，构造时预设强度（桌面 EncoderTransform.apply）。 */
    fun apply(image: FloatArray, w: Int, h: Int): FloatArray =
        apply(image, w, h, strength)

    /** 指定强度的全局追色（`_with_strength` 语义：全强度映射 ↔ 原图线性混合）。 */
    fun apply(image: FloatArray, w: Int, h: Int, strength: Float): FloatArray {
        val mapped = lutFor(thumbOf(image, w, h)).apply(image)
        return withStrength(image, mapped, strength)
    }

    /** chroma 肤色锁定组合（桌面 SkinProtectTransform chroma 分支，skin_match=0）。 */
    fun applyChroma(image: FloatArray, w: Int, h: Int, protect: Float): FloatArray =
        applyChroma(image, w, h, protect, strength)

    /** 指定强度的 chroma 组合：LAB 锚定作用在强度混合后的全局结果上（桌面同序）。 */
    fun applyChroma(image: FloatArray, w: Int, h: Int, protect: Float,
                    strength: Float): FloatArray {
        val outGlobal = apply(image, w, h, strength)
        if (protect <= 0f) return outGlobal          // 桌面 protect<=0 直接返回全局
        val out = FloatArray(image.size)
        if (NativeKernels.chromaAnchorApply(image, outGlobal, protect, out)) return out
        val labIn = LabConv.rgbToLab(image)
        val labOut = LabConv.rgbToLab(outGlobal)
        val weight = SkinWeight.of(labIn)
        return LabConv.labToRgb(SkinWeight.anchorChroma(labIn, labOut, weight, protect))
    }

    // ---- 反向序列化（P2.14） ----

    /**
     * 引擎 → 桌面同构预设 JSON。**Θs 不存**（桌面同口径）：只存 128² 样片缩略图
     * （base64 PNG），载入时重新前向。
     *
     * style_thumb 走 [thumbToPngBase64]（`round(v*255)` → RGB→BGR → PNG）——与桌面
     * `EncoderTransform.__init__` 的 8-bit 量化 + `_thumb_b64` 同口径（`round` 是
     * Kotlin `rint`＝banker's rounding，对齐 `np.round`）。配合 [quantizeThumb]，
     * 「构造 → 存预设 → 读回」的 Θs 逐位一致。
     */
    fun toPresetJson(onnxName: String = DEFAULT_ONNX, meta: JSONObject? = null): String {
        val p = JSONObject()
        p.put("arch", ARCH_NAME)
        p.put("hidden", hidden)
        p.put("image_size", imageSize)
        p.put("onnx", onnxName)
        p.put("lut_size", lutSize)
        p.put("style_thumb", thumbToPngBase64(styleThumbRef, styleSideRef))
        p.put("meta", meta ?: JSONObject())
        return JSONObject()
            .put("version", 1)
            .put("kind", "encoder")
            .put("strength", strength.toDouble())
            .put("params", p)
            .toString()
    }

    /** 样片缩略图 PNG（库列表缩略图用，与 style_thumb 同字节口径）。 */
    fun sampleThumbPng(): ByteArray =
        Base64.getDecoder().decode(thumbToPngBase64(styleThumbRef, styleSideRef))

    companion object {
        /** 库内统一强度语义：原图 ↔ 完整映射线性混合（float32，桌面 _with_strength 同）。 */
        fun withStrength(image: FloatArray, mapped: FloatArray, strength: Float): FloatArray {
            if (strength <= 0f) return image
            if (strength >= 1f) return mapped
            val out = FloatArray(image.size)
            val s = strength
            for (i in image.indices) {
                out[i] = image[i] * (1f - s) + mapped[i] * s
            }
            return out
        }

        // colorfm-l-cnn128 结构常量（桌面 preset params；设备端 fit 无 JSON 可读时用）
        const val ARCH_NAME = "colorfm-l-cnn128"
        const val DEFAULT_ONNX = "encoder_cnn128.onnx"
        const val ARCH_IMAGE_SIZE = 128
        const val ARCH_LUT_SIZE = 33
        const val ARCH_HIDDEN = 128

        /** 量化到 8-bit 网格（桌面 `EncoderTransform.__init__` 同口径：
         *  `(clip(t,0,1)*255).round()/255`，round＝rint/banker's）。 */
        fun quantizeThumb(t: FloatArray): FloatArray =
            FloatArray(t.size) {
                (round(t[it].coerceIn(0f, 1f) * 255.0) / 255.0).toFloat()
            }

        /**
         * 交错 RGB float [0,1] 缩略图 → base64 PNG（桌面 `_thumb_b64` 同口径）：
         * `round(v*255)` → RGB→BGR → PNG（无损，往返逐位）。JPEG 会让编码器输入
         * 漂移（桌面实测往返 ~0.08），必须 PNG。
         */
        fun thumbToPngBase64(thumb: FloatArray, side: Int): String {
            val px = side * side
            require(thumb.size == px * 3) {
                "缩略图长度 ${thumb.size} 与边长 $side 不符"
            }
            val bgr = ByteArray(px * 3)      // OpenCV imencode 吃 BGR 序
            for (i in 0 until px) {
                bgr[3 * i] = round(thumb[3 * i + 2].coerceIn(0f, 1f) * 255.0).toInt().toByte()
                bgr[3 * i + 1] = round(thumb[3 * i + 1].coerceIn(0f, 1f) * 255.0).toInt().toByte()
                bgr[3 * i + 2] = round(thumb[3 * i].coerceIn(0f, 1f) * 255.0).toInt().toByte()
            }
            val m = Mat(side, side, CvType.CV_8UC3)
            m.put(0, 0, bgr)
            val buf = MatOfByte()
            try {
                check(Imgcodecs.imencode(".png", m, buf)) { "缩略图 PNG 编码失败" }
                return Base64.getEncoder().encodeToString(buf.toArray())
            } finally {
                m.release(); buf.release()
            }
        }

        /**
         * 设备端 fit：样片图 → 128² 缩略前向得 Θs（桌面 GUI analyze_sample 同语义）。
         * 与预设路径共用 forward/LUT 管线。
         *
         * P2.14：缩略图先 [quantizeThumb] 再前向——对齐桌面 `EncoderTransform`
         * 构造期的 8-bit 量化，使「设备端 fit → 存预设库 → 读回」的 Θs 逐位一致
         * （此前未量化，存入库后读回会有亚 1/255 级的 Θs 漂移）。
         */
        fun fromSample(sampleImage: FloatArray, w: Int, h: Int,
                       onnxBytes: ByteArray, strength: Float = 1f): EncoderEngine =
            EncoderEngine(onnxBytes,
                          quantizeThumb(resizeThumb(sampleImage, w, h, ARCH_IMAGE_SIZE)),
                          ARCH_IMAGE_SIZE, ARCH_IMAGE_SIZE, ARCH_LUT_SIZE,
                          strength, ARCH_HIDDEN)

        /** float32 INTER_AREA 缩略（桌面 _thumb_of 同参同实现）。 */
        fun resizeThumb(image: FloatArray, w: Int, h: Int, size: Int): FloatArray {
            val src = Mat(h, w, CvType.CV_32FC3)
            src.put(0, 0, image)
            val dst = Mat()
            Imgproc.resize(src, dst, Size(size.toDouble(), size.toDouble()),
                           0.0, 0.0, Imgproc.INTER_AREA)
            val out = FloatArray(size * size * 3)
            dst.get(0, 0, out)
            return out
        }

        private fun loadNet(onnxBytes: ByteArray): Net {
            // 模型别走文件系统（插桩上下文不可靠）：直接吃字节数组
            val m = Mat(1, onnxBytes.size, CvType.CV_8U)
            m.put(0, 0, onnxBytes)
            return Dnn.readNetFromONNX(MatOfByte(m))
        }

        // 解析后的 ONNX 网按内容摘要跨实例共享（P2.12）：Session 重建（换样片/
        // 重载预设）不再重复 parse 6.5MB。调用全部串行在引擎线程——与
        // EngineRepository.selfieNet 的单例共享同模式。
        private val netCache = java.util.concurrent.ConcurrentHashMap<String, Net>()

        private fun sharedNet(onnxBytes: ByteArray): Net {
            val key = java.security.MessageDigest.getInstance("SHA-256").digest(onnxBytes)
                .joinToString("") { "%02x".format(it) }
            return netCache.getOrPut(key) { loadNet(onnxBytes) }
        }

        private fun forwardWeights(net: Net, thumb: FloatArray, side: Int,
                                   hidden: Int): Pair<FlowWeights, FlowWeights> {
            // 这段前向与实例 forwardFlat 同式；独立成静态供属性初始化器使用
            val px = side * side
            val chw = FloatArray(thumb.size)
            for (p in 0 until px) {
                chw[p] = thumb[3 * p]
                chw[px + p] = thumb[3 * p + 1]
                chw[2 * px + p] = thumb[3 * p + 2]
            }
            val flatIn = Mat(1, chw.size, CvType.CV_32F)
            flatIn.put(0, 0, chw)
            net.setInput(flatIn.reshape(1, intArrayOf(1, 3, side, side)))
            val out = net.forward().reshape(1, 1)
            val flat = FloatArray(out.cols())
            out.get(0, 0, flat)
            val flatD = DoubleArray(flat.size) { flat[it].toDouble() }
            val half = flatD.size / 2
            return Pair(FlowWeights.fromFlat(flatD.copyOfRange(0, half), hidden),
                        FlowWeights.fromFlat(flatD.copyOfRange(half, flatD.size), hidden))
        }
    }
}
