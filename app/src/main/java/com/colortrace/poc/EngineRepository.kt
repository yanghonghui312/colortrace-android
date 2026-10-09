package com.colortrace.poc

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import androidx.core.content.FileProvider
import com.colortrace.DebugLog
import com.colortrace.engine.ContentModel
import com.colortrace.engine.EncoderEngine
import com.colortrace.engine.EncoderPreset
import com.colortrace.engine.LabConv
import com.colortrace.engine.NativeKernels
import com.colortrace.engine.RegionEngine
import com.colortrace.engine.SemanticSegmentation
import com.colortrace.engine.StatEngine
import com.colortrace.engine.StatPresets
import kotlin.math.roundToInt
import org.json.JSONObject
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.MatOfInt
import org.opencv.core.Size
import org.opencv.dnn.Dnn
import org.opencv.dnn.Net
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File

/** 肤色保护三档（与桌面 GUI 的 region/chroma/关 对应；face 档不上移动端——桌面
 *  实测它覆盖最差，移动端兜底由 region 的 chroma 回退承担）。
 *  默认 OFF：不一定每张照片都有人（用户定稿）——有人时用户自己开。 */
enum class ProtectMode(val label: String) {
    OFF("关闭"), CHROMA("肤色锁定"), REGION("分区保护")
}

/** Bitmap → 交错 RGB float32 [0,1]（引擎的统一图像表示，桌面 io.py 同语义）。 */
fun bitmapToRgbF32(bmp: Bitmap): Triple<FloatArray, Int, Int> {
    val argb = if (bmp.config == Bitmap.Config.ARGB_8888) bmp
    else bmp.copy(Bitmap.Config.ARGB_8888, false)
    val w = argb.width
    val h = argb.height
    val px = IntArray(w * h)
    argb.getPixels(px, 0, w, 0, 0, w, h)
    if (argb !== bmp) argb.recycle()
    val out = FloatArray(px.size * 3)
    for (i in px.indices) {
        val v = px[i]
        out[3 * i] = (v shr 16 and 0xFF) / 255f
        out[3 * i + 1] = (v shr 8 and 0xFF) / 255f
        out[3 * i + 2] = (v and 0xFF) / 255f
    }
    return Triple(out, w, h)
}

/** float32 RGB [0,1] → Bitmap（u8 用 (x*255+0.5) 截断，与桌面引擎输出同语义）。 */
fun f32ToBitmap(rgb: FloatArray, w: Int, h: Int): Bitmap {
    val px = IntArray(w * h)
    fun u8(v: Float): Int = ((v.coerceIn(0f, 1f) * 255f + 0.5f).toInt())
    for (p in px.indices) {
        px[p] = (0xFF shl 24) or (u8(rgb[3 * p]) shl 16) or
                (u8(rgb[3 * p + 1]) shl 8) or u8(rgb[3 * p + 2])
    }
    return Bitmap.createBitmap(px, w, h, Bitmap.Config.ARGB_8888)
}

/**
 * Bitmap → JPEG 字节（**质量 100 + 4:4:4 无色度子采样**），与桌面
 * `io._imencode_params` 同参数。
 *
 * 为什么不用 `Bitmap.compress`：它不暴露色度采样，Android/Skia 实际输出
 * **4:2:0**——色度横竖各减半，红/青边缘的色度损失与质量参数无关（桌面当年
 * 正是踩了"默认 4:2:0 批量导出肉眼可见色度被压"才改成 q100 + 4:4:4）。
 * 参数常量由 OpenCV 5.0.0 Android 绑定提供（`IMWRITE_JPEG_SAMPLING_FACTOR[_444]`）。
 *
 * 内存：RGBA/BGR 中间 Mat 在 OpenCV 原生内存；返回值是 Java 堆上的 byte[]
 * （12MP q100/4:4:4 ≈ 10~15MB），调用方用完即可回收。
 */
fun encodeJpeg444(bmp: Bitmap, quality: Int = 100): ByteArray {
    val rgba = Mat()
    val bgr = Mat()
    try {
        Utils.bitmapToMat(bmp, rgba)                  // ARGB_8888 → CV_8UC4（RGBA）
        Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
        return encodeJpeg444FromBgr(bgr, quality)
    } finally {
        rgba.release(); bgr.release()
    }
}

/** Mat 版编码主体（P2.12 流式导出直用，免 Bitmap 中转）。bgr 由调用方管理释放。 */
fun encodeJpeg444FromBgr(bgr: Mat, quality: Int = 100): ByteArray {
    val buf = MatOfByte()
    val params = MatOfInt(
        Imgcodecs.IMWRITE_JPEG_QUALITY, quality.coerceIn(1, 100),
        Imgcodecs.IMWRITE_JPEG_SAMPLING_FACTOR,
        Imgcodecs.IMWRITE_JPEG_SAMPLING_FACTOR_444)
    try {
        if (!Imgcodecs.imencode(".jpg", bgr, buf, params)) {
            throw IllegalStateException("JPEG 编码失败（imencode 返回 false）")
        }
        return buf.toArray()
    } finally {
        buf.release(); params.release()
    }
}

/**
 * 引擎会话：桌面预设或设备端 fit 产出的 [ContentModel] + 分区网。
 * 引擎非线程安全（LUT 缓存）——调用方串行使用（MainActivity 的 apply 任务互斥）。
 *
 * P2.13：model 泛化为 ContentModel（encoder / lab_stats / ot_linear）；统计类
 * 整图路径支持 OFF / CHROMA（REGION 的整图版未实现——产品全走 TiledPipeline
 * 分块路径，分割权重在那里可用）。
 */
class Session internal constructor(
    val model: ContentModel,
    private val selfieNet: Net?,
    /** 库条目携带的**全部方法档**（methodId → 预设 JSON；P2.16 一次 fit 全算法存一档）。
     *  预设会话换方法 = 从这里取档 loadPreset（无样片像素也成立）；空 map = 不可换。 */
    val bundleVariants: Map<String, String> = emptyMap(),
) {
    /** encoder 专用（region/chroma 保护路径）；统计类会话不可访问。 */
    val engine: EncoderEngine
        get() = (model as ContentModel.Encoder).engine

    /** 请求的**方法档**（P2.14 存库用）：预设 kind 的同名字符串。
     *  encoder 会话是内容自适应（换图重算 LUT）；统计类与图无关。 */
    val methodKind: String
        get() = when (val m = model) {
            is ContentModel.Encoder -> "encoder"
            is ContentModel.LabStats -> "lab_stats"
            is ContentModel.OtLinear -> "ot_linear"
            // 2026-09-30 合并：分区档用**基方法**作 key——方法选择器只留 3 档算法，
            // 「用不用分区」由保护模式决定；一个 key 只存一份数据（region 结构内含 global）
            is ContentModel.RegionStat -> m.baseKind
        }

    /** 会话是否**带分区数据**（可切「分区保护」）。 */
    val isRegion: Boolean get() = model is ContentModel.RegionStat

    /** UI 方法 pill / 预设名用的方法 id（**3 档**；分区档即其基方法）。 */
    val methodId: String
        get() = when (methodKind) {
            "lab_stats" -> "reinhard"
            "ot_linear" -> "ot"
            else -> "encoder"
        }

    /** 会话 → 桌面同构预设 JSON（P2.14 存入预设库 / 断点续传快照用）。 */
    fun toPresetJson(): String = when (val m = model) {
        is ContentModel.Encoder -> m.engine.toPresetJson()
        is ContentModel.LabStats, is ContentModel.OtLinear,
        is ContentModel.RegionStat -> StatPresets.toPresetJson(m)
    }

    /** 全流程 apply：Bitmap → FloatArray → 引擎 → Bitmap。
     *  @param protect 保护强度 0~1（默认 1＝皮肤恒原色，与桌面默认一致） */
    fun apply(bitmap: Bitmap, mode: ProtectMode, strength: Float,
              protect: Float = 1f): Bitmap {
        val (rgb, w, h) = bitmapToRgbF32(bitmap)
        val out = when (model) {
            is ContentModel.Encoder -> when (mode) {
                ProtectMode.OFF -> model.engine.apply(rgb, w, h, strength)
                ProtectMode.CHROMA -> model.engine.applyChroma(rgb, w, h, protect, strength)
                ProtectMode.REGION ->
                    if (selfieNet == null)   // 语义模型不可用 → 点态锁定（桌面同款回退）
                        model.engine.applyChroma(rgb, w, h, protect, strength)
                    else RegionEngine(model.engine, selfieNet, protect)
                        .apply(rgb, w, h, protect, strength)
            }
            is ContentModel.LabStats -> {
                require(mode != ProtectMode.REGION) {
                    "统计类整图路径不支持 REGION——用编辑页分块路径"
                }
                val mapped = withStatMap(model, rgb)
                when (mode) {
                    ProtectMode.CHROMA -> chromaAnchor(rgb, mapped, protect, strength)
                    else -> mixStrength(rgb, mapped, strength)
                }
            }
            is ContentModel.OtLinear -> {
                require(mode != ProtectMode.REGION) {
                    "统计类整图路径不支持 REGION——用编辑页分块路径"
                }
                val mapped = withStatMap(model, rgb)
                when (mode) {
                    ProtectMode.CHROMA -> chromaAnchor(rgb, mapped, protect, strength)
                    else -> mixStrength(rgb, mapped, strength)
                }
            }
            is ContentModel.RegionStat -> throw IllegalArgumentException(
                "分区统计档请走分块路径（TiledPipeline）——整图 apply 未实现")
        }
        return f32ToBitmap(out, w, h)
    }

    /** 统计类整图全强度映射（内容统计 → 仿射）。 */
    private fun withStatMap(model: ContentModel, rgb: FloatArray): FloatArray =
        when (model) {
            is ContentModel.LabStats -> {
                val (mu, std) = StatEngine.labStats(LabConv.rgbToLab(rgb))
                StatEngine.mapLabStats(rgb, model.refMean, model.refStd, mu, std)
            }
            is ContentModel.OtLinear -> {
                val (mu, cov) = StatEngine.meanCov(rgb)
                val t = StatEngine.otLinearMap(cov, model.refCov)
                StatEngine.mapOtLinear(rgb, t, mu, model.refMean)
            }
            is ContentModel.Encoder -> throw IllegalStateException()
            is ContentModel.RegionStat -> throw IllegalStateException()
        }

    companion object {
        /** chroma 锚定（render 的 chroma 分支同式）：globalS 的 a/b 锚回原片。 */
        fun chromaAnchor(rgb: FloatArray, mapped: FloatArray,
                         protect: Float, strength: Float): FloatArray {
            val globalS = EncoderEngine.withStrength(rgb, mapped, strength)
            if (protect <= 0f) return globalS
            val out = FloatArray(rgb.size)
            if (NativeKernels.chromaAnchorApply(rgb, globalS, protect, out)) return out
            val labIn = LabConv.rgbToLab(rgb)
            val labOut = LabConv.rgbToLab(globalS)
            val weight = com.colortrace.engine.SkinWeight.of(labIn)
            return LabConv.labToRgb(
                com.colortrace.engine.SkinWeight.anchorChroma(labIn, labOut, weight, protect))
        }

        /** 强度混合（`_with_strength` 语义：mapped ↔ 原图线性混合）。 */
        fun mixStrength(rgb: FloatArray, mapped: FloatArray, strength: Float):
                FloatArray = when {
            strength >= 1f -> mapped
            strength <= 0f -> rgb
            else -> FloatArray(rgb.size) {
                rgb[it] * (1f - strength) + mapped[it] * strength
            }
        }
    }
}

/**
 * 保护区域叠加位图：透明底 + 绿色（桌面 `_MASK_TINT` 0.15/0.95/0.30），
 * `alpha = 0.45 × mask`（桌面 `gui/main.py:1483` 同一口径）。
 * 只影响显示——**不写进 afterImg**（保存/分享要用未叠加的结果）。
 */
fun protectionOverlayBitmap(mask: FloatArray, w: Int, h: Int): Bitmap {
    val px = IntArray(w * h)
    val tintR = 38   // 0.15 × 255
    val tintG = 242  // 0.95 × 255
    val tintB = 77   // 0.30 × 255
    for (i in px.indices) {
        val a = (0.45f * mask[i].coerceIn(0f, 1f) * 255f + 0.5f).toInt()
        px[i] = if (a <= 0) 0 else (a shl 24) or (tintR shl 16) or (tintG shl 8) or tintB
    }
    return Bitmap.createBitmap(px, w, h, Bitmap.Config.ARGB_8888)
}

/**
 * 引擎仓库：资产模型加载、图像编解码（Bitmap ↔ 交错 RGB float32 [0,1]）、
 * 设备端 fit / 桌面预设载入、保存与分享。
 */
/** 导入解析结果（P2.17）：name 取文件自带或文件名；methods 至少一档（不可用的档
 *  已在 skippedKinds 里跳过并回报，不静默）；sampleThumb 是 base64 解码后的 PNG。 */
data class ImportedPreset(
    val name: String,
    val methods: Map<String, JSONObject>,
    val sampleThumb: ByteArray?,
    val skippedKinds: List<String>,
)

class EngineRepository(private val context: Context) {

    companion object {
        // 样片解码上限：fit 只用 128² 缩略，2048 足够且省内存
        const val PROC_MAX = 2048
        // 照片预览解码上限（P2.7：2048→1024 换交互跟手；导出/批量仍全分辨率）
        const val PREVIEW_MAX = 1024
        const val THUMB_MAX = 256
    }

    private val encoderOnnx: ByteArray by lazy { assetBytes("encoder_cnn128.onnx") }
    private val selfieOnnx: ByteArray by lazy { assetBytes("selfie_multiclass.onnx") }

    /** 语义分割网；缺失/失败时 region 档回退 chroma（不崩）——但**必须记日志**，
     *  否则"分割没跑"会静默消失，只剩下保护看起来没生效。 */
    val selfieNet: Net? by lazy {
        runCatching {
            val m = Mat(1, selfieOnnx.size, CvType.CV_8U)
            m.put(0, 0, selfieOnnx)
            Dnn.readNetFromONNX(MatOfByte(m))
        }.onFailure {
            DebugLog.e("selfie 分割模型加载失败→region 将回退 chroma", it)
        }.getOrNull()
    }

    /**
     * 设备端 fit：样片 → 方法档（P2.13 起支持算法选择）。
     *  - encoder：样片前向得 Θs（桌面 GUI analyze_sample 同语义）；
     *  - reinhard / ot：样片统计量（LAB 均值方差 / RGB 均值协方差+RIDGE）——
     *    与桌面 `methods.fit(sample, method)` 同语义；样片输入为 PROC_MAX
     *    解码（统计量对缩放不敏感，移动端自有路径无对拍锚）。
     *
     * @param region 统计类是否**同时建立分区档**（桌面 `analyze_sample_region`：全局 +
     *        各区独立 fit，占比 ≥0.5% 的区建映射、否则 null=回退全局）。默认 false
     *        =纯全局（便宜、无分割依赖）；用户切到「分区保护」时按**同张样片**升档
     *        （2026-09-30 合并：方法选择器 3 档 + 保护模式驱动）。语义上分区档**包含**
     *        plain（`params.global` 与 plain 是同一份统计）⇒ 升档不换调子（OFF 模式下
     *        两者逐位一致，`RegionStatTest.plainEqualsRegionAtOff` 有闸门）。
     */
    fun fitSample(sample: Bitmap, method: String = "encoder",
                  region: Boolean = false): Session {
        val (rgb, w, h) = bitmapToRgbF32(sample)
        val model: ContentModel = when (method) {
            "encoder" -> ContentModel.Encoder(
                EncoderEngine.fromSample(rgb, w, h, encoderOnnx))
            "reinhard" -> if (region) fitRegion(rgb, w, h, "lab_stats") else {
                val (mu, std) = StatEngine.labStats(LabConv.rgbToLab(rgb))
                ContentModel.LabStats(mu, std, strength = 1f)
            }
            "ot" -> if (region) fitRegion(rgb, w, h, "ot_linear") else {
                val (mu, cov) = StatEngine.meanCov(rgb)   // 已含 RIDGE（与桌面 fit 同）
                ContentModel.OtLinear(mu, cov, strength = 1f)
            }
            else -> throw IllegalArgumentException(
                "未知方法档: $method（可选 encoder / reinhard / ot）")
        }
        return Session(model, selfieNet)
    }

    /**
     * **一次 fit 全算法存一档**（P2.16，用户原始诉求②）：同张样片 fit 出全部 3 档，
     * 返回 **kind 键**的会话 map（`encoder` / `lab_stats` / `ot_linear`——预设库存储
     * 约定，`Entry.methods` 即这些键）。
     *
     * 各档形态（2026-09-30 定稿）：
     *  - `encoder`：plain（region×encoder 是路线 0 的壳——各区共享全局映射 ⇒ 存 plain
     *    不损失任何能力，REGION 档在套用时由 [RegionEngine] 按内容分割承担）；
     *  - `lab_stats` / `ot_linear`：**region 变体**（⊇ plain：`params.global` 与 plain
     *    是同一份统计，OFF/肤色锁定照常用，分区保护直接可用——`plainEqualsRegionAtOff`
     *    锁 OFF 逐位等价）。预设会话无样片像素 ⇒ 存 plain 就永远失去分区能力。
     *
     * 分割只跑一次，reinhard/ot 两档共用同一张区索引图（`sampleRegionOf`）。
     * 供「存入预设库」一次写入全档；会话主档仍由 `fitSample` 按用户所选拉。
     */
    fun fitAllSamplePresets(sample: Bitmap): Map<String, Session> {
        val (rgb, w, h) = bitmapToRgbF32(sample)
        val out = LinkedHashMap<String, Session>()
        out["encoder"] = Session(
            ContentModel.Encoder(EncoderEngine.fromSample(rgb, w, h, encoderOnnx)), selfieNet)
        val regionOf = sampleRegionOf(rgb, w, h)
        out["lab_stats"] = Session(fitRegionFrom(rgb, regionOf, w, h, "lab_stats"), selfieNet)
        out["ot_linear"] = Session(fitRegionFrom(rgb, regionOf, w, h, "ot_linear"), selfieNet)
        return out
    }

    /**
     * 分区档设备端 fit（P2.15）：样片分割 → 各区 hard 掩码统计 → 每区独立 fit。
     * 桌面 `analyze_sample_region` 同语义（样片缩 ≤2000 是有桌面锚的口径；移动端
     * 已在 PROC_MAX=2048 解码，统计对缩放不敏感 ⇒ 自有路径无对拍锚，同 P2.13 统计档）。
     */
    private fun fitRegion(rgb: FloatArray, w: Int, h: Int, baseKind: String):
            ContentModel.RegionStat =
            fitRegionFrom(rgb, sampleRegionOf(rgb, w, h), w, h, baseKind)

    /** 样片 → 全分辨率区索引图（0=skin 1=hair 2=cloth 3=bg）。分割只跑一次，
     *  供多个基方法共用（fitAllSamplePresets 的 reinhard/ot 两档同一张）。 */
    private fun sampleRegionOf(rgb: FloatArray, w: Int, h: Int): ByteArray {
        val net = selfieNet ?: throw IllegalArgumentException(
            "语义分割模型不可用——无法建立分区预设（encoder/reinhard/ot 仍可用）")
        val (cls, _) = SemanticSegmentation.segment(net, rgb, w, h)   // 256² 类图
        val side = SemanticSegmentation.INPUT_SIDE
        val m = Mat(side, side, CvType.CV_8UC1)
        m.put(0, 0, cls)
        val dst = Mat()
        Imgproc.resize(m, dst, Size(w.toDouble(), h.toDouble()), 0.0, 0.0,
                       Imgproc.INTER_NEAREST)          // 与桌面 resize(cls,(w,h),NEAREST) 同
        m.release()
        val regionOf = ByteArray(w * h).also { dst.get(0, 0, it) }
        dst.release()
        val n = w * h
        for (p in 0 until n) regionOf[p] = StatEngine.regionIndexOf(regionOf[p].toInt()).toByte()
        return regionOf
    }

    /** 分区档 fit 的统计段（分割结果 [regionOf] 已备好）：每区占比 <0.5% 置 null
     *  （缺席区回退全局 ref，桌面 `MIN_REGION_PIXELS`/`_MIN_REGION_RATIO` 同语义）。 */
    private fun fitRegionFrom(rgb: FloatArray, regionOf: ByteArray, w: Int, h: Int,
                              baseKind: String): ContentModel.RegionStat {
        val n = w * h
        val counts = IntArray(4)
        for (p in 0 until n) counts[regionOf[p].toInt()]++

        val isLab = baseKind == "lab_stats"
        val lab = if (isLab) LabConv.rgbToLab(rgb) else null
        val global: ContentModel = if (isLab) {
            val (mu, std) = StatEngine.labStats(lab!!)
            ContentModel.LabStats(mu, std, strength = 1f)
        } else {
            val (mu, cov) = StatEngine.meanCov(rgb)
            ContentModel.OtLinear(mu, cov, strength = 1f)
        }
        val keys = StatPresets.REGION_KEYS
        val ratios = LinkedHashMap<String, Double>()
        val regions = LinkedHashMap<String, ContentModel?>()
        for (k in 0 until 4) {
            val ratio = counts[k].toDouble() / n
            ratios[keys[k]] = ratio
            if (ratio < 0.005) {          // 桌面 _MIN_REGION_RATIO
                regions[keys[k]] = null
                continue
            }
            val sel = BooleanArray(n)
            for (p in 0 until n) sel[p] = regionOf[p].toInt() == k
            regions[keys[k]] = if (isLab) {
                val (mu, std) = StatEngine.maskedLabStats(lab!!, sel)
                ContentModel.LabStats(mu, std, strength = 1f)
            } else {
                val (mu, cov) = StatEngine.maskedMeanCov(rgb, sel)
                ContentModel.OtLinear(mu, cov, strength = 1f)
            }
        }
        DebugLog.i("分区 fit 完成（$baseKind）占比 " +
                keys.joinToString(" ") { "$it=${"%.1f".format(ratios[it]!! * 100)}%" })
        return ContentModel.RegionStat(global, regions, ratios, "global", 1f)
    }

    /**
     * 载入桌面预设 JSON（`fit` 产物）。P2.13 起按 kind 分发：encoder / lab_stats
     * （reinhard）/ ot_linear（ot）；其余档（swot/phr/region 包裹的统计类等）
     * 明确报错——不静默降级。
     *
     * [variants]：库条目的全部方法档（methodId → 预设 JSON，P2.16）——预设会话
     * 换方法用；空 map（默认）= 单档会话、方法 pill 置灰。
     */
    fun loadPreset(jsonText: String,
                   variants: Map<String, String> = emptyMap()): Session {
        val root = JSONObject(jsonText)
        val kind = root.optString("kind", "encoder")
        val model: ContentModel = when {
            kind == "encoder" ->
                ContentModel.Encoder(EncoderEngine(EncoderPreset(jsonText), encoderOnnx))
            // 分区 × encoder（桌面路线 0）：各区共享同一全局映射（apply_region 忽略
            // mask）⇒ 还原为 encoder 会话即可，REGION 档由 [RegionEngine] 承担，零新数学
            kind == "region" && regionBaseKind(root) == "encoder" -> {
                val g = root.getJSONObject("params").getJSONObject("global").toString()
                ContentModel.Encoder(EncoderEngine(EncoderPreset(g), encoderOnnx))
            }
            else -> StatPresets.parse(root)   // 不支持的 kind 抛 IllegalArgumentException
        }
        return Session(model, selfieNet, variants)
    }

    /** 分区预设的基方法 kind（`params.global.kind`）；缺失返回空串。 */
    private fun regionBaseKind(root: JSONObject): String =
        root.optJSONObject("params")?.optJSONObject("global")?.optString("kind") ?: ""

    /**
     * 「从文件导入」的统一解析（P2.17）：
     *  - **bundle**（含 `methods` 对象——预设库导出文件的形态）：逐档校验（[loadPreset]），
     *    本机不支持的 kind **跳过不静默**（[ImportedPreset.skippedKinds] 里回报）；
     *    全部不可用则抛错（不入库）。
     *  - **单档预设**（桌面 `fit` 产物 JSON）：校验通过后包成单档 bundle，
     *    缩略图取 encoder 的 style_thumb（统计类为 null）。
     * name 优先用文件自带（bundle.name），否则用调用方给的 fallback（文件名）。
     */
    fun parseImportableBundle(text: String, fallbackName: String): ImportedPreset {
        val root = JSONObject(text)
        val methodsObj = root.optJSONObject("methods")
        if (methodsObj == null) {
            // 单档预设：loadPreset 校验（不支持的 kind 抛错），包成单档
            val s = loadPreset(text)
            val kind = s.methodKind
            return ImportedPreset(fallbackName, mapOf(kind to JSONObject(text)),
                                  sampleThumbPng(s), emptyList())
        }
        val name = root.optString("name").ifBlank { fallbackName }
        val thumbB64 = if (root.isNull("sampleThumb")) null else root.optString("sampleThumb")
        val thumb = thumbB64?.takeIf { it.isNotEmpty() }?.let {
            runCatching { java.util.Base64.getDecoder().decode(it) }.getOrNull()
        }
        val methods = LinkedHashMap<String, JSONObject>()
        val skipped = mutableListOf<String>()
        val keys = mutableListOf<String>()
        methodsObj.keys().forEach { keys.add(it) }
        for (k in keys) {
            val obj = methodsObj.optJSONObject(k)
            if (obj == null) { skipped.add(k); continue }
            val parse = runCatching { loadPreset(obj.toString()) }
            if (parse.isSuccess) methods[k] = obj
            else skipped.add(k)   // 本机不支持的档——跳过并回报，不静默
        }
        if (methods.isEmpty()) throw IllegalArgumentException(
            "文件里没有任何本机可用的档（${skipped.joinToString("、")}）")
        return ImportedPreset(name, methods, thumb, skipped)
    }

    /** 会话的样片缩略图 PNG（预设库存库/列表用）：encoder 可从 style_thumb 反解，
     *  统计类预设不含图像（返回 null，库列表显示方法标签占位）。 */
    fun sampleThumbPng(session: Session): ByteArray? =
        (session.model as? ContentModel.Encoder)?.engine?.sampleThumbPng()

    // ---- 图像 IO ----

    /**
     * 导出取图（P2.21）：[shortSide] ≤ 0 = 原始尺寸（不缩放，P2.12 以来口径不变）；
     * > 0 = 短边目标像素（长边按比例）。调用方只认这一个入口，别自己算 maxSide。
     */
    fun decodeForExport(uri: Uri, shortSide: Int): Bitmap =
        if (shortSide <= 0) decodeCapped(uri, Int.MAX_VALUE)
        else decodeShortSide(uri, shortSide)

    /**
     * 按**短边**目标解码（P2.21 预压缩导出）：短边 > [shortSide] 时缩到短边 = 目标、
     * 长边按比例；原图短边已 ≤ 目标则**原样返回（不放大）**。
     *
     * 复用 [decodeCapped] 那条路径（inSampleSize 采样 → 精确缩放 → EXIF 摆正），只是
     * 把"短边口径"换算成它要的"长边上限"。EXIF 旋转只交换长短边，**不改变短边口径**
     * （竖拍图 4000×3000 摆正后 3000×4000，短边仍是 3000）⇒ 这里用未旋转的 bounds
     * 计算是对的。
     */
    fun decodeShortSide(uri: Uri, shortSide: Int): Bitmap {
        val (w, h) = boundsOf(uri)
        val long = maxOf(w, h)
        val short = minOf(w, h)
        if (short <= shortSide) return decodeCapped(uri, Int.MAX_VALUE)
        // 四舍五入：短边缩到目标后，长边跟着比例走（误差 ≤1px，与桌面导出同量级）
        val targetLong = ((shortSide.toLong() * long + short / 2) / short).toInt()
        return decodeCapped(uri, targetLong)
    }

    /** 只读图片头拿像素尺寸（不解码像素）。读不到时抛异常，由调用方转人话。 */
    private fun boundsOf(uri: Uri): Pair<Int, Int> {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        context.contentResolver.openInputStream(uri).use {
            if (it == null) throw IllegalArgumentException("无法读取: $uri")
            BitmapFactory.decodeStream(it, null, bounds)
        }
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
            throw IllegalArgumentException("无法识别图片尺寸: $uri")
        }
        return bounds.outWidth to bounds.outHeight
    }

    fun decodeCapped(uri: Uri, maxSide: Int): Bitmap {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        // ⚠ decodeStream 在 inJustDecodeBounds 下"正常"返回 null——不能用
        //   openInputStream()?.use{...} ?: throw 的写法，成功路径也会触发 throw
        context.contentResolver.openInputStream(uri).use {
            if (it == null) throw IllegalArgumentException("无法读取: $uri")
            BitmapFactory.decodeStream(it, null, bounds)
        }
        var inSample = 1
        var s = maxOf(bounds.outWidth, bounds.outHeight)
        while (s / 2 >= maxSide) {
            inSample *= 2
            s /= 2
        }
        val bmp = context.contentResolver.openInputStream(uri).use {
            it ?: throw IllegalArgumentException("无法读取: $uri")
            BitmapFactory.decodeStream(it, null, BitmapFactory.Options().apply {
                this.inSampleSize = inSample
            })
        } ?: throw IllegalArgumentException("解码失败: $uri")
        val long = maxOf(bmp.width, bmp.height)
        val sized = if (long <= maxSide) bmp else {
            val scale = maxSide.toFloat() / long
            // 四舍五入而非截断（2026-10-01 用户拍板"预压缩短边正好 1920"）：float32 下
            // 1920.0 会算成 1919.9997，截断吃成 1919（真机 3712×5568 → 1919×2880 实测）。
            // f32 相对误差 ~1e-7 ⇒ 乘积至多 maxSide + 0.0003px，round 不会越过上限。
            val scaled = Bitmap.createScaledBitmap(
                bmp, (bmp.width * scale).roundToInt().coerceAtLeast(1),
                (bmp.height * scale).roundToInt().coerceAtLeast(1), true)
            if (scaled != bmp) bmp.recycle()
            scaled
        }
        // 竖拍照片的摆正放在缩放之后（省内存、不干扰采样逻辑）——
        // BitmapFactory 不读 EXIF orientation，不修就会把竖拍图强制横置
        return applyExifOrientation(uri, sized)
    }

    /**
     * 按 EXIF orientation 摆正（2..8 全量映射；1/UNDEFINED 原样返回）。
     * 用框架自带 `android.media.ExifInterface`（API 24+）——**零新增依赖**。
     * 非 JPEG／部分 HEIC 会抛 IOException ⇒ 兜底为"正常方向"并记日志。
     */
    private fun applyExifOrientation(uri: Uri, bmp: Bitmap): Bitmap {
        val o = runCatching {
            context.contentResolver.openInputStream(uri)?.use {
                android.media.ExifInterface(it).getAttributeInt(
                    android.media.ExifInterface.TAG_ORIENTATION,
                    android.media.ExifInterface.ORIENTATION_NORMAL)
            } ?: android.media.ExifInterface.ORIENTATION_NORMAL
        }.onFailure {
            DebugLog.e("读取 EXIF 方向失败 $uri（按正常方向处理）", it)
        }.getOrDefault(android.media.ExifInterface.ORIENTATION_NORMAL)

        val m = Matrix()
        when (o) {
            android.media.ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> m.setScale(-1f, 1f)
            android.media.ExifInterface.ORIENTATION_ROTATE_180 -> m.setRotate(180f)
            android.media.ExifInterface.ORIENTATION_FLIP_VERTICAL -> {
                m.setRotate(180f); m.postScale(-1f, 1f)
            }
            android.media.ExifInterface.ORIENTATION_TRANSPOSE -> {
                m.setRotate(90f); m.postScale(-1f, 1f)
            }
            android.media.ExifInterface.ORIENTATION_ROTATE_90 -> m.setRotate(90f)
            android.media.ExifInterface.ORIENTATION_TRANSVERSE -> {
                m.setRotate(-90f); m.postScale(-1f, 1f)
            }
            android.media.ExifInterface.ORIENTATION_ROTATE_270 -> m.setRotate(-90f)
            else -> {
                DebugLog.i("EXIF orientation=$o（无需旋转）" +
                        " ${bmp.width}x${bmp.height}")
                return bmp
            }
        }
        val rotated = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
        DebugLog.i("EXIF orientation=$o 已摆正 " +
                "${bmp.width}x${bmp.height} → ${rotated.width}x${rotated.height}")
        if (rotated != bmp) bmp.recycle()
        return rotated
    }

    // 图像编解码使用文件级 bitmapToRgbF32 / f32ToBitmap（Session 与仓库共用）

    // ---- 保存 / 分享 ----

    /** 保存到相册（API 29+ MediaStore）或应用外部目录（<29）；返回 (uri, 是否相册)。 */
    fun saveBitmap(bmp: Bitmap, displayName: String): Pair<Uri, Boolean>? =
        // 先编码再建 MediaStore 行：编码失败不留空条目（旧写法失败时会残留一行的坑）
        saveJpegBytes(encodeJpeg444(bmp), displayName)

    /** 已编码 JPEG 字节落盘（P2.12 流式导出：管线直接产字节，不再有全尺寸 Bitmap）。 */
    fun saveJpegBytes(bytes: ByteArray, displayName: String): Pair<Uri, Boolean>? {
        val name = "$displayName.jpg"
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, name)
                put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                put(MediaStore.Images.Media.RELATIVE_PATH,
                    Environment.DIRECTORY_PICTURES + "/colortrace")
                // IS_PENDING：写完才置 0——崩溃/被杀只留不可见的 pending 行，
                // 不会在相册里出现半截图（断点续传重跑也不产生重复可见产物）
                put(MediaStore.Images.Media.IS_PENDING, 1)
            }
            val uri = context.contentResolver
                .insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
                ?: return null
            try {
                val os = context.contentResolver.openOutputStream(uri)
                    ?: throw IllegalStateException("MediaStore 输出流打开失败")
                os.use { it.write(bytes) }
                values.clear()
                values.put(MediaStore.Images.Media.IS_PENDING, 0)
                context.contentResolver.update(uri, values, null, null)
            } catch (e: Throwable) {
                // 写失败（含中断）：删掉本行，不让空/半截条目进相册
                runCatching { context.contentResolver.delete(uri, null, null) }
                throw e
            }
            return uri to true
        }
        val dir = context.getExternalFilesDir(Environment.DIRECTORY_PICTURES) ?: return null
        val f = File(dir, name)
        f.outputStream().use { it.write(bytes) }
        return FileProvider.getUriForFile(
            context, "com.colortrace.poc.fileprovider", f) to false
    }

    private fun assetBytes(name: String): ByteArray =
        context.assets.open(name).use { it.readBytes() }
}
