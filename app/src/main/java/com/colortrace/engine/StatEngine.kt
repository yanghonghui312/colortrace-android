package com.colortrace.engine

import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.max
import kotlin.math.round
import kotlin.math.sqrt

/**
 * 统计类方法档的 Kotlin 移植（P2.13）：reinhard（`lab_stats`）与 ot（`ot_linear`）。
 *
 * 背景：桌面 GUI 现列 5 档，此前移动端只认 encoder——桌面存的其余档预设加载
 * 直接失败。本文件补两档**闭式解**方法（预设小、无迭代）；swot（点云投影）与
 * phr（直方图重塑+双边 detail）仍"按需"（见 PLAN §7）。
 *
 * 对拍协议（`scripts/export_android_stats_vectors.py` 金标）：
 *  - 统计量 float64 归约（桌面 `_source_desc`/`_mean_cov` 同语义）；
 *  - 仿射在 **float32**（桌面 scale/bias/t 均 astype(float32) 后运算）；
 *  - ot 的 3×3 特征分解 numpy eigh ↔ 本文件 Jacobi：数值路径不同（特征向量
 *    符号/顺序任意性不影响 T），T 容差 1e-6、最终 u8 ≤2/255（P1 同源关卡）；
 *  - strength 语义与 encoder 同：`mapped ↔ 原图`线性混合（`_with_strength`）。
 *
 * **不支持**：region/chroma 包裹的统计类预设（kind=region/skin_protect 且基方法
 * 为统计类）——解析与迁移路径明确拒绝，不静默降级。
 */
sealed class ContentModel {
    /** 预设强度（滑杆初值；render 时由 UI 传值，mapped 物化始终全强度）。 */
    abstract val strength: Float

    /** 零样本编码器（现役主路径，P1 移植）。 */
    class Encoder(val engine: EncoderEngine) : ContentModel() {
        override val strength: Float get() = engine.strength
    }

    /** reinhard：LAB 均值/方差（桌面 `methods/reinhard.py`，kind=lab_stats）。 */
    class LabStats(
        val refMean: DoubleArray,   // (3) float64 JSON → 保持 double；仿射时转 float
        val refStd: DoubleArray,
        override val strength: Float,
    ) : ContentModel()

    /** ot：RGB 均值 + 3×3 协方差（桌面 `methods/ot.py`，kind=ot_linear）。 */
    class OtLinear(
        val refMean: DoubleArray,   // (3)
        val refCov: Array<DoubleArray>,   // (3,3)，已含 RIDGE
        override val strength: Float,
    ) : ContentModel()

    /**
     * 分区统计档（P2.15，桌面 `methods/region_transfer.py`，kind="region"，基方法
     * reinhard/ot）：全局 + 四区各一组统计量（样片侧 `analyze_sample_region` 的
     * 各区独立 fit）。apply 时**内容侧也按区统计**（每区"内容该区统计 → 对齐样片
     * 该区统计"，`apply_region` 不忽略 mask）→ RGB 域加权融合 → L 取全局映射。
     *
     * 与 encoder 路线 0 的区别：encoder 的 `apply_region` 忽略 mask（各区共享同一
     * 全局映射），故现有 [RegionEngine] 就是该形态；统计类的每区映射**各不相同**，
     * 需要本档 + 分区段级内核。
     */
    class RegionStat(
        val global: ContentModel,                  // LabStats | OtLinear
        val regions: Map<String, ContentModel?>,   // 键序 REGION_KEYS；null=样片无该区
        val ratios: Map<String, Double>,           // 样片各区占比（仅信息/展示）
        val luma: String,
        override val strength: Float,
    ) : ContentModel() {
        /** 基方法 kind（预设库标签 / 方法 pill 用）。 */
        val baseKind: String get() = when (global) {
            is LabStats -> "lab_stats"
            is OtLinear -> "ot_linear"
            else -> "region"
        }
    }
}

/** 预设 JSON → [ContentModel]（按 kind 分发；桌面 `transforms.py` 序列化原样）。 */
object StatPresets {
    /** 分区键序（与桌面 `semantic.REGION_KEYS` 一致）。 */
    val REGION_KEYS = listOf("skin", "hair", "cloth", "bg")

    fun parse(jsonText: String): ContentModel = parse(JSONObject(jsonText))

    /** 由已解析的 JSONObject 起点分发（region 嵌套解析复用）。 */
    fun parse(root: JSONObject): ContentModel {
        val kind = root.getString("kind")
        val strength = root.getDouble("strength").toFloat()
        val p = root.getJSONObject("params")
        return when (kind) {
            "lab_stats" -> ContentModel.LabStats(
                refMean = p.getJSONArray("ref_mean").toDouble3(),
                refStd = p.getJSONArray("ref_std").toDouble3(),
                strength = strength)
            "ot_linear" -> ContentModel.OtLinear(
                refMean = p.getJSONArray("ref_mean").toDouble3(),
                refCov = p.getJSONArray("ref_cov").toMat3(),
                strength = strength)
            "region" -> parseRegion(p, strength)
            else -> throw IllegalArgumentException(
                "手机端暂不支持的方法档: $kind（已支持 encoder / reinhard / ot / region；" +
                        "swot / phr 请用桌面端）")
        }
    }

    /**
     * 分区预设（kind="region"）嵌套解析：`params.global` + `params.regions`（四区，
     * 可为 null=样片无该区）各自递归解析为标准统计档。**luma≠"global" 诚实报错**
     * （桌面 `_LUMA_MODES` 还有 region/keep，移动端产品面只用默认 global）。
     * **encoder 基的分区预设不走这里**（其 apply_region 忽略 mask ⇒ 现有 [RegionEngine]
     * 就是该形态）——由 `EngineRepository.loadPreset` 单独放行。
     */
    private fun parseRegion(p: JSONObject, strength: Float): ContentModel.RegionStat {
        val luma = p.optString("luma", "global")
        require(luma == "global") {
            "手机端分区预设仅支持 luma=\"global\"（实得 \"$luma\"）——请用桌面端导出 global 档"
        }
        val gJson = p.getJSONObject("global")
        val gKind = gJson.optString("kind")
        require(gKind == "lab_stats" || gKind == "ot_linear") {
            "分区预设基方法 $gKind 暂不支持（已支持 lab_stats/ot_linear；" +
                    "encoder 基请走 EngineRepository.loadPreset）"
        }
        val global = parse(gJson)
        val rJson = p.optJSONObject("regions")
        val regions = LinkedHashMap<String, ContentModel?>()
        for (k in REGION_KEYS) {
            val v = rJson?.opt(k)
            val nested = if (v == null || v === JSONObject.NULL) null
                         else parse(v as JSONObject)
            if (nested != null) {
                require(nested is ContentModel.LabStats || nested is ContentModel.OtLinear) {
                    "分区预设各档须为统计类（$k 实得 ${kindOf(nested)}）"
                }
                require(kindOf(nested) == gKind) {
                    "分区预设各档基方法须一致（global=$gKind，$k=${kindOf(nested)}）"
                }
            }
            regions[k] = nested
        }
        val ratios = LinkedHashMap<String, Double>()
        p.optJSONObject("ratios")?.let { r -> for (k in r.keys()) ratios[k] = r.getDouble(k) }
        return ContentModel.RegionStat(global, regions, ratios, luma, strength)
    }

    private fun kindOf(m: ContentModel): String = when (m) {
        is ContentModel.LabStats -> "lab_stats"
        is ContentModel.OtLinear -> "ot_linear"
        is ContentModel.RegionStat -> "region"
        is ContentModel.Encoder -> "encoder"
    }

    private fun JSONArray.toDouble3(): DoubleArray {
        require(length() == 3) { "期望长度 3，收到 ${length()}" }
        return DoubleArray(3) { getDouble(it) }
    }

    private fun JSONArray.toMat3(): Array<DoubleArray> {
        require(length() == 3) { "期望 3×3，收到 ${length()} 行" }
        return Array(3) { r -> getJSONArray(r).toDouble3() }
    }

    /**
     * 反向序列化（P2.14）：统计类 [ContentModel] → 桌面同构预设 JSON。
     *
     * round 口径与桌面 `_params` **逐位对齐**（金标口径坑：预设 JSON 有量化，
     * 金标锚必须用序列化往返后的值重算）：
     *  - reinhard `_to_jsonable` → 6 位（ref_mean / ref_std）；
     *  - ot `ref_cov` → 8 位、`ref_mean` → 6 位。
     * `kotlin.math.round`＝rint/banker's rounding，对齐 `np.round`（不是 Math.round）。
     */
    fun toPresetJson(model: ContentModel): String {
        val root = JSONObject()
        root.put("version", 1)
        root.put("strength", model.strength.toDouble())
        when (model) {
            is ContentModel.LabStats -> {
                root.put("kind", "lab_stats")
                root.put("params", JSONObject()
                    .put("ref_mean", arr3(model.refMean, 6))
                    .put("ref_std", arr3(model.refStd, 6)))
            }
            is ContentModel.OtLinear -> {
                root.put("kind", "ot_linear")
                root.put("params", JSONObject()
                    .put("ref_mean", arr3(model.refMean, 6))
                    .put("ref_cov", JSONArray().apply {
                        for (r in 0 until 3) put(arr3(model.refCov[r], 8))
                    }))
            }
            is ContentModel.RegionStat -> {
                root.put("kind", "region")
                val regionsJson = JSONObject()
                for (k in REGION_KEYS) {
                    val m = model.regions[k]
                    regionsJson.put(k, if (m == null) JSONObject.NULL
                                        else JSONObject(toPresetJson(m)))
                }
                val ratiosJson = JSONObject()
                for ((k, v) in model.ratios) ratiosJson.put(k, rnd(v, 5))   // 桌面 round(v,5)
                root.put("params", JSONObject()
                    .put("global", JSONObject(toPresetJson(model.global)))
                    .put("regions", regionsJson)
                    .put("ratios", ratiosJson)
                    .put("luma", model.luma)
                    .put("protect", 1.0))       // 移动端 protect 由 UI 滑杆给，此处写默认
            }
            is ContentModel.Encoder ->
                throw IllegalArgumentException("encoder 会话用 EncoderEngine.toPresetJson 序列化")
        }
        return root.toString()
    }

    private fun arr3(v: DoubleArray, digits: Int): JSONArray =
        JSONArray().apply { for (x in v) put(rnd(x, digits)) }

    private fun rnd(v: Double, digits: Int): Double {
        val f = Math.pow(10.0, digits.toDouble())
        return round(v * f) / f
    }
}

/** 统计量与映射（整图版：Session.apply 与测试用；分块版在 TiledPipeline 内联）。 */
object StatEngine {

    // ---- reinhard ----

    /**
     * LAB 统计量（float64 两遍归约，桌面 `_source_desc` 同语义）：
     * mean 与 std（`sqrt(mean((x-mean)²))`，std 下限 1e-6）。
     */
    fun labStats(lab: FloatArray): Pair<DoubleArray, DoubleArray> {
        val n = lab.size / 3
        require(n > 0) { "空数组" }
        val sum = DoubleArray(3)
        for (p in 0 until n) {
            sum[0] += lab[3 * p].toDouble(); sum[1] += lab[3 * p + 1].toDouble()
            sum[2] += lab[3 * p + 2].toDouble()
        }
        val mean = DoubleArray(3) { sum[it] / n }
        val sq = DoubleArray(3)
        for (p in 0 until n) {
            for (c in 0 until 3) {
                val d = lab[3 * p + c].toDouble() - mean[c]
                sq[c] += d * d
            }
        }
        val std = DoubleArray(3) { max(sqrt(sq[it] / n), 1e-6) }
        return mean to std
    }

    /**
     * reinhard 全强度映射：LAB 逐通道仿射（**float32**，与桌面 `_map` 同）→
     * clip L∈[0,100] / ab∈[-127,127] → RGB。输入输出均为交错 RGB [0,1]。
     */
    fun mapLabStats(
        rgb: FloatArray, refMean: DoubleArray, refStd: DoubleArray,
        srcMean: DoubleArray, srcStd: DoubleArray,
    ): FloatArray {
        // scale/bias：float64 统计量转 float32 后再算（桌面 np.asarray(..., float32) 同；
        // ⚠ 桌面 bias = ref_mean - src_mean * scale，scale 是先算好的 float32——
        // 必须同序：scale 先 f32，bias 用 f32 乘加，不能在 double 域合并算完再转）
        val scale = FloatArray(3) { c ->
            (refStd[c] / max(srcStd[c], 1e-6)).toFloat()
        }
        val bias = FloatArray(3) { c ->
            refMean[c].toFloat() - srcMean[c].toFloat() * scale[c]
        }
        val lab = LabConv.rgbToLab(rgb)
        val out = FloatArray(lab.size)
        val nPx = lab.size / 3
        for (p in 0 until nPx) {
            val i = 3 * p
            var l = lab[i] * scale[0] + bias[0]
            var a = lab[i + 1] * scale[1] + bias[1]
            var b = lab[i + 2] * scale[2] + bias[2]
            if (l < 0f) l = 0f else if (l > 100f) l = 100f
            if (a < -127f) a = -127f else if (a > 127f) a = 127f
            if (b < -127f) b = -127f else if (b > 127f) b = 127f
            out[i] = l; out[i + 1] = a; out[i + 2] = b
        }
        return LabConv.labToRgb(out)
    }

    // ---- ot ----

    /**
     * RGB 均值与协方差（float64 两遍累加，桌面 `_mean_cov` 同语义：float32 输入、
     * float64 累加、除以 max(n-1,1)）；返回协方差已含 RIDGE（+1e-6·I）。
     */
    fun meanCov(rgb: FloatArray): Pair<DoubleArray, Array<DoubleArray>> {
        val n = rgb.size / 3
        require(n > 1) { "像素数必须 >1" }
        val sum = DoubleArray(3)
        for (p in 0 until n) {
            sum[0] += rgb[3 * p].toDouble(); sum[1] += rgb[3 * p + 1].toDouble()
            sum[2] += rgb[3 * p + 2].toDouble()
        }
        val mu = DoubleArray(3) { sum[it] / n }
        val outer = Array(3) { DoubleArray(3) }
        for (p in 0 until n) {
            val d0 = rgb[3 * p].toDouble() - mu[0]
            val d1 = rgb[3 * p + 1].toDouble() - mu[1]
            val d2 = rgb[3 * p + 2].toDouble() - mu[2]
            outer[0][0] += d0 * d0; outer[0][1] += d0 * d1; outer[0][2] += d0 * d2
            outer[1][1] += d1 * d1; outer[1][2] += d1 * d2
            outer[2][2] += d2 * d2
        }
        val inv = 1.0 / max(n - 1, 1)
        val cov = Array(3) { r -> DoubleArray(3) }
        for (r in 0 until 3) for (c in 0 until 3) {
            val sym = if (r <= c) outer[r][c] else outer[c][r]
            cov[r][c] = sym * inv + (if (r == c) 1e-6 else 0.0)
        }
        return mu to cov
    }

    /**
     * 对称 PSD 3×3 的平方根与逆平方根（Jacobi 特征分解；桌面 `_sqrtm_psd`
     * 用 numpy eigh——数值路径不同但 T 唯一，容差 1e-6 锁）。
     */
    fun sqrtmPsd(m: Array<DoubleArray>): Pair<Array<DoubleArray>, Array<DoubleArray>> {
        // 对称化 (m + mᵀ)/2
        val a = Array(3) { r -> DoubleArray(3) { c -> (m[r][c] + m[c][r]) / 2.0 } }
        // Jacobi 特征分解：a = U diag(w) Uᵀ
        val u = Array(3) { r -> DoubleArray(3) { c -> if (r == c) 1.0 else 0.0 } }
        for (sweep in 0 until 32) {
            var off = 0.0
            for (r in 0 until 3) for (c in r + 1 until 3) off += a[r][c] * a[r][c]
            if (off < 1e-28) break           // ⟨1e-14 的非对角元
            for (p in 0 until 2) for (q in p + 1 until 3) {
                if (kotlin.math.abs(a[p][q]) < 1e-300) continue
                val theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                val t = kotlin.math.sign(theta) /
                        (kotlin.math.abs(theta) + sqrt(theta * theta + 1.0))
                val c1 = 1.0 / sqrt(t * t + 1.0)
                val s = t * c1
                for (k in 0 until 3) {
                    val akp = a[k][p]; val akq = a[k][q]
                    a[k][p] = c1 * akp - s * akq
                    a[k][q] = s * akp + c1 * akq
                }
                for (k in 0 until 3) {
                    val apk = a[p][k]; val aqk = a[q][k]
                    a[p][k] = c1 * apk - s * aqk
                    a[q][k] = s * apk + c1 * aqk
                }
                for (k in 0 until 3) {
                    val ukp = u[k][p]; val ukq = u[k][q]
                    u[k][p] = c1 * ukp - s * ukq
                    u[k][q] = s * ukp + c1 * ukq
                }
            }
        }
        val w = DoubleArray(3) { max(a[it][it], 0.0) }   // 桌面 np.maximum(w, 0)
        val root = Array(3) { DoubleArray(3) }
        val invRoot = Array(3) { DoubleArray(3) }
        // A = U·diag(w)·Uᵀ（U 列 = 特征向量）⇒ root[r][c] = Σ_k U[r][k]·√w[k]·U[c][k]
        for (r in 0 until 3) for (c in 0 until 3) {
            var sr = 0.0; var ir = 0.0
            for (k in 0 until 3) {
                sr += u[r][k] * sqrt(w[k]) * u[c][k]
                ir += u[r][k] * (1.0 / sqrt(max(w[k], 1e-12))) * u[c][k]
            }
            root[r][c] = sr; invRoot[r][c] = ir
        }
        return root to invRoot
    }

    /** 高斯 OT 最优线性映射 T（桌面 `ot_linear_map`，float64 全程）。 */
    fun otLinearMap(covSrc: Array<DoubleArray>, covRef: Array<DoubleArray>): Array<DoubleArray> {
        val (srcHalf, srcInvHalf) = sqrtmPsd(covSrc)
        // mid = sqrtm(srcHalf · covRef · srcHalf)
        val mid = sqrtmPsd(mat3(mat3(srcHalf, covRef), srcHalf)).first
        return mat3(mat3(srcInvHalf, mid), srcInvHalf)
    }

    // ---- 分区（hard 掩码）统计：P2.15 ----

    /** 语义类索引（0..5）→ 区索引（0=skin,1=hair,2=cloth,3=bg）——与桌面
     *  `semantic.region_masks_of` 同一份定义（others 并入 bg）。 */
    fun regionIndexOf(cls: Int): Int = when (cls) {
        2, 3 -> 0        // BODY_SKIN, FACE_SKIN
        1 -> 1           // HAIR
        4 -> 2           // CLOTHES
        else -> 3        // BG, OTHERS
    }

    /** 桌面 `transforms.MIN_REGION_PIXELS`：区像素数低于此值退回全图统计。 */
    const val MIN_REGION_PIXELS = 100

    /** 掩码限定 LAB 统计（桌面 `masked_pixels` + reinhard `_source_desc`）。 */
    fun maskedLabStats(lab: FloatArray, sel: BooleanArray): Pair<DoubleArray, DoubleArray> {
        var c = 0
        for (b in sel) if (b) c++
        if (c < MIN_REGION_PIXELS) return labStats(lab)
        val px = FloatArray(c * 3)
        var j = 0
        for (p in sel.indices) if (sel[p]) {
            px[j++] = lab[3 * p]; px[j++] = lab[3 * p + 1]; px[j++] = lab[3 * p + 2]
        }
        return labStats(px)
    }

    /** 掩码限定 RGB 均值/协方差（桌面 `masked_pixels` + ot `_source_desc`）。 */
    fun maskedMeanCov(rgb: FloatArray, sel: BooleanArray):
            Pair<DoubleArray, Array<DoubleArray>> {
        var c = 0
        for (b in sel) if (b) c++
        if (c < MIN_REGION_PIXELS) return meanCov(rgb)
        val px = FloatArray(c * 3)
        var j = 0
        for (p in sel.indices) if (sel[p]) {
            px[j++] = rgb[3 * p]; px[j++] = rgb[3 * p + 1]; px[j++] = rgb[3 * p + 2]
        }
        return meanCov(px)
    }

    /**
     * 分区内容统计累加器（分块路径）：按 hard 掩码区索引（[regionIndexOf]）分段累加。
     * reinhard 单遍 Σ/Σ²；ot 先 Σ 得各区均值、第二遍 [addBandOuter] 加协方差
     * （桌面 `_mean_cov` 两遍式）。**某区像素数 <100 时该区退回全图统计**
     * （桌面 `masked_pixels` 兜底）——finalize 时按 [globalStats] 替换。
     */
    class RegionStatAccumulator(private val isLab: Boolean) {
        val count = IntArray(4)
        private val sum = Array(4) { DoubleArray(3) }
        private val sq = Array(4) { DoubleArray(3) }
        private val outer = Array(4) { Array(3) { DoubleArray(3) } }

        /** 喂一段：rgb 交错 RGB f32，region 每像素区索引（0..3），长度 n。 */
        fun addBand(rgb: FloatArray, region: ByteArray, n: Int) {
            if (isLab) {
                val lab = LabConv.rgbToLab(rgb)
                for (p in 0 until n) {
                    val k = region[p].toInt()
                    count[k]++
                    val i = 3 * p
                    for (c in 0 until 3) {
                        val x = lab[i + c].toDouble()
                        sum[k][c] += x; sq[k][c] += x * x
                    }
                }
            } else {
                for (p in 0 until n) {
                    val k = region[p].toInt()
                    count[k]++
                    val i = 3 * p
                    sum[k][0] += rgb[i].toDouble()
                    sum[k][1] += rgb[i + 1].toDouble()
                    sum[k][2] += rgb[i + 2].toDouble()
                }
            }
        }

        /** ot 第二遍：用各区均值累加协方差（仅 count≥100 的区；其余走全局）。 */
        fun addBandOuter(rgb: FloatArray, region: ByteArray, n: Int,
                         mu: Array<DoubleArray>) {
            for (p in 0 until n) {
                val k = region[p].toInt()
                if (count[k] < MIN_REGION_PIXELS) continue
                val i = 3 * p
                val d0 = rgb[i].toDouble() - mu[k][0]
                val d1 = rgb[i + 1].toDouble() - mu[k][1]
                val d2 = rgb[i + 2].toDouble() - mu[k][2]
                val o = outer[k]
                o[0][0] += d0 * d0; o[0][1] += d0 * d1; o[0][2] += d0 * d2
                o[1][1] += d1 * d1; o[1][2] += d1 * d2
                o[2][2] += d2 * d2
            }
        }

        /** 各区内容均值（ot 第二遍用；<100 的区取全局均值）。 */
        fun means(globalMean: DoubleArray): Array<DoubleArray> = Array(4) { k ->
            if (count[k] < MIN_REGION_PIXELS) globalMean.copyOf()
            else DoubleArray(3) { sum[k][it] / count[k] }
        }

        /** reinhard：各区 (mean,std)；<100 的区退回全局。 */
        fun labStats(globalMean: DoubleArray, globalStd: DoubleArray):
                Pair<Array<DoubleArray>, Array<DoubleArray>> {
            val mean = means(globalMean)
            val std = Array(4) { k ->
                if (count[k] < MIN_REGION_PIXELS) globalStd.copyOf()
                else DoubleArray(3) { c ->
                    val v = sq[k][c] / count[k] - mean[k][c] * mean[k][c]
                    max(sqrt(max(v, 0.0)), 1e-6)
                }
            }
            return mean to std
        }

        /** ot：各区 (mu,cov+RIDGE)；<100 的区退回全局。 */
        fun otStats(globalMu: DoubleArray, globalCov: Array<DoubleArray>):
                Pair<Array<DoubleArray>, Array<Array<DoubleArray>>> {
            val mu = means(globalMu)
            val cov = Array(4) { k ->
                if (count[k] < MIN_REGION_PIXELS) Array(3) { globalCov[it].copyOf() }
                else {
                    val inv = 1.0 / max(count[k] - 1, 1)
                    Array(3) { r -> DoubleArray(3) { c ->
                        val sym = outer[k][minOf(r, c)][maxOf(r, c)]
                        sym * inv + (if (r == c) 1e-6 else 0.0)   // RIDGE（桌面同值）
                    } }
                }
            }
            return mu to cov
        }
    }

    /** 3×3 矩阵乘。 */
    fun mat3(a: Array<DoubleArray>, b: Array<DoubleArray>): Array<DoubleArray> =
        Array(3) { r ->
            DoubleArray(3) { c ->
                var s = 0.0
                for (k in 0 until 3) s += a[r][k] * b[k][c]
                s
            }
        }

    /**
     * ot 全强度映射：RGB 仿射（**float32** 运算：t/mu/ref 转 float32）→ clip [0,1]。
     * 输入输出均为交错 RGB [0,1]。`tT` 传 [otLinearMap] 结果的**转置会自动处理**——
     * 桌面 `(flat - mu32) @ t.T` 等价于 `out[i] = Σ_j t[i][j]·(x[j]−mu[j])`。
     */
    fun mapOtLinear(
        rgb: FloatArray, t: Array<DoubleArray>,
        muSrc: DoubleArray, refMean: DoubleArray,
    ): FloatArray {
        val t32 = Array(3) { r -> FloatArray(3) { c -> t[r][c].toFloat() } }
        val mu32 = FloatArray(3) { muSrc[it].toFloat() }
        val ref32 = FloatArray(3) { refMean[it].toFloat() }
        val out = FloatArray(rgb.size)
        val nPx = rgb.size / 3
        for (p in 0 until nPx) {
            val i = 3 * p
            val x0 = rgb[i] - mu32[0]; val x1 = rgb[i + 1] - mu32[1]; val x2 = rgb[i + 2] - mu32[2]
            var r = t32[0][0] * x0 + t32[0][1] * x1 + t32[0][2] * x2 + ref32[0]
            var g = t32[1][0] * x0 + t32[1][1] * x1 + t32[1][2] * x2 + ref32[1]
            var b = t32[2][0] * x0 + t32[2][1] * x1 + t32[2][2] * x2 + ref32[2]
            if (r < 0f) r = 0f else if (r > 1f) r = 1f
            if (g < 0f) g = 0f else if (g > 1f) g = 1f
            if (b < 0f) b = 0f else if (b > 1f) b = 1f
            out[i] = r; out[i + 1] = g; out[i + 2] = b
        }
        return out
    }
}
