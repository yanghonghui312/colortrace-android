package com.colortrace.engine

import com.colortrace.DebugLog
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.dnn.Net
import org.opencv.imgproc.Imgproc

/**
 * 语义分割（selfie_multiclass）的 Kotlin 移植（桌面 `segmentation/semantic.py`）。
 *
 * 预处理与 softmax 逐式对应桌面：u8 = (clip*255+0.5) 截断 → 256² INTER_AREA →
 * /255 → NHWC blob → 减最大值 → exp → 归一。输出概率转 **CHW**（6,256,256）——
 * 桌面 segment 返回 CHW，区域权重按 CHW 索引（曾经 NHWC/CHW 混淆的教训）。
 * P0 已证前向 probs 与桌面一致（max|Δ|≈1.8e-06，类图逐位一致）。
 */
object SemanticSegmentation {
    const val INPUT_SIDE = 256

    /** 返回 (cls 256² u8, probs CHW 6×256² float32)；输入为交错 RGB float [0,1]。 */
    fun segment(net: Net, image: FloatArray, w: Int, h: Int):
            Pair<ByteArray, FloatArray> {
        val n = image.size / 3
        val u8 = ByteArray(n * 3)
        for (i in u8.indices) {
            u8[i] = ((image[i].coerceIn(0f, 1f) * 255.0 + 0.5).toInt()).toByte()
        }
        val src = Mat(h, w, CvType.CV_8UC3)
        src.put(0, 0, u8)
        val result = segmentMat(net, src)
        src.release()
        return result
    }

    /**
     * 分块管线入口：直接吃全分辨率 CV_8UC3 Mat（通道序 RGB，语义与上面的
     * u8 构造完全一致——u8 = (clip*255+0.5) 截断，对不透明 Bitmap 而言
     * 位图字节就是同一个值）。返回 (cls 256² u8, probs CHW 6×256² float32)。
     */
    fun segmentMat(net: Net, src8: Mat): Pair<ByteArray, FloatArray> {
        val w = src8.cols()
        val h = src8.rows()
        val src = src8
        val small = Mat()
        Imgproc.resize(src, small, Size(INPUT_SIDE.toDouble(), INPUT_SIDE.toDouble()),
                       0.0, 0.0, Imgproc.INTER_AREA)
        val side = INPUT_SIDE
        val px = side * side
        val smallU8 = ByteArray(px * 3)
        small.get(0, 0, smallU8)
        val f32 = FloatArray(px * 3)
        for (i in f32.indices) f32[i] = (smallU8[i].toInt() and 0xFF) / 255f
        val blob = Mat(1, f32.size, CvType.CV_32F)
        blob.put(0, 0, f32)
        net.setInput(blob.reshape(1, intArrayOf(1, side, side, 3)))
        val out = net.forward().reshape(1, side)
        val logits = FloatArray(px * 6)
        out.get(0, 0, logits)

        val cls = ByteArray(px)
        val probsChw = FloatArray(6 * px)
        val e = FloatArray(6)
        var k = 0
        for (p in 0 until px) {
            var m = Float.NEGATIVE_INFINITY
            for (c in 0 until 6) m = maxOf(m, logits[k + c])
            var s = 0f
            for (c in 0 until 6) { e[c] = kotlin.math.exp(logits[k + c] - m); s += e[c] }
            var best = 0; var bv = 0f
            for (c in 0 until 6) {
                val pr = e[c] / s
                probsChw[c * px + p] = pr
                if (pr > bv) { bv = pr; best = c }
            }
            cls[p] = best.toByte()
            k += 6
        }
        return Pair(cls, probsChw)
    }
}

/**
 * 分区迁移（全局/兜底层基方法 + 语义皮肤保护）的 Kotlin 移植（桌面
 * `methods/region_transfer.py`，2026-09-27 路线 0 形态）。
 *
 * 桌面语义：皮肤区 protect=1 恒原色、protect<1 向全局追色插值（基方法的
 * apply_region 忽略 mask ⇒ 皮肤分支与非肤区共享同一次全局映射结果）；亮度
 * 策略 luma_mode="global"（融合结果只保留分区色调 a/b，L 取全局映射）。
 * 权重域 1600px + 羽化 σ=边长 0.4% + 归一化 Σw=1（GaussianBlur/线性上采样
 * 与桌面同一 OpenCV 实现）。P1 范围：分割失效回退走 chroma 锁定（桌面
 * `_fallback_chroma` 同式）。
 *
 * P2.13：基方法泛化为 [ContentModel]——统计类（reinhard/ot）走 `withLut=false`
 * 的分块路径（融合层只吃权重与调用方物化的 mapped，与基方法无关）；整图
 * apply 仍是 encoder 专用（统计类整图保护走 Session.apply 的 chroma 分支）。
 */
class RegionEngine constructor(private val model: ContentModel,
                               private val selfieNet: Net,
                               private val protect: Float) {

    /** encoder 档兼容构造（既有调用点零改动）。 */
    constructor(encoder: EncoderEngine, selfieNet: Net, protect: Float) :
            this(ContentModel.Encoder(encoder) as ContentModel, selfieNet, protect)

    private val encoder: EncoderEngine? get() = (model as? ContentModel.Encoder)?.engine

    private class RegionField(val skin: FloatArray, val hair: FloatArray,
                              val cloth: FloatArray, val bg: FloatArray,
                              val ww: Int, val wh: Int)

    /** 桌面 _region_weights：概率 CHW → 4 区权重（键序 skin,hair,cloth,bg）。 */
    private fun regionWeights(probsChw: FloatArray, w: Int, h: Int): RegionField {
        val side = SemanticSegmentation.INPUT_SIDE
        val px = side * side
        val scale = minOf(1.0, 1600.0 / maxOf(w, h))
        val wh = maxOf(8, Math.round(h * scale).toInt())
        val ww = maxOf(8, Math.round(w * scale).toInt())
        val sigma = maxOf(1.0, minOf(wh, ww) * 0.004)
        val dstSize = Size(ww.toDouble(), wh.toDouble())

        fun channel(c: Int): Mat {
            val m = Mat(side, side, CvType.CV_32F)
            m.put(0, 0, probsChw.copyOfRange(c * px, (c + 1) * px))
            return m
        }
        // 先逐通道线性上采样，再求和（桌面 pw[i]=resize(...) 后 w["skin"]=pw[2]+pw[3]）
        fun resized(c: Int): Mat {
            val r = Mat()
            Imgproc.resize(channel(c), r, dstSize, 0.0, 0.0, Imgproc.INTER_LINEAR)
            return r
        }
        fun blurAdd(mats: List<Mat>): Mat {
            val sum = Mat.zeros(wh, ww, CvType.CV_32F)
            for (m in mats) Core.add(sum, m, sum)
            Imgproc.GaussianBlur(sum, sum, Size(0.0, 0.0), sigma)
            return sum
        }
        val skin = blurAdd(listOf(resized(2), resized(3)))   // BODY_SKIN + FACE_SKIN
        val hair = blurAdd(listOf(resized(1)))
        val cloth = blurAdd(listOf(resized(4)))
        val bg = blurAdd(listOf(resized(0), resized(5)))     // BG + OTHERS

        // 归一化 Σw=1（键序 skin→hair→cloth→bg，float32）。
        // 每个字段先整块 get 成 FloatArray 再归一化：逐像素 Mat.get 是 JNI 级
        // 开销（≤1600px 时 ww×wh 即全分辨率，可达数百万次调用），批量读的
        // 数值与累加序完全一致 ⇒ 逐位不变。
        val fields = arrayOf(skin, hair, cloth, bg)
        val n = wh * ww
        val flat = Array(4) { i -> FloatArray(n).also { fields[i].get(0, 0, it) } }
        val out = Array(4) { FloatArray(n) }
        for (i in 0 until n) {
            var s = 0f
            for (r in 0 until 4) s += flat[r][i]
            val denom = maxOf(s, 1e-6f)
            for (r in 0 until 4) out[r][i] = flat[r][i] / denom
        }
        return RegionField(out[0], out[1], out[2], out[3], ww, wh)
    }

    private fun fullRes(weights: FloatArray, ww: Int, wh: Int, w: Int, h: Int):
            FloatArray {
        if (ww == w && wh == h) return weights
        val m = Mat(wh, ww, CvType.CV_32F)
        m.put(0, 0, weights)
        val dst = Mat()
        Imgproc.resize(m, dst, Size(w.toDouble(), h.toDouble()), 0.0, 0.0,
                       Imgproc.INTER_LINEAR)
        val out = FloatArray(w * h)
        dst.get(0, 0, out)
        return out
    }

    /** 分区迁移 apply（桌面 RegionTransform._map，luma_mode="global" 分支）。 */
    fun apply(image: FloatArray, w: Int, h: Int): FloatArray =
        apply(image, w, h, protect)

    /** 指定保护强度/追色强度的分区迁移（P2 UI 用；默认值保持原语义）。 */
    fun apply(image: FloatArray, w: Int, h: Int, protect: Float,
              strength: Float = 1f): FloatArray {
        val enc = encoder ?: throw IllegalStateException(
            "统计类基方法不支持整图 apply——用 TiledPipeline 分块路径")
        val globalOut = enc.apply(image, w, h, strength)
        if (protect <= 0f) return globalOut
        val seg = runCatching {
            SemanticSegmentation.segment(selfieNet, image, w, h)
        }.onFailure {
            DebugLog.e("语义分割失败→回退 chroma 点态锁定 ${w}x${h}", it)
        }.getOrNull()
            ?: return fallbackChroma(image, w, h, globalOut, protect)   // 模型不可用 → 点态锁定

        val field = regionWeights(seg.second, w, h)
        val wSkin = fullRes(field.skin, field.ww, field.wh, w, h)

        // 皮肤分支：protect=1 恒原色；protect<1 向全局追色插值
        val skinBranch: FloatArray =
            if (protect >= 0.999f) image
            else FloatArray(image.size) {
                image[it] * protect + globalOut[it] * (1f - protect)
            }

        // 四区软融合（键序 skin→hair→cloth→bg，float32，与桌面累加序一致；
        // encoder 的 apply_region 忽略 mask ⇒ 非肤三区共享同一次全局映射）
        val out = FloatArray(image.size)
        val branches = listOf(skinBranch, globalOut, globalOut, globalOut)
        val wHair = fullRes(field.hair, field.ww, field.wh, w, h)
        val wCloth = fullRes(field.cloth, field.ww, field.wh, w, h)
        val wBg = fullRes(field.bg, field.ww, field.wh, w, h)
        val wAll = listOf(wSkin, wHair, wCloth, wBg)
        for (r in 0 until 4) {
            val wr = wAll[r]; val br = branches[r]
            for (p in 0 until w * h) {
                val wk = wr[p]
                out[3 * p] += wk * br[3 * p]
                out[3 * p + 1] += wk * br[3 * p + 1]
                out[3 * p + 2] += wk * br[3 * p + 2]
            }
        }

        // 亮度策略 global：融合结果只留 a/b，L 取全局映射（皮肤区排除在替换外）
        val ref = LabConv.rgbToLab(globalOut)
        val labOut = LabConv.rgbToLab(out)
        for (p in 0 until w * h) {
            val lw = wSkin[p]
            labOut[3 * p] = ref[3 * p] * (1f - lw) + labOut[3 * p] * lw
        }
        return LabConv.labToRgb(labOut)
    }

    /** 分割不可用的保护回退（桌面 _fallback_chroma）：全局追色 + a/b 锚回原片。 */
    private fun fallbackChroma(image: FloatArray, w: Int, h: Int,
                               globalOut: FloatArray, protect: Float): FloatArray {
        if (protect <= 0f) return globalOut
        val out = FloatArray(image.size)
        if (NativeKernels.chromaAnchorApply(image, globalOut, protect, out)) return out
        val labIn = LabConv.rgbToLab(image)
        val labOut = LabConv.rgbToLab(globalOut)
        val weight = SkinWeight.of(labIn)
        return LabConv.labToRgb(SkinWeight.anchorChroma(labIn, labOut, weight, protect))
    }

    // ---- 分块（tiled）路径：全分辨率导出用 ----
    //
    // 逐像素数学与整图 apply 完全一致（LUT 查表 / withStrength / 权重融合 /
    // luma 替换都是点态运算），只是把"整图物化"换成"按行段处理"——
    // Java 堆上只存在段级数组（几十 MB），全分辨率的大缓冲全部留在
    // OpenCV 原生内存（Mat）。逐位一致性由 TiledTest 用 0/255 关卡锁住。

    /** 分块会话：内容 LUT（encoder 档；统计类为 null）+ 上采样到全分辨率的原生权重 Mat
     *  + 可选的**全分辨率类图 Mat**（P2.15 分区统计用：hard 掩码来源，用完即弃）。 */
    class Tiled internal constructor(
        val lut: Lut3D?,
        val wSkin: Mat, val wHair: Mat, val wCloth: Mat, val wBg: Mat,
        var clsFull: Mat?,
    ) {
        fun release() {
            wSkin.release(); wHair.release(); wCloth.release(); wBg.release()
            clsFull?.release(); clsFull = null
        }

        /** 释放类图（分区统计算完后立即调用，别让它驻留整个迁移会话）。 */
        fun releaseCls() {
            clsFull?.release(); clsFull = null
        }
    }

    /**
     * 准备段：src32（全分辨率 CV_32FC3）→ 内容缩略图 → LUT；
     * src8（同尺寸 CV_8UC3，RGB 序）→ 语义分割 → 小权重场 → INTER_LINEAR
     * 上采样到全分辨率的原生 Mat（与整图路径的 fullRes 同一 OpenCV 调用，
     * 逐位一致）。输入 Mat 由调用方负责释放；返回的 Tiled 用完要 release。
     *
     * @param withLut 是否构建 encoder 内容 LUT（P2.13：统计类基方法传 false，
     *        只做分割+权重——融合层 applyBandMapped 只吃权重与调用方给的 mapped，
     *        与基方法无关）。默认 true（encoder 档原语义）。
     * @param withHardMasks 是否额外产出全分辨率类图 Mat（`Tiled.clsFull`，P2.15
     *        分区统计的 hard 掩码来源）。默认 false（encoder 路线 0 用不到）。
     * @param timing 可选的阶段计时回调（名字 → 毫秒），默认 null 零开销；
     *        名字集合：`thumb` / `lut` / `seg` / `weights` / `toFull`
     */
    fun prepareTiled(src32: Mat, src8: Mat, withLut: Boolean = true,
                     withHardMasks: Boolean = false,
                     timing: ((String, Double) -> Unit)? = null): Tiled {
        val w = src32.cols()
        val h = src32.rows()
        var t = System.nanoTime()
        fun lap(name: String) {
            val now = System.nanoTime()
            timing?.invoke(name, (now - t) / 1e6)
            t = now
        }
        val lut = if (withLut) {
            val enc = encoder ?: throw IllegalStateException(
                "withLut=true 需要 encoder 基方法（统计类传 withLut=false）")
            enc.lutFor(enc.thumbFromMat(src32)).also { lap("thumb+lut") }
        } else null
        val (cls, probs) = SemanticSegmentation.segmentMat(selfieNet, src8); lap("seg")
        val field = regionWeights(probs, w, h); lap("weights")

        // hard 掩码来源（P2.15）：256² 类图最近邻上采样到全分辨率——与桌面
        // `cv2.resize(cls, (w,h), INTER_NEAREST)` 同一 OpenCV 调用，逐位一致
        val clsFull = if (withHardMasks) {
            val side = SemanticSegmentation.INPUT_SIDE
            val m = Mat(side, side, CvType.CV_8UC1)
            m.put(0, 0, cls)
            val dst = Mat()
            Imgproc.resize(m, dst, Size(w.toDouble(), h.toDouble()),
                           0.0, 0.0, Imgproc.INTER_NEAREST)
            m.release()
            dst
        } else null

        fun toFull(a: FloatArray): Mat {
            val m = Mat(field.wh, field.ww, CvType.CV_32F)
            m.put(0, 0, a)
            val dst = Mat()
            Imgproc.resize(m, dst, Size(w.toDouble(), h.toDouble()),
                           0.0, 0.0, Imgproc.INTER_LINEAR)
            return dst
        }
        val tiled = Tiled(lut, toFull(field.skin), toFull(field.hair),
                          toFull(field.cloth), toFull(field.bg), clsFull)
        lap("toFull")
        return tiled
    }

    /**
     * 应用段：处理一个行段 [y0, y0+rows)。rgb = 该段的交错 RGB float32
     * （从全分辨率 src32 Mat 的对应行段 get 出来，值域/序与整图路径一致）；
     * out = 同长度输出段。逐像素公式与整图 apply 相同（含 float 累加序）。
     */
    fun applyBand(tiled: Tiled, rgb: FloatArray, y0: Int, rows: Int,
                  strength: Float, out: FloatArray) {
        applyBand(tiled, rgb, y0, rows, protect, strength, out)
    }

    fun applyBand(tiled: Tiled, rgb: FloatArray, y0: Int, rows: Int,
                  protect: Float, strength: Float, out: FloatArray) {
        // lut 仅 encoder 档有（统计类走 applyBandMapped 直喂 mapped）——此入口
        // 是 encoder 专用（migrate 时 withLut=true），统计类调不到
        applyBandMapped(tiled, rgb, tiled.lut!!.apply(rgb), y0, rows,
                        protect, strength, out)
    }

    companion object {
        /**
         * hard 掩码区统计入口（P2.15）：全分辨率类图 Mat（[Tiled.clsFull]）+ src32
         * 分段喂入累加器。类索引→区索引在此统一换算（与 [StatEngine.regionIndexOf]
         * 同源）。`outer` 非空时走 ot 第二遍（协方差，用各区均值）。
         */
        fun regionStats(clsFull: Mat, src32: Mat, acc: StatEngine.RegionStatAccumulator,
                        rows: Int, outer: Array<DoubleArray>? = null) {
            val w = src32.cols()
            val h = src32.rows()
            val maskBuf = ByteArray(w * rows)
            val rgbBuf = FloatArray(w * rows * 3)
            for (y0 in 0 until h step rows) {
                val r = minOf(rows, h - y0)
                val n = w * r
                val clsSeg = if (r == rows) maskBuf else ByteArray(n)
                clsFull.get(y0, 0, clsSeg)
                val rgbSeg = if (r == rows) rgbBuf else FloatArray(n * 3)
                src32.get(y0, 0, rgbSeg)
                for (p in 0 until n) {
                    clsSeg[p] = StatEngine.regionIndexOf(clsSeg[p].toInt()).toByte()
                }
                if (outer == null) acc.addBand(rgbSeg, clsSeg, n)
                else acc.addBandOuter(rgbSeg, clsSeg, n, outer)
            }
        }

        /**
         * 应用段（用调用方给定的全强度 LUT 输出 `mapped`）——与 [applyBand]
         * 逐位同式：`mapped` 就是 `tiled.lut.apply(rgb)` 的结果，供两阶段
         * 拆分（TiledPipeline.migrate 缓存 LUT 输出）复用，避免重算 LUT。
         */
        fun applyBandMapped(tiled: Tiled, rgb: FloatArray, mapped: FloatArray,
                            y0: Int, rows: Int, protect: Float,
                            strength: Float, out: FloatArray) {
            val n = tiled.wSkin.cols() * rows
            fun readBand(m: Mat): FloatArray {
                val b = FloatArray(n)
                m.get(y0, 0, b)
                return b
            }
            val wSkin = readBand(tiled.wSkin)
            val wHair = readBand(tiled.wHair)
            val wCloth = readBand(tiled.wCloth)
            val wBg = readBand(tiled.wBg)

            // native 优先：withStrength→四区融合→L 替换→labToRgb 整段一次算完，
            // 且不建中间数组（原 Kotlin 每段要建 5~6 个大 FloatArray）。
            if (NativeKernels.useNative) {
                if (NativeKernels.fuseBand(rgb, mapped, wSkin, wHair, wCloth, wBg,
                                           protect, strength, out)) return
                NativeKernels.useNative = false
                DebugLog.e("native fuseBand 失败→回退纯 Kotlin（后续不再尝试）")
            }
            applyBandMappedKotlin(rgb, mapped, wSkin, wHair, wCloth, wBg,
                                  protect, strength, out, n)
        }

        /** 纯 Kotlin 参考实现——native 的对拍基准，也是回退路径。 */
        internal fun applyBandMappedKotlin(rgb: FloatArray, mapped: FloatArray,
                                          wSkin: FloatArray, wHair: FloatArray,
                                          wCloth: FloatArray, wBg: FloatArray,
                                          protect: Float, strength: Float,
                                          out: FloatArray, n: Int) {
            val globalS = EncoderEngine.withStrength(rgb, mapped, strength)
            if (protect <= 0f) {
                System.arraycopy(globalS, 0, out, 0, n * 3)
                return
            }

            // 皮肤分支：protect=1 恒原色；protect<1 向全局追色插值（整图 apply 同式）
            val skinBranch: FloatArray =
                if (protect >= 0.999f) rgb
                else FloatArray(n * 3) { rgb[it] * protect + globalS[it] * (1f - protect) }

            // 四区软融合（键序 skin→hair→cloth→bg；float 累加序与整图一致）
            val fused = FloatArray(n * 3)
            val branches = listOf(skinBranch, globalS, globalS, globalS)
            val weights = listOf(wSkin, wHair, wCloth, wBg)
            for (r in 0 until 4) {
                val wr = weights[r]; val br = branches[r]
                for (p in 0 until n) {
                    fused[3 * p] += wr[p] * br[3 * p]
                    fused[3 * p + 1] += wr[p] * br[3 * p + 1]
                    fused[3 * p + 2] += wr[p] * br[3 * p + 2]
                }
            }

            // 亮度策略 global：融合结果只留 a/b，L 取全局映射（皮肤区排除在替换外）
            val ref = LabConv.rgbToLab(globalS)
            val labOut = LabConv.rgbToLab(fused)
            for (p in 0 until n) {
                val lw = wSkin[p]
                labOut[3 * p] = ref[3 * p] * (1f - lw) + labOut[3 * p] * lw
            }
            val result = LabConv.labToRgb(labOut)
            System.arraycopy(result, 0, out, 0, n * 3)
        }
    }
}

/**
 * 分区**统计**档（P2.15）的渲染：migrate 期把「样片各区 ref 统计 + 内容各区 src 统计」
 * 压成 [Plan]（每区映射参数），render 期逐段点态融合。
 *
 * 逐像素与桌面 `RegionTransform._map`（luma_mode="global"）+ `_apply_luma_policy`
 * + `apply`（外层 `_with_strength`）逐式对应：
 *   每区仿射（`apply_region`：内容该区统计 → 样片该区统计）→ RGB 域 Σw·t_r →
 *   L 取全局映射（皮肤区排除在替换外）→ 结果 ↔ 原图按外层 strength 线性混合。
 *
 * 与 [RegionEngine]（encoder 路线 0）的差别只在"每区映射是否相同"：encoder 的
 * `apply_region` 忽略 mask（各区共享同一全局映射），统计类每区映射各不相同。
 */
object RegionStatEngine {

    /** 分区渲染计划（无原生内存；权重 Mat 在 [RegionEngine.Tiled] 里随 Migration 释放）。 */
    class Plan internal constructor(
        val isLab: Boolean,                  // true=reinhard(LAB)，false=ot(RGB)
        val t: Array<Array<DoubleArray>>?,   // ot：每区完整 T 矩阵 [4][3][3]
        val refMean: Array<DoubleArray>,     // [4][3]
        val refStd: Array<DoubleArray>?,     // [4][3]（lab）
        val srcMean: Array<DoubleArray>,     // [4][3]（lab=均值 / ot=mu）
        val srcStd: Array<DoubleArray>?,     // [4][3]（lab）
        val regionStrength: FloatArray,      // [4] 各档嵌套强度（通常 1.0）
        val globalStrength: Float,
        /** native 内核（NativeKernels.regionStatBand）的参数打包：[0]=isLab；
         *  lab 路径每区 scale(3)+bias(3)，ot 路径每区 t(9)+mu(3)+ref(3)（均
         *  float32，与 Kotlin 内核逐像素时算的值同式同序 ⇒ 预构建逐位一致）；
         *  尾接 regionStrength(4) + globalStrength。 */
        val packed: FloatArray,
    )

    private fun refStats(m: ContentModel):
            Triple<DoubleArray, DoubleArray?, Array<DoubleArray>?> = when (m) {
        is ContentModel.LabStats -> Triple(m.refMean, m.refStd, null)
        is ContentModel.OtLinear -> Triple(m.refMean, null, m.refCov)
        else -> throw IllegalArgumentException(
            "分区预设基方法须为统计类（实得 ${m.javaClass.simpleName}）")
    }

    /**
     * 由「样片各区 ref（模型）+ 内容各区 src（统计）」构建计划。
     * 缺席区（样片占比过小 ⇒ 预设里 null）**一律回退全局 ref**——含皮肤区
     * （桌面 `_map` 的 `if t is None: t = self.global_transform`；`_skin_branch`
     * 2026-09-30 起同规则）。此前皮肤缺席时恒原色 ⇒ protect 滑块完全失效。
     */
    fun buildPlan(model: ContentModel.RegionStat,
                  srcMean: Array<DoubleArray>, srcStd: Array<DoubleArray>?,
                  srcCov: Array<Array<DoubleArray>>?): Plan {
        val isLab = model.global is ContentModel.LabStats
        val keys = StatPresets.REGION_KEYS
        val refMean = Array(4) { DoubleArray(3) }
        val refStd = if (isLab) Array(4) { DoubleArray(3) } else null
        val t = if (isLab) null else Array(4) { Array(3) { DoubleArray(3) } }
        val strength = FloatArray(4)
        for (k in 0 until 4) {
            val useModel = model.regions[keys[k]] ?: model.global
            val (rm, rs, rc) = refStats(useModel)
            refMean[k] = rm
            strength[k] = useModel.strength
            if (isLab) refStd!![k] = rs!!
            else t!![k] = StatEngine.otLinearMap(srcCov!![k], rc!!)
        }
        // native 参数打包：scale/bias / t32·mu32·ref32 与 StatEngine.mapLabStats /
        // mapOtLinear **逐像素循环里的同一条式子**同式同序（预构建 ⇒ 逐位一致）
        val packed = if (isLab) FloatArray(30) else FloatArray(66)
        packed[0] = if (isLab) 1f else 0f
        for (k in 0 until 4) {
            if (isLab) {
                // StatEngine.mapLabStats：scale 是 double 除法后截 f32，bias 用
                // f32 乘加（不能在 double 域合并算完再转——桌面口径注释同款）
                val base = 1 + k * 6
                for (c in 0 until 3) {
                    packed[base + c] =
                        (refStd!![k][c] / maxOf(srcStd!![k][c], 1e-6)).toFloat()
                }
                for (c in 0 until 3) {
                    packed[base + 3 + c] =
                        refMean[k][c].toFloat() - srcMean[k][c].toFloat() * packed[base + c]
                }
            } else {
                val base = 1 + k * 15
                for (r in 0 until 3) for (c in 0 until 3) {
                    packed[base + r * 3 + c] = t!![k][r][c].toFloat()
                }
                for (c in 0 until 3) packed[base + 9 + c] = srcMean[k][c].toFloat()
                for (c in 0 until 3) packed[base + 12 + c] = refMean[k][c].toFloat()
            }
        }
        val sBase = if (isLab) 25 else 61
        for (k in 0 until 4) packed[sBase + k] = strength[k]
        packed[sBase + 4] = model.global.strength
        return Plan(isLab, t, refMean, refStd, srcMean, srcStd, strength,
                    model.global.strength, packed)
    }

    /** 整图参考实现（对拍/调试用）：一次算完全图，与 [applyBand] 逐位同式。 */
    fun applyWhole(plan: Plan, tiled: RegionEngine.Tiled, src: Mat, mapped: Mat,
                   protect: Float, strength: Float): FloatArray {
        val w = tiled.wSkin.cols()
        val h = tiled.wSkin.rows()
        val n = w * h
        val rgb = FloatArray(n * 3).also { src.get(0, 0, it) }
        val mp = FloatArray(n * 3).also { mapped.get(0, 0, it) }
        fun full(m: Mat): FloatArray = FloatArray(n).also { m.get(0, 0, it) }
        val out = FloatArray(n * 3)
        applyBandKotlin(plan, rgb, mp, full(tiled.wSkin), full(tiled.wHair),
                        full(tiled.wCloth), full(tiled.wBg), protect, strength, out, n)
        return out
    }

    /** 应用段（读权重 Mat 段 + 计划 → 输出段）。 */
    fun applyBand(tiled: RegionEngine.Tiled, plan: Plan,
                  rgb: FloatArray, mapped: FloatArray, y0: Int, rows: Int,
                  protect: Float, strength: Float, out: FloatArray) {
        val n = tiled.wSkin.cols() * rows
        fun readBand(m: Mat): FloatArray {
            val b = FloatArray(n)
            m.get(y0, 0, b)
            return b
        }
        val wSkin = readBand(tiled.wSkin)
        val wHair = readBand(tiled.wHair)
        val wCloth = readBand(tiled.wCloth)
        val wBg = readBand(tiled.wBg)
        // native 优先（2026-10-02 性能轮）：整段一次 JNI（每区映射 + 融合 +
        // 2×LAB 往返 + L 替换全在 C 里），不建 Kotlin 中间大数组。
        if (NativeKernels.useNative) {
            if (NativeKernels.regionStatBand(rgb, mapped, wSkin, wHair, wCloth,
                                             wBg, plan.packed, protect, strength,
                                             out)) {
                return
            }
            NativeKernels.useNative = false
            DebugLog.e("native regionStatBand 失败→回退纯 Kotlin（后续不再尝试）")
        }
        applyBandKotlin(plan, rgb, mapped, wSkin, wHair, wCloth, wBg,
                        protect, strength, out, n)
    }

    /** 纯 Kotlin 段级内核（对拍基准 + 回退路径）。 */
    internal fun applyBandKotlin(plan: Plan, rgb: FloatArray, mapped: FloatArray,
                                 wSkin: FloatArray, wHair: FloatArray,
                                 wCloth: FloatArray, wBg: FloatArray,
                                 protect: Float, strength: Float,
                                 out: FloatArray, n: Int) {
        val n3 = n * 3
        // 每区全强度映射（工作空间 → RGB）：与桌面 `t.apply_region` 里的 `_map` 同式
        val maps = Array(4) { k ->
            if (plan.isLab) StatEngine.mapLabStats(rgb, plan.refMean[k], plan.refStd!![k],
                                                   plan.srcMean[k], plan.srcStd!![k])
            else StatEngine.mapOtLinear(rgb, plan.t!![k], plan.srcMean[k], plan.refMean[k])
        }
        // 嵌套强度混合（`apply_region` 的 `_with_strength`；预设里通常 s=1）
        fun branch(k: Int): FloatArray {
            val s = plan.regionStrength[k]
            if (s >= 1f) return maps[k]
            if (s <= 0f) return rgb
            val mk = maps[k]
            return FloatArray(n3) { rgb[it] * (1f - s) + mk[it] * s }
        }
        // 皮肤分支：protect=1 恒原色（保护不依赖分割成败）；否则原色 ↔ 该区映射插值。
        // 样片无肤区时 maps[0] 已是"回退全局 ref"的映射（buildPlan 统一规则）——
        // 与桌面 `_skin_branch` 2026-09-30 起同规则，protect 滑块不再失效。
        val skinBranch: FloatArray = if (protect >= 0.999f) rgb else {
            val b0 = branch(0)
            FloatArray(n3) { rgb[it] * protect + b0[it] * (1f - protect) }
        }
        // 四区软融合（键序 skin→hair→cloth→bg；float 累加序与桌面一致）
        val branches = arrayOf(skinBranch, branch(1), branch(2), branch(3))
        val ws = arrayOf(wSkin, wHair, wCloth, wBg)
        val fused = FloatArray(n3)
        for (k in 0 until 4) {
            val wr = ws[k]; val br = branches[k]
            for (p in 0 until n) {
                fused[3 * p] += wr[p] * br[3 * p]
                fused[3 * p + 1] += wr[p] * br[3 * p + 1]
                fused[3 * p + 2] += wr[p] * br[3 * p + 2]
            }
        }
        // luma="global"：融合结果只留 a/b，L 取全局映射（皮肤区排除在替换外）
        val globalS = EncoderEngine.withStrength(rgb, mapped, plan.globalStrength)
        val ref = LabConv.rgbToLab(globalS)
        val labOut = LabConv.rgbToLab(fused)
        for (p in 0 until n) {
            val lw = wSkin[p]
            labOut[3 * p] = ref[3 * p] * (1f - lw) + labOut[3 * p] * lw
        }
        // 外层 strength：整段结果 ↔ 原图线性混合（桌面 RegionTransform.apply）
        val res = EncoderEngine.withStrength(rgb, LabConv.labToRgb(labOut), strength)
        System.arraycopy(res, 0, out, 0, n3)
    }
}
