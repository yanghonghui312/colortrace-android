package com.colortrace.engine

import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * 手动微调参数层（桌面 `methods/develop.py` 的 Kotlin 移植，P2.6）。
 *
 * 全部逐像素点态（CIELAB 域）：影调组只作用 L 通道、色彩平衡组作用 a/b、
 * 分离色调按 L 高/低权重给 a/b 着色——因此可与自动追色串联（develop 在后）
 * 且天然适配 TiledPipeline 的按段处理。
 *
 * 数值口径（与桌面逐式对应，金标 develop_vectors.json 锁 ≤1/255）：
 * exposure ±5 EV（L 幂变换，+1EV ≈ Ln^0.5）、contrast/高光/阴影/白色/黑色
 * ±100、temp/tint ±100（LAB a/b ±20）、饱和/自然饱和 ±100、色相 ±180°、
 * 分离色调色相 0~360°、饱和 0~100（最大彩度 30）、平衡 ±100（±0.4 分界）、
 * gamma 0.1~3.0（>1 压暗）。
 */
object DevelopSpec {
    data class Param(val key: String, val cn: String, val en: String,
                     val min: Float, val max: Float, val def: Float,
                     val decimals: Int)

    val ALL = listOf(
        Param("exposure", "曝光", "Exposure", -5.0f, 5.0f, 0.0f, 2),
        Param("contrast", "对比度", "Contrast", -100.0f, 100.0f, 0.0f, 0),
        Param("highlights", "高光", "Highlights", -100.0f, 100.0f, 0.0f, 0),
        Param("shadows", "阴影", "Shadows", -100.0f, 100.0f, 0.0f, 0),
        Param("whites", "白色", "Whites", -100.0f, 100.0f, 0.0f, 0),
        Param("blacks", "黑色", "Blacks", -100.0f, 100.0f, 0.0f, 0),
        Param("temp", "色温", "Temp", -100.0f, 100.0f, 0.0f, 0),
        Param("tint", "色调", "Tint", -100.0f, 100.0f, 0.0f, 0),
        Param("saturation", "饱和度", "Saturation", -100.0f, 100.0f, 0.0f, 0),
        Param("vibrance", "自然饱和度", "Vibrance", -100.0f, 100.0f, 0.0f, 0),
        Param("hue_shift", "色相偏移", "Hue", -180.0f, 180.0f, 0.0f, 0),
        Param("split_high_hue", "高光色相", "Highlights Hue", 0.0f, 360.0f, 30.0f, 0),
        Param("split_high_sat", "高光饱和度", "Highlights Sat", 0.0f, 100.0f, 0.0f, 0),
        Param("split_low_hue", "阴影色相", "Shadows Hue", 0.0f, 360.0f, 210.0f, 0),
        Param("split_low_sat", "阴影饱和度", "Shadows Sat", 0.0f, 100.0f, 0.0f, 0),
        Param("split_balance", "平衡", "Balance", -100.0f, 100.0f, 0.0f, 0),
        Param("gamma", "伽马", "Gamma", 0.1f, 3.0f, 1.0f, 2),
    )
    val BY_KEY = ALL.associateBy { it.key }
    val DEFAULTS = ALL.associate { it.key to it.def }

    // 分组（顺序即桌面面板顺序；UI 换页用）
    val GROUPS = listOf(
        "基础影调" to listOf("exposure", "contrast", "highlights",
                             "shadows", "whites", "blacks"),
        "色彩平衡" to listOf("temp", "tint", "saturation", "vibrance", "hue_shift"),
        "分离色调" to listOf("split_high_hue", "split_high_sat",
                             "split_low_hue", "split_low_sat", "split_balance"),
        "其他" to listOf("gamma"),
    )

    fun clamp(values: Map<String, Float>): Map<String, Float> =
        DEFAULTS.toMutableMap().also { out ->
            for ((key, v) in values) {
                val spec = BY_KEY[key] ?: continue
                out[key] = v.coerceIn(spec.min, spec.max)
            }
        }

    fun isIdentity(values: Map<String, Float>): Boolean =
        ALL.all { abs((values[it.key] ?: it.def) - it.def) < 1e-6f }
}

private const val SPLIT_MAX_CHROMA = 30.0f

/** values 为全键 Map（缺键取默认）；数学逐式对应桌面 _map（float32 语义）。 */
class DevelopTransform(values: Map<String, Float> = emptyMap()) {

    companion object {
        /**
         * native 参数字向量的键序——必须与 `kernels.cpp` 的 developMapLab
         * 顶部注释 p[0..16] 逐一对应（顺序写错不会崩，只会算错）。
         */
        val PARAM_VECTOR_KEYS = listOf(
            "exposure", "contrast", "highlights", "shadows", "whites", "blacks",
            "temp", "tint", "saturation", "vibrance", "hue_shift",
            "split_high_hue", "split_high_sat", "split_low_hue", "split_low_sat",
            "split_balance", "gamma")
    }

    private val p = DevelopSpec.clamp(values)

    /** [NativeKernels.developApply] 的 17 个参数（键序 = [PARAM_VECTOR_KEYS]）。 */
    internal val paramVector: FloatArray =
        FloatArray(PARAM_VECTOR_KEYS.size) { p[PARAM_VECTOR_KEYS[it]] ?: 0f }

    fun isIdentity(): Boolean = DevelopSpec.isIdentity(p)

    /**
     * 交错 RGB float32 [0,1] → 追加微调后的 RGB（identity 时原样返回副本）。
     *
     * native 可用时走 [NativeKernels.developApply]（`kernels.cpp` 的 developCore：
     * RGB→LAB→mapLab→RGB 合成一趟 C++ 循环，1 次 JNI 调用、无 Java 中间数组）；
     * 返回 false 或 native 不可用时回退 [applyKotlin]，绝不静默产出错值。
     */
    fun apply(image: FloatArray): FloatArray {
        if (isIdentity()) return image.copyOf()
        if (NativeKernels.useNative) {
            val out = FloatArray(image.size)
            if (NativeKernels.developApply(image, paramVector, out)) return out
        }
        return applyKotlin(image)
    }

    /**
     * 纯 Kotlin 参考实现（native 的对拍基准，也是回退路径）。
     * 刻意用 [LabConv.rgbToLabKotlin] / [LabConv.labToRgbKotlin]——不含任何
     * native 成分，A/B 关卡比较的才是同一件事。
     */
    internal fun applyKotlin(image: FloatArray): FloatArray {
        if (isIdentity()) return image.copyOf()
        return LabConv.labToRgbKotlin(mapLab(LabConv.rgbToLabKotlin(image)))
    }

    /** LAB（交错 L,a,b）→ 微调后的 LAB；纯逐像素，分块管线按段调用。 */
    fun mapLab(lab: FloatArray): FloatArray {
        val out = lab.copyOf()
        val n = out.size / 3

        // 标量门控（与桌面一致：参数为 0 时整段跳过）
        val ev = p["exposure"] ?: 0f
        val contrast = p["contrast"] ?: 0f
        val shadows = p["shadows"] ?: 0f
        val highlights = p["highlights"] ?: 0f
        val whites = p["whites"] ?: 0f
        val blacks = p["blacks"] ?: 0f
        val gamma = p["gamma"] ?: 1f
        val temp = p["temp"] ?: 0f
        val tint = p["tint"] ?: 0f
        val saturation = p["saturation"] ?: 0f
        val vibrance = p["vibrance"] ?: 0f
        val hueShift = p["hue_shift"] ?: 0f
        val splitHighSat = p["split_high_sat"] ?: 0f
        val splitHighHue = p["split_high_hue"] ?: 30f
        val splitLowSat = p["split_low_sat"] ?: 0f
        val splitLowHue = p["split_low_hue"] ?: 210f
        val splitBalance = p["split_balance"] ?: 0f

        val evExp = Math.pow(2.0, -ev.toDouble()).toFloat()
        val conK = 1.0f + (contrast / 100.0f) * 0.8f
        val shAmp = (shadows / 100.0f) * 0.35f
        val hiAmp = (highlights / 100.0f) * 0.35f
        val whAmp = (whites / 100.0f) * 0.18f
        val blAmp = (blacks / 100.0f) * 0.18f
        val bal = (splitBalance / 100.0f) * 0.4f
        var hueCa = 1f; var hueSa = 0f
        if (abs(hueShift) > 1e-9f) {
            val th = Math.toRadians(hueShift.toDouble())
            hueCa = cos(th).toFloat(); hueSa = sin(th).toFloat()
        }
        var cHigh = 0f; var hcCos = 1f; var hcSin = 0f
        if (abs(splitHighSat) > 1e-9f) {
            val th = Math.toRadians(splitHighHue.toDouble())
            cHigh = splitHighSat / 100.0f * SPLIT_MAX_CHROMA
            hcCos = cos(th).toFloat(); hcSin = sin(th).toFloat()
        }
        var cLow = 0f; var lcCos = 1f; var lcSin = 0f
        if (abs(splitLowSat) > 1e-9f) {
            val th = Math.toRadians(splitLowHue.toDouble())
            cLow = splitLowSat / 100.0f * SPLIT_MAX_CHROMA
            lcCos = cos(th).toFloat(); lcSin = sin(th).toFloat()
        }

        for (i in 0 until n) {
            var a = out[3 * i + 1]
            var b = out[3 * i + 2]
            var ln = (out[3 * i] / 100.0f).coerceIn(0.0f, 1.0f)

            // ---- 基础影调（L 通道） ----
            if (abs(ev) > 1e-9f) {
                ln = maxOf(ln, 1e-6f).pow(evExp)
            }
            if (abs(contrast) > 1e-9f) {
                ln = (0.5f + (ln - 0.5f) * conK).coerceIn(0.0f, 1.0f)
            }
            if (abs(shadows) > 1e-9f) {
                val w = 1.0f - smoothstep(0.0f, 0.5f, ln)
                ln = (ln + shAmp * w).coerceIn(0.0f, 1.0f)
            }
            if (abs(highlights) > 1e-9f) {
                val w = smoothstep(0.5f, 1.0f, ln)
                ln = (ln + hiAmp * w).coerceIn(0.0f, 1.0f)
            }
            if (abs(whites) > 1e-9f) {
                ln = (ln + whAmp * (ln * ln)).coerceIn(0.0f, 1.0f)
            }
            if (abs(blacks) > 1e-9f) {
                ln = (ln + blAmp * ((1.0f - ln) * (1.0f - ln))).coerceIn(0.0f, 1.0f)
            }
            if (abs(gamma - 1.0f) > 1e-9f) {
                ln = maxOf(ln, 1e-6f).pow(gamma)
            }

            // ---- 色彩平衡（a/b 通道） ----
            if (abs(temp) > 1e-9f) b += (temp / 100.0f) * 20.0f
            if (abs(tint) > 1e-9f) a += (tint / 100.0f) * 20.0f
            if (abs(saturation) > 1e-9f) {
                val k = 1.0f + saturation / 100.0f
                a *= k; b *= k
            }
            if (abs(vibrance) > 1e-9f) {
                val chroma = sqrt(a * a + b * b)
                val w = 1.0f - (chroma / 60.0f).coerceIn(0.0f, 1.0f)
                val k = 1.0f + (vibrance / 100.0f) * w
                a *= k; b *= k
            }
            if (abs(hueShift) > 1e-9f) {
                val na = a * hueCa - b * hueSa
                b = a * hueSa + b * hueCa
                a = na
            }

            // ---- 分离色调 ----
            if (abs(splitHighSat) > 1e-9f || abs(splitLowSat) > 1e-9f) {
                val x = (ln + bal).coerceIn(0.0f, 1.0f)
                val wHigh = x * x
                val wLow = (1.0f - x) * (1.0f - x)
                if (abs(splitHighSat) > 1e-9f) {
                    a += cHigh * hcCos * wHigh
                    b += cHigh * hcSin * wHigh
                }
                if (abs(splitLowSat) > 1e-9f) {
                    a += cLow * lcCos * wLow
                    b += cLow * lcSin * wLow
                }
            }

            out[3 * i] = (ln * 100.0f).coerceIn(0.0f, 100.0f)
            out[3 * i + 1] = a.coerceIn(-127.0f, 127.0f)
            out[3 * i + 2] = b.coerceIn(-127.0f, 127.0f)
        }
        return out
    }

    private fun smoothstep(e0: Float, e1: Float, x: Float): Float {
        val t = ((x - e0) / maxOf(e1 - e0, 1e-6f)).coerceIn(0.0f, 1.0f)
        return t * t * (3.0f - 2.0f * t)
    }
}
