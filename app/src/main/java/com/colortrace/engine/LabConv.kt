package com.colortrace.engine

import com.colortrace.DebugLog
import kotlin.math.abs
import kotlin.math.cbrt
import kotlin.math.pow

/**
 * sRGB ↔ CIELAB 的 Kotlin 移植（桌面 `src/colortrace/io.py`，Lindbloom/D65）。
 *
 * 精度约定：与桌面一致全程 **Float**（float32）——桌面刻意用 float32（20MP 内存
 * 带宽），矩阵/常量与桌面同源（XYZ→RGB 是 numpy float64 求逆后取 float32 的
 * 硬编码值）。幂/立方根走 double 再取 float，与 numpy 的 float32 powf 差
 * ~1 ulp，远低于 u8 对拍口径。
 *
 * 图像布局：FloatArray 交错 RGB（H*W*3），L∈[0,100]，a/b≈[-127,127]。
 */
object LabConv {
    private val M_RGB2XYZ = floatArrayOf(
        0.4124564f, 0.3575761f, 0.1804375f,
        0.2126729f, 0.7151522f, 0.0721750f,
        0.0193339f, 0.1191920f, 0.9503041f)
    // np.linalg.inv(RGB2XYZ as f64).astype(f32) 的逐字节同源值
    private val M_XYZ2RGB = floatArrayOf(
        3.240454912185669f, -1.5371389389038086f, -0.4985315799713135f,
        -0.969266414642334f, 1.8760108947753906f, 0.041556086391210556f,
        0.055643416941165924f, -0.20402584969997406f, 1.057225227355957f)
    private val WHITE = floatArrayOf(0.95047f, 1.0f, 1.08883f)
    private const val EPS = (216.0 / 24389.0).toFloat()
    private const val KAPPA = (24389.0 / 27.0).toFloat()
    private const val INV_12_92 = (1.0 / 12.92).toFloat()
    private const val INV_1_055 = (1.0 / 1.055).toFloat()
    private const val GAMMA = 2.4f
    private const val INV_GAMMA = (1.0 / 2.4).toFloat()

    private fun srgbToLinear(c: Float): Float =
        if (c <= 0.04045f) c * INV_12_92
        else ((c + 0.055f) * INV_1_055).pow(GAMMA)

    private fun linearToSrgb(c: Float): Float {
        val cc = if (c < 0f) 0f else c
        val hi = 1.055f * cc.pow(INV_GAMMA) - 0.055f
        return if (cc <= 0.0031308f) 12.92f * cc else hi
    }

    private fun fCie(t: Float): Float {
        val tt = if (t < 0f) 0f else t
        return if (tt > EPS) cbrt(tt.toDouble()).toFloat()
        else (KAPPA * tt + 16.0f) * (1.0f / 116.0f)
    }

    /**
     * 是否走 native（默认 [NativeLab.available]）。
     * 仅供回退与对拍开关使用——P1Test 用它逐位比较 native 与 Kotlin 两条路径。
     */
    @Volatile
    var useNative: Boolean = NativeLab.available

    /**
     * RGB [0,1] → LAB（新数组）。
     *
     * native 可用时走 [NativeLab]（`src/main/cpp/lab_conv.cpp`，逐位等价）；
     * native 返回 null（分配失败/参数异常）时**永久切回** [rgbToLabKotlin]。
     */
    fun rgbToLab(rgb: FloatArray): FloatArray {
        if (useNative) {
            val out = NativeLab.rgbToLab(rgb)
            if (out != null) return out
            useNative = false
            DebugLog.e("native rgbToLab 返回 null→回退纯 Kotlin（后续不再尝试）")
        }
        return rgbToLabKotlin(rgb)
    }

    /** 纯 Kotlin 参考实现——native 的对拍基准，也是回退路径。 */
    internal fun rgbToLabKotlin(rgb: FloatArray): FloatArray {
        val n = rgb.size / 3
        val out = FloatArray(n * 3)
        for (p in 0 until n) {
            // 逐像素不再分配临时数组（原 FloatArray(3) 在 ART 下是明显热点）；
            // 数值与累加序不变 ⇒ 逐位一致。
            val lr = srgbToLinear(rgb[3 * p].coerceIn(0f, 1f))
            val lg = srgbToLinear(rgb[3 * p + 1].coerceIn(0f, 1f))
            val lb = srgbToLinear(rgb[3 * p + 2].coerceIn(0f, 1f))
            val x = lr * M_RGB2XYZ[0] + lg * M_RGB2XYZ[1] + lb * M_RGB2XYZ[2]
            val y = lr * M_RGB2XYZ[3] + lg * M_RGB2XYZ[4] + lb * M_RGB2XYZ[5]
            val z = lr * M_RGB2XYZ[6] + lg * M_RGB2XYZ[7] + lb * M_RGB2XYZ[8]
            val fx = fCie(x / WHITE[0]); val fy = fCie(y / WHITE[1]); val fz = fCie(z / WHITE[2])
            out[3 * p] = 116.0f * fy - 16.0f
            out[3 * p + 1] = 500.0f * (fx - fy)
            out[3 * p + 2] = 200.0f * (fy - fz)
        }
        return out
    }

    private fun invF(f: Float): Float {
        val f3 = f * f * f
        return if (f3 > EPS) f3 else (116.0f * f - 16.0f) / KAPPA
    }

    /** LAB → RGB [0,1]（越界裁剪，新数组）。native 优先，见 [rgbToLab]。 */
    fun labToRgb(lab: FloatArray): FloatArray {
        if (useNative) {
            val out = NativeLab.labToRgb(lab)
            if (out != null) return out
            useNative = false
            DebugLog.e("native labToRgb 返回 null→回退纯 Kotlin（后续不再尝试）")
        }
        return labToRgbKotlin(lab)
    }

    /** 纯 Kotlin 参考实现——native 的对拍基准，也是回退路径。 */
    internal fun labToRgbKotlin(lab: FloatArray): FloatArray {
        val n = lab.size / 3
        val out = FloatArray(n * 3)
        for (p in 0 until n) {
            val L = lab[3 * p]; val a = lab[3 * p + 1]; val b = lab[3 * p + 2]
            val fy = (L + 16.0f) * (1.0f / 116.0f)
            val fx = fy + a * (1.0f / 500.0f)
            val fz = fy - b * (1.0f / 200.0f)

            val yr = if (L > KAPPA * EPS) fy * fy * fy else L / KAPPA
            val x = invF(fx) * WHITE[0]; val y = yr * WHITE[1]; val z = invF(fz) * WHITE[2]
            for (c in 0 until 3) {
                val lin = x * M_XYZ2RGB[3 * c] + y * M_XYZ2RGB[3 * c + 1] +
                        z * M_XYZ2RGB[3 * c + 2]
                var v = linearToSrgb(lin)
                if (v < 0f) v = 0f else if (v > 1f) v = 1f
                out[3 * p + c] = v
            }
        }
        return out
    }

    /** 逐点色差 ΔE76（测试辅助）。 */
    fun de76(a: FloatArray, b: FloatArray): Float {
        var s = 0.0
        for (i in a.indices) {
            val d = (a[i] - b[i]).toDouble()
            s += d * d
        }
        return kotlin.math.sqrt(s / (a.size / 3.0)).toFloat()
    }

    fun maxAbsDiff(a: FloatArray, b: FloatArray): Float {
        var m = 0f
        for (i in a.indices) m = maxOf(m, abs(a[i] - b[i]))
        return m
    }
}
