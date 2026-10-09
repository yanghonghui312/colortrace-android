package com.colortrace.engine

import kotlin.math.abs
import kotlin.math.sqrt

/**
 * 肤色保护（chroma 色彩域锁定）的 Kotlin 移植（桌面
 * `segmentation/skin_mask.skin_weight` + `methods/skin_protect._map_chroma`）。
 *
 * 精度约定：与桌面一致全程 **Float**。固定隶属函数（色相窗口 × 彩度双沿），
 * 不含 skin_match（样片肤色签名，P1+ 范围）与自适应升级（YuNet + 脸部统计，
 * P1+ 范围）——金标生成时桌面侧同样 skin_adaptive=False、skin_match=0。
 */
object SkinWeight {
    private const val HUE_CENTER = 45.0f
    private const val HUE_HALF = 26.0f        // 全权重半宽
    private const val HUE_SOFT = 24.0f        // 过渡带宽（角度）
    private const val C_LO = 6.0f; private const val C_HI = 14.0f
    private const val C_FALL_LO = 32.0f; private const val C_FALL_HI = 46.0f

    /** C1 连续 smoothstep：x<=e0 得 0，x>=e1 得 1（与桌面同式）。 */
    private fun smoothstep(e0: Float, e1: Float, x: Float): Float {
        val t = ((x - e0) / maxOf(e1 - e0, 1e-6f)).coerceIn(0f, 1f)
        return t * t * (3f - 2f * t)
    }

    /** 色相角距离（0~180），环绕安全。⚠ numpy 的 % 恒非负，Kotlin 要 floorMod 语义。 */
    private fun angularDistance(deg: Float, center: Float): Float {
        var m = (deg - center + 180.0f) % 360.0f
        if (m < 0f) m += 360.0f
        return abs(m - 180.0f)
    }

    /** 肤色隶属度 w∈[0,1]，逐像素（LAB 输入，交错 H*W*3）。 */
    fun of(lab: FloatArray): FloatArray {
        val n = lab.size / 3
        val out = FloatArray(n)
        for (p in 0 until n) {
            val a = lab[3 * p + 1]; val b = lab[3 * p + 2]
            val chroma = sqrt(a.toDouble() * a + b.toDouble() * b).toFloat()
            val hue = Math.toDegrees(kotlin.math.atan2(b.toDouble(), a.toDouble())).toFloat()
            val wHue = 1f - smoothstep(HUE_HALF, HUE_HALF + HUE_SOFT,
                                       angularDistance(hue, HUE_CENTER))
            var wC = smoothstep(C_LO, C_HI, chroma)
            wC *= 1f - smoothstep(C_FALL_LO, C_FALL_HI, chroma)
            out[p] = wHue * wC
        }
        return out
    }

    /**
     * chroma 锁定映射（桌面 `_map_chroma`，skin_match=0 分支）：
     * a/b 按 w·protect 锚回原片，L 完全跟随全局追色。全 float32。
     */
    fun anchorChroma(labIn: FloatArray, labOut: FloatArray,
                     w: FloatArray, protect: Float): FloatArray {
        val n = labIn.size / 3
        val out = labOut.copyOf()
        for (p in 0 until n) {
            val we = w[p] * protect
            if (we > 0f) {
                for (c in 1..2) {
                    val o = out[3 * p + c]
                    out[3 * p + c] = o - we * (o - labIn[3 * p + c])
                }
            }
        }
        return out
    }
}
