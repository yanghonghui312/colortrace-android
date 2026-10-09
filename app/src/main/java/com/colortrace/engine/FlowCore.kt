package com.colortrace.engine

import kotlin.math.ceil
import kotlin.math.tanh

/**
 * Rectified Flow 传输引擎的 Kotlin 移植（桌面 `src/colortrace/methods/flow_core.py`
 * + `encoder_transform._flat_to_weights`，2026-09-28 P1）。
 *
 * 精度约定：与桌面一致全程 **Double**（float64）；Euler 定步长、`t += dt`
 * 逐步累加（不是 t = from + k*dt——浮点累加顺序参与对拍）。乘加顺序与 numpy
 * 的逐元素语义一致，残差 ~1e-15，远低于任何断言口径。
 *
 * 权重布局（与训练端 unpack 严格一致）：W1(4,h) 行主序 → b1(h) → W2(h,3) → b2(3)。
 */
class FlowWeights(val hidden: Int, W1: DoubleArray, b1: DoubleArray,
                  W2: DoubleArray, b2: DoubleArray) {
    val w1 = W1; val b1 = b1; val w2 = W2; val b2 = b2

    /**
     * 打平成 native 内核（NativeKernels.flowProbes）的单数组布局：
     * W1(4h) | b1(h) | W2(3h) | b2(3)，与 [fromFlat] 的切片序一一对应。
     */
    fun pack(): DoubleArray = DoubleArray(size) {
        when {
            it < 4 * hidden -> w1[it]
            it < 5 * hidden -> b1[it - 4 * hidden]
            it < 8 * hidden -> w2[it - 5 * hidden]
            else -> b2[it - 8 * hidden]
        }
    }

    private val size: Int get() = 8 * hidden + 3

    companion object {
        fun fromFlat(flat: DoubleArray, hidden: Int): FlowWeights {
            val n4 = 4 * hidden
            require(flat.size == n4 + hidden + 3 * hidden + 3) {
                "参数量不符: ${flat.size} != ${n4 + hidden + 3 * hidden + 3} (hidden=$hidden)"
            }
            var i = 0
            val W1 = flat.copyOfRange(i, i + n4); i += n4
            val b1 = flat.copyOfRange(i, i + hidden); i += hidden
            val W2 = flat.copyOfRange(i, i + 3 * hidden); i += 3 * hidden
            val b2 = flat.copyOfRange(i, i + 3)
            return FlowWeights(hidden, W1, b1, W2, b2)
        }
    }
}

/** 速度场 v_θ(x,t)：(RGB,t) → 隐层 tanh → 线性。结果写入 out。 */
private fun velocity(x: DoubleArray, t: Double, w: FlowWeights,
                     hbuf: DoubleArray, out: DoubleArray) {
    val h = w.hidden
    for (j in 0 until h) {
        var s = w.b1[j]
        s += x[0] * w.w1[j]; s += x[1] * w.w1[h + j]; s += x[2] * w.w1[2 * h + j]
        s += t * w.w1[3 * h + j]
        hbuf[j] = tanh(s)
    }
    for (k in 0 until 3) {
        var s = w.b2[k]
        for (j in 0 until h) s += hbuf[j] * w.w2[j * 3 + k]
        out[k] = s
    }
}

/** Euler 定步长积分 dX/dt = v_θ(X,t)，t 从 tFrom 到 tTo（可逆向）。原地修改 x。 */
fun integrate(x: DoubleArray, w: FlowWeights, tFrom: Double, tTo: Double,
              steps: Int): DoubleArray {
    if (steps <= 0 || tTo == tFrom) return x
    val dt = (tTo - tFrom) / steps
    var t = tFrom
    val hbuf = DoubleArray(w.hidden)
    val v = DoubleArray(3)
    for (s in 0 until steps) {
        velocity(x, t, w, hbuf, v)
        x[0] += dt * v[0]; x[1] += dt * v[1]; x[2] += dt * v[2]
        t += dt
    }
    return x
}

/**
 * 色彩迁移前向：F_s^{1→(1−τ)} ∘ F_c^{0→τ}（与桌面 flow_core.transfer 同式）。
 * x 原地修改并返回。steps = 每单位区间的 Euler 步数（encoder 路线用 1）。
 */
fun transfer(x: DoubleArray, wContent: FlowWeights, wStyle: FlowWeights,
             strength: Double = 1.0, steps: Int = 8): DoubleArray {
    val tau = strength.coerceIn(0.0, 1.0)
    if (tau <= 0.0) return x
    val n1 = maxOf(1, ceil(steps * tau).toInt())
    integrate(x, wContent, 0.0, tau, n1)      // 内容分布 → (部分)均匀
    integrate(x, wStyle, 1.0, 1.0 - tau, n1)  // (部分)均匀 → 样片分布
    return x
}
