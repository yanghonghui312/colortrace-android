package com.colortrace.engine

import com.colortrace.DebugLog
import kotlin.math.floor

/**
 * 33³ 三维查找表的 Kotlin 移植（桌面 `src/colortrace/lut.py`，trilinear）。
 *
 * 精度约定：表值与插值全程 **Double**（桌面 `_apply_table` 为 float64），
 * 输入像素 float32 → double，输出裁剪 [0,1] 后取 float32——与桌面
 * `apply_rgb` 的 `np.clip(out,0,1).astype(np.float32)` 同序。
 *
 * 表布局：table[ ((r*size)+g)*size+b )*3 + c ]，按 [R,G,B] 索引（红变化最快，
 * 与 .cube 的 B→G→R 外中内层序对应）。8 角累加顺序与桌面 `_cell_weights`
 * 的 offs 序 (dr,dg,db) 三重循环一致。
 */
class Lut3D(val size: Int, table: DoubleArray) {
    val table: DoubleArray

    init {
        require(table.size == size * size * size * 3) {
            "LUT 表大小不符: ${table.size} != ${size * size * size * 3}"
        }
        // 桌面 LUT3D.__init__ 的 np.clip(t, 0, 1)
        this.table = DoubleArray(table.size) { table[it].coerceIn(0.0, 1.0) }
    }

    /**
     * 查表套用到交错 RGB float32 图像（返回新数组）。
     *
     * native 优先（`kernels.cpp` 的 `lutApply`，逐位等价、double 累加序照搬）；
     * 返回 false 时**永久**切回 [applyKotlin]——与 LabConv 同一降级纪律。
     */
    fun apply(image: FloatArray): FloatArray {
        val out = FloatArray(image.size)
        if (NativeKernels.useNative) {
            if (NativeKernels.lutApply(table, size, image, out)) return out
            NativeKernels.useNative = false
            DebugLog.e("native lutApply 失败→回退纯 Kotlin（后续不再尝试）")
        }
        return applyKotlin(image)
    }

    /** 纯 Kotlin 参考实现——native 的对拍基准，也是回退路径。 */
    internal fun applyKotlin(image: FloatArray): FloatArray {
        val n = image.size / 3
        val out = FloatArray(image.size)
        val last = size - 1
        for (p in 0 until n) {
            val g0 = image[3 * p].toDouble().coerceIn(0.0, 1.0) * last
            val g1 = image[3 * p + 1].toDouble().coerceIn(0.0, 1.0) * last
            val g2 = image[3 * p + 2].toDouble().coerceIn(0.0, 1.0) * last
            var i0 = floor(g0).toInt(); var i1 = floor(g1).toInt(); var i2 = floor(g2).toInt()
            if (i0 > last - 1) i0 = last - 1
            if (i1 > last - 1) i1 = last - 1
            if (i2 > last - 1) i2 = last - 1
            val f0 = g0 - i0; val f1 = g1 - i1; val f2 = g2 - i2
            var acc0 = 0.0; var acc1 = 0.0; var acc2 = 0.0
            for (dr in 0..1) for (dg in 0..1) for (db in 0..1) {
                val wr = if (dr == 1) f0 else 1.0 - f0
                val wg = if (dg == 1) f1 else 1.0 - f1
                val wb = if (db == 1) f2 else 1.0 - f2
                val wk = wr * wg * wb
                val idx = (((i0 + dr) * size + (i1 + dg)) * size + (i2 + db)) * 3
                acc0 += wk * table[idx]; acc1 += wk * table[idx + 1]; acc2 += wk * table[idx + 2]
            }
            out[3 * p] = acc0.coerceIn(0.0, 1.0).toFloat()
            out[3 * p + 1] = acc1.coerceIn(0.0, 1.0).toFloat()
            out[3 * p + 2] = acc2.coerceIn(0.0, 1.0).toFloat()
        }
        return out
    }
}
