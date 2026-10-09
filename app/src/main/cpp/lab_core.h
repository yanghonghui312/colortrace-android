// sRGB ↔ CIELAB 的共享声明：定义在 lab_conv.cpp，kernels.cpp 的四区融合也要用。
//
// 逐位纪律（与 Kotlin / 桌面 numpy 同源）：float32、同序乘加、超越函数 double 再截
// float、编译期禁 FMA 收缩与 fast-math——详见 lab_conv.cpp 顶部注释。
#pragma once

#include <cstddef>

namespace ct {

/** RGB [0,1] 交错 → LAB（L∈[0,100]，a/b≈[-127,127]）。 */
void rgbToLab(const float* rgb, float* out, std::size_t n);

/** LAB → RGB [0,1]（越界裁剪）。 */
void labToRgb(const float* lab, float* out, std::size_t n);

}  // namespace ct