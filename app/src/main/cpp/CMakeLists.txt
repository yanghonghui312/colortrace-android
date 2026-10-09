cmake_minimum_required(VERSION 3.22.1)
project(colortrace CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(colortrace SHARED
    lab_conv.cpp
    kernels.cpp
    film_kernels.cpp)

# ---- 逐位纪律 ----
# 参考实现是 float32 且刻意对齐 numpy：pow/cbrt 走 double 再截 float、乘加不重排。
# 因此这里必须关掉 FMA 收缩与 fast-math，否则舍入会漂移、0/255 逐位关卡直接红。
target_compile_options(colortrace PRIVATE
    -O3
    -ffp-contract=off
    -fno-fast-math
    -fvisibility=hidden)

# Android 15+ 要求所有打包 .so 满足 16KB page size（调研文档 §6 风险清单第 2 条）
target_link_options(colortrace PRIVATE "-Wl,-z,max-page-size=16384")