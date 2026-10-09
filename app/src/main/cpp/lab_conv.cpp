// sRGB ↔ CIELAB 的 native 实现（LabConv.kt 的逐位等价移植）。
//
// 逐位纪律（与 Kotlin / 桌面 numpy 严格同源）：
//  - 全程 float32，表达式结合序与 Kotlin 逐字对应（a*b + c*d + e*f 左结合）；
//  - 超越函数按 Kotlin 语义 **double 计算再截 float**：
//      Kotlin `Float.pow(Float)` = (float)Math.pow((double)this, (double)x)
//      Kotlin `cbrt(Double)`     = Math.cbrt
//    两者在 Android 上都落到 bionic libm，与 NDK 的 std::pow/std::cbrt 同一实现；
//  - 编译期关掉 FMA 收缩与 fast-math（见 CMakeLists），否则 vfma 的单次舍入
//    会与参考的"先乘后加"产生 1 ulp 级漂移。
//
// 常量与 LabConv.kt 的位模式一一对应（含 double 除法后截 float 的写法）。

#include "lab_core.h"
#include "parallel.h"

#include <jni.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>

namespace {

constexpr float kM_RGB2XYZ[9] = {
    0.4124564f, 0.3575761f, 0.1804375f,
    0.2126729f, 0.7151522f, 0.0721750f,
    0.0193339f, 0.1191920f, 0.9503041f};

constexpr float kM_XYZ2RGB[9] = {
    3.240454912185669f, -1.5371389389038086f, -0.4985315799713135f,
    -0.969266414642334f, 1.8760108947753906f, 0.041556086391210556f,
    0.055643416941165924f, -0.20402584969997406f, 1.057225227355957f};

constexpr float kWhite[3] = {0.95047f, 1.0f, 1.08883f};

// Kotlin: (216.0 / 24389.0).toFloat() 等——double 常量运算后截 float
constexpr float kEps = static_cast<float>(216.0 / 24389.0);
constexpr float kKappa = static_cast<float>(24389.0 / 27.0);
constexpr float kInv1292 = static_cast<float>(1.0 / 12.92);
constexpr float kInv1055 = static_cast<float>(1.0 / 1.055);
constexpr float kGamma = 2.4f;
constexpr float kInvGamma = static_cast<float>(1.0 / 2.4);

inline float clamp01(float v) {
    if (v < 0.0f) return 0.0f;
    if (v > 1.0f) return 1.0f;
    return v;
}

// Kotlin: if (c <= 0.04045f) c * INV_12_92 else ((c + 0.055f) * INV_1_055).pow(GAMMA)
inline float srgbToLinear(float c) {
    if (c <= 0.04045f) return c * kInv1292;
    const float base = (c + 0.055f) * kInv1055;
    return static_cast<float>(std::pow(static_cast<double>(base),
                                       static_cast<double>(kGamma)));
}

// Kotlin: cc = max(c,0); hi = 1.055f * cc.pow(INV_GAMMA) - 0.055f;
//         if (cc <= 0.0031308f) 12.92f * cc else hi
inline float linearToSrgb(float c) {
    const float cc = (c < 0.0f) ? 0.0f : c;
    const float hi = 1.055f * static_cast<float>(
        std::pow(static_cast<double>(cc), static_cast<double>(kInvGamma))) - 0.055f;
    return (cc <= 0.0031308f) ? 12.92f * cc : hi;
}

// Kotlin: if (tt > EPS) cbrt(tt) else (KAPPA * tt + 16.0f) * (1.0f / 116.0f)
inline float fCie(float t) {
    const float tt = (t < 0.0f) ? 0.0f : t;
    if (tt > kEps) return static_cast<float>(std::cbrt(static_cast<double>(tt)));
    return (kKappa * tt + 16.0f) * (1.0f / 116.0f);
}

// Kotlin: f3 = f*f*f; if (f3 > EPS) f3 else (116.0f * f - 16.0f) / KAPPA
inline float invF(float f) {
    const float f3 = f * f * f;
    if (f3 > kEps) return f3;
    return (116.0f * f - 16.0f) / kKappa;
}

}  // namespace

// 对外（同一 .so 内共享）：kernels.cpp 的四区融合要复用这两个
namespace ct {

void rgbToLab(const float* rgb, float* out, std::size_t n) {
    for (std::size_t p = 0; p < n; ++p) {
        const float lr = srgbToLinear(clamp01(rgb[3 * p]));
        const float lg = srgbToLinear(clamp01(rgb[3 * p + 1]));
        const float lb = srgbToLinear(clamp01(rgb[3 * p + 2]));
        const float x = lr * kM_RGB2XYZ[0] + lg * kM_RGB2XYZ[1] + lb * kM_RGB2XYZ[2];
        const float y = lr * kM_RGB2XYZ[3] + lg * kM_RGB2XYZ[4] + lb * kM_RGB2XYZ[5];
        const float z = lr * kM_RGB2XYZ[6] + lg * kM_RGB2XYZ[7] + lb * kM_RGB2XYZ[8];
        const float fx = fCie(x / kWhite[0]);
        const float fy = fCie(y / kWhite[1]);
        const float fz = fCie(z / kWhite[2]);
        out[3 * p] = 116.0f * fy - 16.0f;
        out[3 * p + 1] = 500.0f * (fx - fy);
        out[3 * p + 2] = 200.0f * (fy - fz);
    }
}

void labToRgb(const float* lab, float* out, std::size_t n) {
    for (std::size_t p = 0; p < n; ++p) {
        const float L = lab[3 * p];
        const float a = lab[3 * p + 1];
        const float b = lab[3 * p + 2];
        const float fy = (L + 16.0f) * (1.0f / 116.0f);
        const float fx = fy + a * (1.0f / 500.0f);
        const float fz = fy - b * (1.0f / 200.0f);
        const float yr = (L > kKappa * kEps) ? fy * fy * fy : L / kKappa;
        const float x = invF(fx) * kWhite[0];
        const float y = yr * kWhite[1];
        const float z = invF(fz) * kWhite[2];
        for (int c = 0; c < 3; ++c) {
            const float lin = x * kM_XYZ2RGB[3 * c] + y * kM_XYZ2RGB[3 * c + 1] +
                              z * kM_XYZ2RGB[3 * c + 2];
            out[3 * p + c] = clamp01(linearToSrgb(lin));
        }
    }
}

}  // namespace ct

namespace {

// 共用骨架：src → 计算 → dst（个数不符/分配失败返回 null 由 Kotlin 侧兜底）。
// 2026-10-01 性能轮：JNI 入口（chroma 路径每次一整段）按像素区间并行——
// fn 逐像素独立，区间内不改任何浮点运算（逐位不变，见 parallel.h）。
// 注意 ct::rgbToLab/ct::labToRgb 本体保持单线程：它们也被 fuseBandCore 的
// 并行区间内部调用，这里若再并行会嵌套线程爆炸。
jfloatArray run(JNIEnv* env, jfloatArray src,
                void (*fn)(const float*, float*, std::size_t)) {
    if (src == nullptr) return nullptr;
    const jsize len = env->GetArrayLength(src);
    if (len <= 0 || len % 3 != 0) return nullptr;
    jfloatArray dst = env->NewFloatArray(len);
    if (dst == nullptr) return nullptr;

    const std::size_t n = static_cast<std::size_t>(len) / 3;
    float* buf = static_cast<float*>(std::malloc(sizeof(float) * static_cast<std::size_t>(len)));
    if (buf == nullptr) return nullptr;

    jfloat* in = env->GetFloatArrayElements(src, nullptr);
    if (in == nullptr) { std::free(buf); return nullptr; }
    try {
        ct::parallelFor(n, 65536, [&](std::size_t b, std::size_t e) {
            fn(in + 3 * b, buf + 3 * b, e - b);
        });
    } catch (...) {
        std::free(buf);
        env->ReleaseFloatArrayElements(src, in, JNI_ABORT);
        env->DeleteLocalRef(dst);
        return nullptr;
    }
    env->ReleaseFloatArrayElements(src, in, JNI_ABORT);   // 只读，不写回

    env->SetFloatArrayRegion(dst, 0, len, buf);
    std::free(buf);
    return dst;
}

}  // namespace

extern "C" {

JNIEXPORT jstring JNICALL
Java_com_colortrace_engine_NativeLab_nativeVersion(JNIEnv* env, jobject /* this */) {
    return env->NewStringUTF("colortrace-native/1 (lab)");
}

JNIEXPORT jfloatArray JNICALL
Java_com_colortrace_engine_NativeLab_rgbToLab(JNIEnv* env, jobject /* this */,
                                              jfloatArray rgb) {
    return run(env, rgb, ct::rgbToLab);
}

JNIEXPORT jfloatArray JNICALL
Java_com_colortrace_engine_NativeLab_labToRgb(JNIEnv* env, jobject /* this */,
                                              jfloatArray lab) {
    return run(env, lab, ct::labToRgb);
}

}  // extern "C"