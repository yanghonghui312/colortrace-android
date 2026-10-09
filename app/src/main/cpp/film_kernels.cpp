// 胶片质感层（桌面 methods/film.py 的安卓移植，2026-10-02）逐像素算子。
//
// 与桌面的口径关系（用户拍板）：
//  - 柔光/光晕是纯确定性算法：与桌面**同式同序**（float32、-ffp-contract=off），
//    u8 出图按 ≤1/255 金标协议对拍（luma 点积的 1ulp 平台差可容忍，同 region 档）；
//  - 颗粒是随机纹理：**不与桌面逐位**（两端各自确定性，算法/观感同构）——
//    粗网格高斯噪声上采样 + 封顶细噪声 + 全图 std 归一（幅度 = σ·数量），
//    RNG 用端内自有的 splitmix64 + Box-Muller（同图同参逐位可复现的端内承诺
//    仍成立）。种子由 Kotlin 侧派生后传入。
//
// 高斯模糊与 base 上采样不放这里（Kotlin 层直接用 OpenCV Imgproc——与桌面
// 同一 cv2 实现家族，逐位概率最高）；本文件只做逐像素部分，因此不依赖
// OpenCV C++ 头。

#include "lab_core.h"
#include "parallel.h"

#include <jni.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

// 桌面 film._LUMA（Rec709 亮度）
constexpr float kLumaR = 0.2126f;
constexpr float kLumaG = 0.7152f;
constexpr float kLumaB = 0.0722f;
// 桌面 film._HALATION_WARM：色调 100 时的辉光颜色（G/B 压低、R 保持）
constexpr float kWarmG = 0.42f;
constexpr float kWarmB = 0.16f;

// Kotlin Float.coerceIn：NaN 原样返回（两个比较都 false）
inline float coerceIn(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

// 桌面 film._smoothstep（e0<e1）：t = clip((x−e0)/max(e1−e0,1e-6)); t²(3−2t)
inline float smoothstepF(float e0, float e1, float x) {
    const float span = (e1 - e0 > 1e-6f) ? (e1 - e0) : 1e-6f;
    const float t = coerceIn((x - e0) / span, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// 桌面 film._screen：1−(1−a)(1−b)（定义在 [0,1]，float32 同序）
inline float screenF(float a, float b) {
    return 1.0f - (1.0f - a) * (1.0f - b);
}

// ---- 端内确定性 RNG（splitmix64 + Box-Muller；**不**追求与 numpy 逐位——
// 用户拍板颗粒两端各自随机、算法同构即可；端内同参可复现由确定性序列保证）----
inline std::uint64_t splitmix64(std::uint64_t& s) {
    s += 0x9E3779B97F4A7C15ull;
    std::uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// (0,1) 开区间均匀（53 位精度，同 java/ng 的常见构造）
inline double nextUniform(std::uint64_t& s) {
    return static_cast<double>(splitmix64(s) >> 11) * (1.0 / 9007199254740992.0);
}

void grainNormalCore(std::uint64_t seed, std::size_t n, float* out) {
    std::uint64_t s = seed;
    // 预热若干步（避免低熵种子的首相关）
    for (int i = 0; i < 4; ++i) splitmix64(s);
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        // Box-Muller：u1∈(0,1] 防log(0)；两输出一循环
        const double u1 = nextUniform(s);
        const double u2 = nextUniform(s);
        const double r = std::sqrt(-2.0 * std::log(u1));
        const double th = 6.283185307179586 * u2;
        out[i] = static_cast<float>(r * std::cos(th));
        out[i + 1] = static_cast<float>(r * std::sin(th));
    }
    if (i < n) {
        const double u1 = nextUniform(s);
        const double u2 = nextUniform(s);
        const double r = std::sqrt(-2.0 * std::log(u1));
        out[i] = static_cast<float>(r * std::cos(6.283185307179586 * u2));
    }
}

// 柔光逐段：out = rgb + (screen(rgb,blurred) − rgb) · (soft·m)，
// m = smoothstep(t0, t0+band, luma(rgb))。与桌面 _map 柔光分支同式。
void filmSoftBandCore(const float* rgb, const float* blurred, float soft,
                      float t0, float band, float* out, std::size_t n) {
    ct::parallelFor(n, 32768, [&](std::size_t b, std::size_t e) {
        for (std::size_t p = b; p < e; ++p) {
            const std::size_t i = 3 * p;
            const float luma = rgb[i] * kLumaR + rgb[i + 1] * kLumaG +
                               rgb[i + 2] * kLumaB;
            const float k = soft * smoothstepF(t0, t0 + band, luma);
            for (int c = 0; c < 3; ++c) {
                out[i + c] = rgb[i + c] +
                             (screenF(rgb[i + c], blurred[i + c]) - rgb[i + c]) * k;
            }
        }
    });
}

// 光晕·高光提取：mask = smoothstep(0.7, 1.0, luma(rgb))；masked = rgb·mask
// （masked 进高斯模糊，与桌面 `out * mask` 同式）。
void filmBloomMaskCore(const float* rgb, float* masked, float* mask,
                       std::size_t n) {
    ct::parallelFor(n, 32768, [&](std::size_t b, std::size_t e) {
        for (std::size_t p = b; p < e; ++p) {
            const std::size_t i = 3 * p;
            const float luma = rgb[i] * kLumaR + rgb[i + 1] * kLumaG +
                               rgb[i + 2] * kLumaB;
            const float m = smoothstepF(0.7f, 1.0f, luma);
            mask[p] = m;
            masked[i] = rgb[i] * m;
            masked[i + 1] = rgb[i + 1] * m;
            masked[i + 2] = rgb[i + 2] * m;
        }
    });
}

// 光晕·混合：tint>0 时 glow 先乘 (1+(WARM−1)·tint)（逐通道，f32），
// out = rgb + (screen(rgb,glow) − rgb)·bloom。与桌面同式。
void filmBloomApplyCore(const float* rgb, const float* glow, float bloom,
                        float tint, float* out, std::size_t n) {
    const float fg = 1.0f + (kWarmG - 1.0f) * tint;
    const float fb = 1.0f + (kWarmB - 1.0f) * tint;
    ct::parallelFor(n, 32768, [&](std::size_t b, std::size_t e) {
        for (std::size_t p = b; p < e; ++p) {
            const std::size_t i = 3 * p;
            const float g1 = glow[i] * 1.0f;
            const float g2 = glow[i + 1] * fg;
            const float g3 = glow[i + 2] * fb;
            out[i] = rgb[i] + (screenF(rgb[i], g1) - rgb[i]) * bloom;
            out[i + 1] = rgb[i + 1] + (screenF(rgb[i + 1], g2) - rgb[i + 1]) * bloom;
            out[i + 2] = rgb[i + 2] + (screenF(rgb[i + 2], g3) - rgb[i + 2]) * bloom;
        }
    });
}

// 颗粒·全图 std 归一：scale = σ·amount / std(noise)（double 顺序累加；
// 端内口径——不与桌面 numpy 归约对位）。std≈0 时跳过（base 全零等病态输入）。
void filmGrainNormalizeCore(float* noise, std::size_t n, float sigmaAmount) {
    double s = 0.0;
    double sq = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(noise[i]);
        s += x;
        sq += x * x;
    }
    const double mean = s / static_cast<double>(n);
    const double var = sq / static_cast<double>(n) - mean * mean;
    const double std = (var > 0.0) ? std::sqrt(var) : 0.0;
    if (std > 1e-6) {
        const float k = static_cast<float>(sigmaAmount / std);
        ct::parallelFor(n, 65536, [&](std::size_t b, std::size_t e) {
            for (std::size_t i = b; i < e; ++i) noise[i] *= k;
        });
    }
}

// 颗粒·段应用：LAB 往返只动 L（darktable 口径）——lab = rgbToLab(rgb)，
// L = clip(L + noise, 0, 100)，out = labToRgb(lab)。LAB 是点态 ⇒ 段处理与
// 整图逐位一致（ct:: 转换与金标同源）。
void filmGrainBandCore(const float* rgb, const float* noise, float* out,
                       std::size_t n) {
    ct::parallelFor(n, 32768, [&](std::size_t b, std::size_t e) {
        const std::size_t m = e - b;
        std::vector<float> buf(m * 3);
        float* lab = buf.data();
        ct::rgbToLab(rgb + 3 * b, lab, m);
        for (std::size_t p = 0; p < m; ++p) {
            float l = lab[3 * p] + noise[b + p];
            if (l < 0.0f) l = 0.0f;
            if (l > 100.0f) l = 100.0f;
            lab[3 * p] = l;
        }
        ct::labToRgb(lab, out + 3 * b, m);
    });
}

// 细噪声条带并行版：每条带独立子流（seed+si）⇒ 条带间无序列依赖，条带并行
// 后与 Kotlin 串行逐条带生成**逐位一致**（端内确定性不变，噪声相位不变）。
// 直接把 fine·strip 累加进 noise 全图数组（省条带 JNI 往返）。
void grainBandsCore(std::uint64_t seed, std::size_t bands,
                    std::size_t rowsPerBand, std::size_t w, std::size_t h,
                    float fine, float* noise) {
    ct::parallelFor(bands, 1, [&](std::size_t b, std::size_t e) {
        std::vector<float> strip;
        for (std::size_t bi = b; bi < e; ++bi) {
            const std::size_t off = bi * rowsPerBand;
            const std::size_t rows = std::min(rowsPerBand, h - off);
            const std::size_t n = rows * w;
            strip.resize(n);
            grainNormalCore(seed + bi, n, strip.data());
            float* dst = noise + off * w;
            for (std::size_t q = 0; q < n; ++q) dst[q] += fine * strip[q];
        }
    });
}

}  // namespace

extern "C" {

JNIEXPORT void JNICALL
Java_com_colortrace_engine_NativeKernels_filmSoftBand(
        JNIEnv* env, jobject /*this*/, jfloatArray jrgb, jfloatArray jblurred,
        jfloat soft, jfloat t0, jfloat band, jfloatArray jout) {
    if (jrgb == nullptr || jblurred == nullptr || jout == nullptr) return;
    const jsize len = env->GetArrayLength(jrgb);
    if (len <= 0 || len % 3 != 0) return;
    if (env->GetArrayLength(jblurred) != len || env->GetArrayLength(jout) != len) return;
    jfloat* rgb = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(jrgb, nullptr));
    if (rgb == nullptr) return;
    jfloat* blurred = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(jblurred, nullptr));
    if (blurred == nullptr) {
        env->ReleasePrimitiveArrayCritical(jrgb, rgb, JNI_ABORT);
        return;
    }
    jfloat* out = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(jout, nullptr));
    if (out == nullptr) {
        env->ReleasePrimitiveArrayCritical(jrgb, rgb, JNI_ABORT);
        env->ReleasePrimitiveArrayCritical(jblurred, blurred, JNI_ABORT);
        return;
    }
    filmSoftBandCore(rgb, blurred, soft, t0, band, out,
                     static_cast<std::size_t>(len) / 3);
    env->ReleasePrimitiveArrayCritical(jrgb, rgb, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(jblurred, blurred, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(jout, out, 0);
}

JNIEXPORT void JNICALL
Java_com_colortrace_engine_NativeKernels_filmBloomMask(
        JNIEnv* env, jobject /*this*/, jfloatArray jrgb,
        jfloatArray jmasked, jfloatArray jmask) {
    if (jrgb == nullptr || jmasked == nullptr || jmask == nullptr) return;
    const jsize len = env->GetArrayLength(jrgb);
    if (len <= 0 || len % 3 != 0) return;
    if (env->GetArrayLength(jmasked) != len ||
        env->GetArrayLength(jmask) != len / 3) return;
    jfloat* rgb = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(jrgb, nullptr));
    if (rgb == nullptr) return;
    jfloat* masked = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(jmasked, nullptr));
    if (masked == nullptr) {
        env->ReleasePrimitiveArrayCritical(jrgb, rgb, JNI_ABORT);
        return;
    }
    jfloat* mask = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(jmask, nullptr));
    if (mask == nullptr) {
        env->ReleasePrimitiveArrayCritical(jrgb, rgb, JNI_ABORT);
        env->ReleasePrimitiveArrayCritical(jmasked, masked, JNI_ABORT);
        return;
    }
    filmBloomMaskCore(rgb, masked, mask, static_cast<std::size_t>(len) / 3);
    env->ReleasePrimitiveArrayCritical(jrgb, rgb, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(jmasked, masked, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(jmask, mask, 0);
}

JNIEXPORT void JNICALL
Java_com_colortrace_engine_NativeKernels_filmBloomApply(
        JNIEnv* env, jobject /*this*/, jfloatArray jrgb, jfloatArray jglow,
        jfloat bloom, jfloat tint, jfloatArray jout) {
    if (jrgb == nullptr || jglow == nullptr || jout == nullptr) return;
    const jsize len = env->GetArrayLength(jrgb);
    if (len <= 0 || len % 3 != 0) return;
    if (env->GetArrayLength(jglow) != len || env->GetArrayLength(jout) != len) return;
    jfloat* rgb = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(jrgb, nullptr));
    if (rgb == nullptr) return;
    jfloat* glow = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(jglow, nullptr));
    if (glow == nullptr) {
        env->ReleasePrimitiveArrayCritical(jrgb, rgb, JNI_ABORT);
        return;
    }
    jfloat* out = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(jout, nullptr));
    if (out == nullptr) {
        env->ReleasePrimitiveArrayCritical(jrgb, rgb, JNI_ABORT);
        env->ReleasePrimitiveArrayCritical(jglow, glow, JNI_ABORT);
        return;
    }
    filmBloomApplyCore(rgb, glow, bloom, tint, out,
                       static_cast<std::size_t>(len) / 3);
    env->ReleasePrimitiveArrayCritical(jrgb, rgb, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(jglow, glow, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(jout, out, 0);
}

JNIEXPORT void JNICALL
Java_com_colortrace_engine_NativeKernels_filmGrainBase(
        JNIEnv* env, jobject /*this*/, jlong seed, jint n, jfloatArray jout) {
    if (jout == nullptr || n <= 0) return;
    if (env->GetArrayLength(jout) < n) return;
    jfloat* out = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(jout, nullptr));
    if (out == nullptr) return;
    grainNormalCore(static_cast<std::uint64_t>(seed),
                    static_cast<std::size_t>(n), out);
    env->ReleasePrimitiveArrayCritical(jout, out, 0);
}

JNIEXPORT void JNICALL
Java_com_colortrace_engine_NativeKernels_filmGrainNormalize(
        JNIEnv* env, jobject /*this*/, jfloatArray jnoise, jfloat sigmaAmount) {
    if (jnoise == nullptr) return;
    const jsize len = env->GetArrayLength(jnoise);
    if (len <= 0) return;
    jfloat* noise = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(jnoise, nullptr));
    if (noise == nullptr) return;
    filmGrainNormalizeCore(noise, static_cast<std::size_t>(len), sigmaAmount);
    env->ReleasePrimitiveArrayCritical(jnoise, noise, 0);
}

JNIEXPORT void JNICALL
Java_com_colortrace_engine_NativeKernels_filmGrainBands(
        JNIEnv* env, jobject /*this*/, jlong seed, jint bands, jint rowsPerBand,
        jint w, jint h, jfloat fine, jfloatArray jnoise) {
    if (jnoise == nullptr || bands <= 0 || rowsPerBand <= 0 || w <= 0 || h <= 0)
        return;
    if (env->GetArrayLength(jnoise) != w * h) return;
    jfloat* noise = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(jnoise, nullptr));
    if (noise == nullptr) return;
    grainBandsCore(static_cast<std::uint64_t>(seed),
                   static_cast<std::size_t>(bands),
                   static_cast<std::size_t>(rowsPerBand),
                   static_cast<std::size_t>(w), static_cast<std::size_t>(h),
                   fine, noise);
    env->ReleasePrimitiveArrayCritical(jnoise, noise, 0);
}

JNIEXPORT void JNICALL
Java_com_colortrace_engine_NativeKernels_filmGrainBand(
        JNIEnv* env, jobject /*this*/, jfloatArray jrgb, jfloatArray jnoise,
        jfloatArray jout) {
    if (jrgb == nullptr || jnoise == nullptr || jout == nullptr) return;
    const jsize len = env->GetArrayLength(jrgb);
    if (len <= 0 || len % 3 != 0) return;
    if (env->GetArrayLength(jnoise) != len / 3 ||
        env->GetArrayLength(jout) != len) return;
    jfloat* rgb = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(jrgb, nullptr));
    if (rgb == nullptr) return;
    jfloat* noise = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(jnoise, nullptr));
    if (noise == nullptr) {
        env->ReleasePrimitiveArrayCritical(jrgb, rgb, JNI_ABORT);
        return;
    }
    jfloat* out = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(jout, nullptr));
    if (out == nullptr) {
        env->ReleasePrimitiveArrayCritical(jrgb, rgb, JNI_ABORT);
        env->ReleasePrimitiveArrayCritical(jnoise, noise, JNI_ABORT);
        return;
    }
    filmGrainBandCore(rgb, noise, out, static_cast<std::size_t>(len) / 3);
    env->ReleasePrimitiveArrayCritical(jrgb, rgb, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(jnoise, noise, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(jout, out, 0);
}

// ---- Mat 数据指针版（生产路径，2026-10-03）----
//
// 修复：段缓冲与颗粒噪声原先走 Java 堆数组（PreciseBufs / FloatArray(w·h)），
// 5011×3341 全尺寸导出在 256MB 堆上限下必然 OOM →「胶片质感层失败→整层跳过」
// 静默出无效果图（真机 2026-10-02 三连复现）。改为 Kotlin 侧 `Mat.dataAddr()`
// 直传数据指针，(y0, rows) 行窗在 native 侧按连续内存偏移——中间数据全程留在
// native，Java 堆零大分配；顺带省掉段 get/put 的 JNI 数组拷贝。
//
// 核心与数组版**共用**（逐位同算子同序，数组版继续作金标基准）；行窗 Mat
// （rowRange 视图 / 全图）均连续，故偏移 = 行号 × 行元素数。就地写回安全：
// 逐像素核先读后写、parallelFor 块间区域不相交。

JNIEXPORT void JNICALL
Java_com_colortrace_engine_NativeKernels_filmSoftBandMat(
        JNIEnv* /*env*/, jobject /*this*/, jlong matAddr, jint y0,
        jlong blurredAddr, jint a, jint rows, jint w,
        jfloat soft, jfloat t0, jfloat band) {
    if (matAddr == 0L || blurredAddr == 0L || rows <= 0 || w <= 0) return;
    const std::size_t rowElems = static_cast<std::size_t>(w) * 3;
    float* mat = reinterpret_cast<float*>(matAddr) +
                 static_cast<std::size_t>(y0) * rowElems;
    const float* blurred = reinterpret_cast<const float*>(blurredAddr) +
                           static_cast<std::size_t>(a) * rowElems;
    filmSoftBandCore(mat, blurred, soft, t0, band, mat,
                     static_cast<std::size_t>(rows) * w);
}

JNIEXPORT void JNICALL
Java_com_colortrace_engine_NativeKernels_filmBloomMaskMat(
        JNIEnv* /*env*/, jobject /*this*/, jlong matAddr, jint yA,
        jlong maskedAddr, jlong maskAddr, jint rowsSub, jint w) {
    if (matAddr == 0L || maskedAddr == 0L || maskAddr == 0L ||
        rowsSub <= 0 || w <= 0) return;
    const std::size_t rowElems = static_cast<std::size_t>(w) * 3;
    filmBloomMaskCore(
            reinterpret_cast<const float*>(matAddr) +
                    static_cast<std::size_t>(yA) * rowElems,
            reinterpret_cast<float*>(maskedAddr),
            reinterpret_cast<float*>(maskAddr),
            static_cast<std::size_t>(rowsSub) * w);
}

JNIEXPORT void JNICALL
Java_com_colortrace_engine_NativeKernels_filmBloomApplyMat(
        JNIEnv* /*env*/, jobject /*this*/, jlong matAddr, jint y0,
        jlong glowAddr, jint a, jint rows, jint w,
        jfloat bloom, jfloat tint) {
    if (matAddr == 0L || glowAddr == 0L || rows <= 0 || w <= 0) return;
    const std::size_t rowElems = static_cast<std::size_t>(w) * 3;
    float* mat = reinterpret_cast<float*>(matAddr) +
                 static_cast<std::size_t>(y0) * rowElems;
    const float* glow = reinterpret_cast<const float*>(glowAddr) +
                        static_cast<std::size_t>(a) * rowElems;
    filmBloomApplyCore(mat, glow, bloom, tint, mat,
                       static_cast<std::size_t>(rows) * w);
}

JNIEXPORT void JNICALL
Java_com_colortrace_engine_NativeKernels_filmGrainBaseMat(
        JNIEnv* /*env*/, jobject /*this*/, jlong seed, jlong baseAddr, jint n) {
    if (baseAddr == 0L || n <= 0) return;
    grainNormalCore(static_cast<std::uint64_t>(seed),
                    static_cast<std::size_t>(n),
                    reinterpret_cast<float*>(baseAddr));
}

JNIEXPORT void JNICALL
Java_com_colortrace_engine_NativeKernels_filmGrainScaleMat(
        JNIEnv* /*env*/, jobject /*this*/, jlong noiseAddr, jint n, jfloat inv) {
    if (noiseAddr == 0L || n <= 0) return;
    float* noise = reinterpret_cast<float*>(noiseAddr);
    // 逐元素乘法与顺序无关 ⇒ 与 Kotlin 串行 *= 逐位一致
    ct::parallelFor(static_cast<std::size_t>(n), 65536,
                    [&](std::size_t b, std::size_t e) {
                        for (std::size_t i = b; i < e; ++i) noise[i] *= inv;
                    });
}

JNIEXPORT void JNICALL
Java_com_colortrace_engine_NativeKernels_filmGrainBandsMat(
        JNIEnv* /*env*/, jobject /*this*/, jlong seed, jint bands,
        jint rowsPerBand, jint w, jint h, jfloat fine, jlong noiseAddr) {
    if (noiseAddr == 0L || bands <= 0 || rowsPerBand <= 0 || w <= 0 || h <= 0)
        return;
    grainBandsCore(static_cast<std::uint64_t>(seed),
                   static_cast<std::size_t>(bands),
                   static_cast<std::size_t>(rowsPerBand),
                   static_cast<std::size_t>(w), static_cast<std::size_t>(h),
                   fine, reinterpret_cast<float*>(noiseAddr));
}

JNIEXPORT void JNICALL
Java_com_colortrace_engine_NativeKernels_filmGrainNormalizeMat(
        JNIEnv* /*env*/, jobject /*this*/, jlong noiseAddr, jint n,
        jfloat sigmaAmount) {
    if (noiseAddr == 0L || n <= 0) return;
    filmGrainNormalizeCore(reinterpret_cast<float*>(noiseAddr),
                           static_cast<std::size_t>(n), sigmaAmount);
}

JNIEXPORT void JNICALL
Java_com_colortrace_engine_NativeKernels_filmGrainBandMat(
        JNIEnv* /*env*/, jobject /*this*/, jlong matAddr, jint y0,
        jlong noiseAddr, jint rows, jint w) {
    if (matAddr == 0L || noiseAddr == 0L || rows <= 0 || w <= 0) return;
    const std::size_t rowElems = static_cast<std::size_t>(w) * 3;
    float* mat = reinterpret_cast<float*>(matAddr) +
                 static_cast<std::size_t>(y0) * rowElems;
    const float* noise = reinterpret_cast<const float*>(noiseAddr) +
                         static_cast<std::size_t>(y0) * w;
    filmGrainBandCore(mat, noise, mat, static_cast<std::size_t>(rows) * w);
}

}  // extern "C"
