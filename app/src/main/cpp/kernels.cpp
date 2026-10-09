// 逐像素热循环第二批：33³ LUT 三线性查表 + 四区融合整段。
//
// 逐位纪律与 lab_conv.cpp 完全一致（float32 同序乘加 / 禁 FMA / double 语义照搬）。
// 两个函数都**只写调用方给的 out**，不新建 Java 数组——省掉 Kotlin 侧每段的
// 若干个大 FloatArray 分配（原来 applyBandMapped 每段要建 5~6 个）。
//
// 边界约定：out 必须与所有输入数组不重叠（调用方保证，Kotlin 侧本来也是分开的）。

#include "lab_core.h"
#include "parallel.h"

#include <jni.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

inline double clamp01d(double v) {
    if (v < 0.0) return 0.0;
    if (v > 1.0) return 1.0;
    return v;
}

// ---------------------------------------------------------------------------
// Lut3D.apply（Lut3D.kt）：double 三线性，8 角累加序 (dr,dg,db) 原样保留。
// ---------------------------------------------------------------------------
void lutApplyCore(const double* table, int size, const float* rgb, float* out,
                  std::size_t n) {
    const int last = size - 1;
    const double dLast = static_cast<double>(last);
    // 像素间独立 ⇒ 按区间并行，单像素内的 8 角累加序原样（逐位不变，见 parallel.h）
    ct::parallelFor(n, 32768, [&](std::size_t b, std::size_t e) {
    for (std::size_t p = b; p < e; ++p) {
        // Kotlin: image[3p].toDouble().coerceIn(0.0,1.0) * last
        const double g0 = clamp01d(static_cast<double>(rgb[3 * p])) * dLast;
        const double g1 = clamp01d(static_cast<double>(rgb[3 * p + 1])) * dLast;
        const double g2 = clamp01d(static_cast<double>(rgb[3 * p + 2])) * dLast;
        int i0 = static_cast<int>(std::floor(g0));
        int i1 = static_cast<int>(std::floor(g1));
        int i2 = static_cast<int>(std::floor(g2));
        if (i0 > last - 1) i0 = last - 1;
        if (i1 > last - 1) i1 = last - 1;
        if (i2 > last - 1) i2 = last - 1;
        const double f0 = g0 - static_cast<double>(i0);
        const double f1 = g1 - static_cast<double>(i1);
        const double f2 = g2 - static_cast<double>(i2);

        double acc0 = 0.0;
        double acc1 = 0.0;
        double acc2 = 0.0;
        for (int dr = 0; dr < 2; ++dr) {
            for (int dg = 0; dg < 2; ++dg) {
                for (int db = 0; db < 2; ++db) {
                    const double wr = dr ? f0 : 1.0 - f0;
                    const double wg = dg ? f1 : 1.0 - f1;
                    const double wb = db ? f2 : 1.0 - f2;
                    const double wk = wr * wg * wb;
                    const std::size_t idx =
                        ((static_cast<std::size_t>(i0 + dr) * size + (i1 + dg)) * size +
                         (i2 + db)) * 3;
                    acc0 += wk * table[idx];
                    acc1 += wk * table[idx + 1];
                    acc2 += wk * table[idx + 2];
                }
            }
        }
        out[3 * p] = static_cast<float>(clamp01d(acc0));
        out[3 * p + 1] = static_cast<float>(clamp01d(acc1));
        out[3 * p + 2] = static_cast<float>(clamp01d(acc2));
    }
    });
}

// 段级复用缓冲（同一线程内复用容量，避免每段 malloc；量级 4×段像素×3 float）
std::vector<float>& scratch() {
    static thread_local std::vector<float> buf;
    return buf;
}

// ---------------------------------------------------------------------------
// develop（手动微调）整段：RGB → LAB → mapLab → RGB 一趟走完。
//
// 对应 DevelopTransform.apply（Kotlin）：原先要 2 次 JNI 往返（rgbToLab/labToRgb）
// 加 Kotlin 的 mapLab 标量循环，每次滑块停顿约 0.8s（真机 V2324A 实测）。
// 这里把三步合成一个 C++ 循环，只做 1 次 JNI 调用、不建任何 Java 中间数组。
//
// 逐位纪律与 lab_conv.cpp 相同：float32、同序乘加、超越函数按 Kotlin 语义
// double 计算再截 float、-ffp-contract=off。因此与 Kotlin 参考实现逐位一致
// （NativeKernelsTest.developIsBitExact 锁）。
//
// 参数向量 p（17 个，顺序 = 下面注释；DevelopTransform.paramVector 构造）：
//   [0]曝光 [1]对比度 [2]高光 [3]阴影 [4]白色 [5]黑色 [6]色温 [7]色调
//   [8]饱和度 [9]自然饱和度 [10]色相偏移
//   [11]高光色相 [12]高光饱和度 [13]阴影色相 [14]阴影饱和度 [15]平衡 [16]伽马
// ---------------------------------------------------------------------------
namespace {

constexpr float kSplitMaxChroma = 30.0f;

// Kotlin Float.coerceIn：NaN 原样返回（底下两个比较都 false）
inline float coerceIn(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

// Kotlin: t = ((x - e0) / maxOf(e1 - e0, 1e-6f)).coerceIn(0,1); t*t*(3-2t)
inline float smoothstepF(float e0, float e1, float x) {
    const float span = (e1 - e0 > 1e-6f) ? (e1 - e0) : 1e-6f;
    const float t = coerceIn((x - e0) / span, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// Kotlin: Math.toRadians(deg) = deg / 180.0 * Math.PI（double）
inline double toRadians(double deg) { return (deg / 180.0) * 3.141592653589793; }

// Kotlin Float.pow(Float) = (float)Math.pow((double)a,(double)b)；
// Float.sqrt() 同理由 double 计算再截 float（bionic 与 NDK 同一 libm）
inline float powF(float a, float b) {
    return static_cast<float>(std::pow(static_cast<double>(a), static_cast<double>(b)));
}
inline float sqrtF(float v) {
    return static_cast<float>(std::sqrt(static_cast<double>(v)));
}

void developMapLab(float* lab, const float* p, std::size_t n) {
    const float ev = p[0];
    const float contrast = p[1];
    const float highlights = p[2];
    const float shadows = p[3];
    const float whites = p[4];
    const float blacks = p[5];
    const float temp = p[6];
    const float tint = p[7];
    const float saturation = p[8];
    const float vibrance = p[9];
    const float hueShift = p[10];
    const float splitHighHue = p[11];
    const float splitHighSat = p[12];
    const float splitLowHue = p[13];
    const float splitLowSat = p[14];
    const float splitBalance = p[15];
    const float gamma = p[16];

    // ---- 标量门控（与 Kotlin 一致：参数为 0 时整段跳过）----
    const float evExp = static_cast<float>(std::pow(2.0, -static_cast<double>(ev)));
    const float conK = 1.0f + (contrast / 100.0f) * 0.8f;
    const float shAmp = (shadows / 100.0f) * 0.35f;
    const float hiAmp = (highlights / 100.0f) * 0.35f;
    const float whAmp = (whites / 100.0f) * 0.18f;
    const float blAmp = (blacks / 100.0f) * 0.18f;
    const float bal = (splitBalance / 100.0f) * 0.4f;

    float hueCa = 1.0f;
    float hueSa = 0.0f;
    if (std::abs(hueShift) > 1e-9f) {
        const double th = toRadians(static_cast<double>(hueShift));
        hueCa = static_cast<float>(std::cos(th));
        hueSa = static_cast<float>(std::sin(th));
    }
    float cHigh = 0.0f;
    float hcCos = 1.0f;
    float hcSin = 0.0f;
    if (std::abs(splitHighSat) > 1e-9f) {
        const double th = toRadians(static_cast<double>(splitHighHue));
        cHigh = splitHighSat / 100.0f * kSplitMaxChroma;
        hcCos = static_cast<float>(std::cos(th));
        hcSin = static_cast<float>(std::sin(th));
    }
    float cLow = 0.0f;
    float lcCos = 1.0f;
    float lcSin = 0.0f;
    if (std::abs(splitLowSat) > 1e-9f) {
        const double th = toRadians(static_cast<double>(splitLowHue));
        cLow = splitLowSat / 100.0f * kSplitMaxChroma;
        lcCos = static_cast<float>(std::cos(th));
        lcSin = static_cast<float>(std::sin(th));
    }

    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t k = 3 * i;
        float a = lab[k + 1];
        float b = lab[k + 2];
        float ln = coerceIn(lab[k] / 100.0f, 0.0f, 1.0f);

        // ---- 基础影调（L 通道）----
        if (std::abs(ev) > 1e-9f) {
            ln = powF(std::max(ln, 1e-6f), evExp);
        }
        if (std::abs(contrast) > 1e-9f) {
            ln = coerceIn(0.5f + (ln - 0.5f) * conK, 0.0f, 1.0f);
        }
        if (std::abs(shadows) > 1e-9f) {
            const float w = 1.0f - smoothstepF(0.0f, 0.5f, ln);
            ln = coerceIn(ln + shAmp * w, 0.0f, 1.0f);
        }
        if (std::abs(highlights) > 1e-9f) {
            const float w = smoothstepF(0.5f, 1.0f, ln);
            ln = coerceIn(ln + hiAmp * w, 0.0f, 1.0f);
        }
        if (std::abs(whites) > 1e-9f) {
            ln = coerceIn(ln + whAmp * (ln * ln), 0.0f, 1.0f);
        }
        if (std::abs(blacks) > 1e-9f) {
            ln = coerceIn(ln + blAmp * ((1.0f - ln) * (1.0f - ln)), 0.0f, 1.0f);
        }
        if (std::abs(gamma - 1.0f) > 1e-9f) {
            ln = powF(std::max(ln, 1e-6f), gamma);
        }

        // ---- 色彩平衡（a/b 通道）----
        if (std::abs(temp) > 1e-9f) b += (temp / 100.0f) * 20.0f;
        if (std::abs(tint) > 1e-9f) a += (tint / 100.0f) * 20.0f;
        if (std::abs(saturation) > 1e-9f) {
            const float kk = 1.0f + saturation / 100.0f;
            a *= kk;
            b *= kk;
        }
        if (std::abs(vibrance) > 1e-9f) {
            const float chroma = sqrtF(a * a + b * b);
            const float w = 1.0f - coerceIn(chroma / 60.0f, 0.0f, 1.0f);
            const float kk = 1.0f + (vibrance / 100.0f) * w;
            a *= kk;
            b *= kk;
        }
        if (std::abs(hueShift) > 1e-9f) {
            const float na = a * hueCa - b * hueSa;
            b = a * hueSa + b * hueCa;
            a = na;
        }

        // ---- 分离色调 ----
        if (std::abs(splitHighSat) > 1e-9f || std::abs(splitLowSat) > 1e-9f) {
            const float x = coerceIn(ln + bal, 0.0f, 1.0f);
            const float wHigh = x * x;
            const float wLow = (1.0f - x) * (1.0f - x);
            if (std::abs(splitHighSat) > 1e-9f) {
                a += cHigh * hcCos * wHigh;
                b += cHigh * hcSin * wHigh;
            }
            if (std::abs(splitLowSat) > 1e-9f) {
                a += cLow * lcCos * wLow;
                b += cLow * lcSin * wLow;
            }
        }

        lab[k] = coerceIn(ln * 100.0f, 0.0f, 100.0f);
        lab[k + 1] = coerceIn(a, -127.0f, 127.0f);
        lab[k + 2] = coerceIn(b, -127.0f, 127.0f);
    }
}

void developRange(const float* rgb, const float* p, float* out,
                  std::size_t begin, std::size_t end) {
    const std::size_t n = end - begin;
    std::vector<float>& buf = scratch();
    buf.resize(n * 3);
    float* lab = buf.data();
    ct::rgbToLab(rgb + 3 * begin, lab, n);
    developMapLab(lab, p, n);
    ct::labToRgb(lab, out + 3 * begin, n);
}

void developCore(const float* rgb, const float* p, float* out, std::size_t n) {
    ct::parallelFor(n, 32768, [&](std::size_t b, std::size_t e) {
        developRange(rgb, p, out, b, e);
    });
}

}  // namespace

// ---------------------------------------------------------------------------
// 出图段 float RGB [0,1] → u8 BGR 字节流（P2.12 流式 JPEG 直编用）。
// 逐位口径与 TiledPipeline.render / f32ToBitmap 的 u8 式完全一致：
//   u8(v) = (int)(coerce(v,0,1) * 255f + 0.5f)
// float 乘加不收缩（-ffp-contract=off），NaN 归 0（对齐 JVM toInt(NaN)=0）。
// 字节序 = BGR，等价旧链路 Bitmap→RGBA→cvtColor(RGBA2BGR) 的最终字节
// （alpha 恒 0xFF，不参与）。
// ---------------------------------------------------------------------------
inline std::uint8_t u8Channel(float v) {
    float x;
    if (v >= 0.0f && v <= 1.0f) {
        x = v;
    } else if (v > 1.0f) {
        x = 1.0f;
    } else {
        x = 0.0f;                       // 负数与 NaN（NaN 比较全 false）
    }
    return static_cast<std::uint8_t>(static_cast<int>(x * 255.0f + 0.5f));
}

void rgbToBgrU8Core(const float* rgb, std::uint8_t* bgr, std::size_t n) {
    ct::parallelFor(n, 65536, [&](std::size_t b, std::size_t e) {
        for (std::size_t p = b; p < e; ++p) {
            const std::size_t i = 3 * p;
            bgr[i] = u8Channel(rgb[i + 2]);
            bgr[i + 1] = u8Channel(rgb[i + 1]);
            bgr[i + 2] = u8Channel(rgb[i]);
        }
    });
}

// ---------------------------------------------------------------------------
// Bitmap 段 ARGB int → RGB float32 [0,1]（+ 可选 RGB u8）——migrate 源段的
// Kotlin 逐像素循环 native 化（2026-10-01 性能轮：5.5MP 的 src 段 1762ms 里
// 主要是这两个循环）。逐位口径与 TiledPipeline.migrate 的 Kotlin 循环一致：
//   fb[3p+c] = (通道 and 0xFF) / 255f   —— float **除法**（不是乘 1/255f，舍入不同）
//   bb[3p+c] = (通道 and 0xFF)          —— RGB 序（非 BGR）；alpha 不参与
// Kotlin 的 shr 对负 Int 是算术移位，但 and 0xFF 只取低 8 位 ⇒ 与这里的
// uint32 逻辑移位结果相同。
// ---------------------------------------------------------------------------
void argbToRgbF32Core(const std::uint32_t* argb, std::size_t n,
                      float* rgb, std::uint8_t* rgbU8 /*可空*/) {
    ct::parallelFor(n, 65536, [&](std::size_t b, std::size_t e) {
        for (std::size_t p = b; p < e; ++p) {
            const std::uint32_t v = argb[p];
            const std::uint32_t r = (v >> 16) & 0xFFu;
            const std::uint32_t g = (v >> 8) & 0xFFu;
            const std::uint32_t c = v & 0xFFu;
            rgb[3 * p] = static_cast<float>(r) / 255.0f;
            rgb[3 * p + 1] = static_cast<float>(g) / 255.0f;
            rgb[3 * p + 2] = static_cast<float>(c) / 255.0f;
            if (rgbU8 != nullptr) {
                rgbU8[3 * p] = static_cast<std::uint8_t>(r);
                rgbU8[3 * p + 1] = static_cast<std::uint8_t>(g);
                rgbU8[3 * p + 2] = static_cast<std::uint8_t>(c);
            }
        }
    });
}

// ---------------------------------------------------------------------------
// EncoderEngine.withStrength 的 native 版（OFF 档出图段）：原 Kotlin 循环每段
// 多一次大数组分配 + 一次 arraycopy。逐位口径：out[i] = rgb[i]*(1f-s)+mapped[i]*s
// （inv 预先算好——(1f-s) 是同一个 float 值，逐位不变）。s<=0/=1 由 Kotlin 侧
// arraycopy 快路径处理，这里只管 (0,1) 开区间。
// ---------------------------------------------------------------------------
void strengthMixCore(const float* rgb, const float* mapped, float s,
                     float* out, std::size_t n) {
    ct::parallelFor(n, 32768, [&](std::size_t b, std::size_t e) {
        const float inv = 1.0f - s;
        for (std::size_t i = 3 * b; i < 3 * e; ++i) {
            out[i] = rgb[i] * inv + mapped[i] * s;
        }
    });
}

// ---------------------------------------------------------------------------
// EncoderEngine.lutFor 的 33³ 探针循环 native 化（2026-10-02 性能轮）：
// 每探针 transfer = F_s^{1→1−τ} ∘ F_c^{0→τ}（FlowCore.kt 的 velocity/integrate
// 同式同序）。Kotlin 线程并行此循环实测负优化（P2.22，疑功耗墙/GC），正解 =
// native：探针间独立 ⇒ parallelFor；double 全程、tanh 走同一 libm（P1 已证
// ART Math.tanh ↔ 桌面 numpy 在金标输入上逐位，此处对拍锁 native ↔ Kotlin）。
// 权重打包布局（FlowWeights.pack，行主序）：W1(4h) | b1(h) | W2(3h) | b2(3)。
// ---------------------------------------------------------------------------
inline void flowVelocity(const double* x, double t, const double* w1,
                         const double* b1, const double* w2, const double* b2,
                         int h, double* hbuf, double* out) {
    for (int j = 0; j < h; ++j) {
        double s = b1[j];
        s += x[0] * w1[j];
        s += x[1] * w1[h + j];
        s += x[2] * w1[2 * h + j];
        s += t * w1[3 * h + j];
        hbuf[j] = std::tanh(s);
    }
    for (int k = 0; k < 3; ++k) {
        double s = b2[k];
        for (int j = 0; j < h; ++j) s += hbuf[j] * w2[j * 3 + k];
        out[k] = s;
    }
}

inline void flowIntegrate(double* x, const double* w1, const double* b1,
                          const double* w2, const double* b2, int h,
                          double tFrom, double tTo, int steps,
                          double* hbuf, double* v) {
    if (steps <= 0 || tTo == tFrom) return;
    const double dt = (tTo - tFrom) / steps;
    double t = tFrom;
    for (int s = 0; s < steps; ++s) {
        flowVelocity(x, t, w1, b1, w2, b2, h, hbuf, v);
        x[0] += dt * v[0]; x[1] += dt * v[1]; x[2] += dt * v[2];
        t += dt;                            // Kotlin：t += dt 逐步累加（非乘法）
    }
}

// Kotlin transfer(x, wContent, wStyle, strength, steps) 的单探针等价。
inline void flowTransferOne(double* x, const double* wc, const double* ws,
                            int h, double strength, int steps,
                            double* hbuf, double* v) {
    double tau = strength;                  // Kotlin coerceIn：NaN 两个比较都 false
    if (tau < 0.0) tau = 0.0;
    if (tau > 1.0) tau = 1.0;
    if (tau <= 0.0) return;
    const int n1 = std::max(1, static_cast<int>(std::ceil(steps * tau)));
    flowIntegrate(x, wc, wc + 4 * h, wc + 5 * h, wc + 8 * h, h,
                  0.0, tau, n1, hbuf, v);
    flowIntegrate(x, ws, ws + 4 * h, ws + 5 * h, ws + 8 * h, h,
                  1.0, 1.0 - tau, n1, hbuf, v);
}

void flowProbesCore(double* probes, std::size_t nProbes, const double* wc,
                    const double* ws, int hidden, double strength, int steps) {
    ct::parallelFor(nProbes, 8192, [&](std::size_t b, std::size_t e) {
        std::vector<double> hbuf(static_cast<std::size_t>(hidden));
        std::vector<double> v(3);
        for (std::size_t i = b; i < e; ++i) {
            flowTransferOne(probes + 3 * i, wc, ws, hidden, strength, steps,
                            hbuf.data(), v.data());
        }
    });
}

// ---------------------------------------------------------------------------
// RegionStatEngine.applyBandKotlin（RegionEngine.kt）的整段等价实现（2026-10-02
// 性能轮）：每区映射（lab=LAB 仿射 / ot=RGB 仿射）→ 嵌套强度 → 皮肤分支 →
// 四区融合 → L 替换（global）→ 外层 strength。
//
// 与 fuseBand 的两点语义差异（都照 Kotlin 原样，勿"优化"掉）：
//   1. protect<=0 **不提前返回**——皮肤分支 = rgb*0 + branch0*1 = 全量映射，
//      融合照做（Kotlin 无该早退）；
//   2. protect>=0.999 时皮肤分支恒原图，branch0 的映射**不参与**（native 跳过
//      计算，数值无影响）。
// 参数打包 prm（Plan.packed，buildPlan 构建一次）：[0]=isLab；lab 路径每区
// scale(3)+bias(3)（float32，与 StatEngine.mapLabStats 内部同式同序预算），
// ot 路径每区 t(9 行主序)+mu(3)+ref(3)（mapOtLinear 同口径 f32）；尾接
// regionStrength(4) + globalStrength。
// ---------------------------------------------------------------------------
void regionStatBandRange(const float* rgb, const float* mapped,
                         const float* wSkin, const float* wHair,
                         const float* wCloth, const float* wBg,
                         const float* prm, float protect, float strength,
                         float* out, std::size_t begin, std::size_t end) {
    const std::size_t n = end - begin;
    const std::size_t n3 = n * 3;
    rgb += 3 * begin; mapped += 3 * begin; out += 3 * begin;
    wSkin += begin; wHair += begin; wCloth += begin; wBg += begin;

    const int isLab = prm[0] != 0.0f;
    const float* rp = prm + 1;                            // 4 区映射参数
    const float* regS = isLab ? prm + 25 : prm + 61;      // regionStrength[4]
    const float globalStrength = regS[4];

    std::vector<float>& buf = scratch();
    buf.resize(n3 * 7);          // [globalS][labIn][labTmp][mbuf][fused][labRef][labOut]
    float* pGlobal = buf.data();
    float* labIn = pGlobal + n3;
    float* labTmp = labIn + n3;
    float* mbuf = labTmp + n3;
    float* fused = mbuf + n3;
    float* labRef = fused + n3;
    float* labOut = labRef + n3;

    // globalS（Kotlin 在融合后调 withStrength——与顺序无关，提前算同值）
    const float* globalS;
    if (globalStrength <= 0.0f) {
        globalS = rgb;
    } else if (globalStrength >= 1.0f) {
        globalS = mapped;
    } else {
        const float s = globalStrength;
        const float inv = 1.0f - s;
        for (std::size_t i = 0; i < n3; ++i) {
            pGlobal[i] = rgb[i] * inv + mapped[i] * s;
        }
        globalS = pGlobal;
    }

    // lab 路径的输入 LAB 一次算完（Kotlin 的 mapLabStats 每区各转一次，结果相同）
    if (isLab) ct::rgbToLab(rgb, labIn, n);

    // fused = Σ_k w_k · branch_k：Kotlin 是 k 外层 / p 内层累加，对固定像素的
    // 加法顺序同为 skin→hair→cloth→bg；逐区算映射+branch 再累加，序不变。
    std::memset(fused, 0, n3 * sizeof(float));
    for (int k = 0; k < 4; ++k) {
        const float* wk = (k == 0) ? wSkin : (k == 1) ? wHair
                        : (k == 2) ? wCloth : wBg;
        const float s = regS[k];

        // maps[k] → mbuf
        if (isLab) {
            if (!(k == 0 && protect >= 0.999f)) {     // Kotlin：此分支不算 maps[0]
                const float* sc = rp + k * 6;
                const float* bi = sc + 3;
                for (std::size_t p = 0; p < n; ++p) {
                    const std::size_t i = 3 * p;
                    float l = labIn[i] * sc[0] + bi[0];
                    float a = labIn[i + 1] * sc[1] + bi[1];
                    float bb = labIn[i + 2] * sc[2] + bi[2];
                    if (l < 0.0f) l = 0.0f; else if (l > 100.0f) l = 100.0f;
                    if (a < -127.0f) a = -127.0f; else if (a > 127.0f) a = 127.0f;
                    if (bb < -127.0f) bb = -127.0f; else if (bb > 127.0f) bb = 127.0f;
                    labTmp[i] = l; labTmp[i + 1] = a; labTmp[i + 2] = bb;
                }
                ct::labToRgb(labTmp, mbuf, n);
            }
        } else {
            if (!(k == 0 && protect >= 0.999f)) {
                const float* t = rp + k * 15;         // 行主序 (3,3)
                const float* mu = t + 9;
                const float* ref = mu + 3;
                for (std::size_t p = 0; p < n; ++p) {
                    const std::size_t i = 3 * p;
                    const float x0 = rgb[i] - mu[0];
                    const float x1 = rgb[i + 1] - mu[1];
                    const float x2 = rgb[i + 2] - mu[2];
                    float r = t[0] * x0 + t[1] * x1 + t[2] * x2 + ref[0];
                    float g = t[3] * x0 + t[4] * x1 + t[5] * x2 + ref[1];
                    float bb = t[6] * x0 + t[7] * x1 + t[8] * x2 + ref[2];
                    if (r < 0.0f) r = 0.0f; else if (r > 1.0f) r = 1.0f;
                    if (g < 0.0f) g = 0.0f; else if (g > 1.0f) g = 1.0f;
                    if (bb < 0.0f) bb = 0.0f; else if (bb > 1.0f) bb = 1.0f;
                    mbuf[i] = r; mbuf[i + 1] = g; mbuf[i + 2] = bb;
                }
            }
        }

        // branch 值 → 原地定在 mbuf（或指向 rgb），随后累加
        const float* br;
        if (k == 0) {
            if (protect >= 0.999f) {
                br = rgb;                             // Kotlin：skinBranch 恒原图
            } else {
                if (s >= 1.0f) {
                    // mbuf 原样
                } else if (s <= 0.0f) {
                    std::memcpy(mbuf, rgb, n3 * sizeof(float));
                } else {
                    const float inv = 1.0f - s;
                    for (std::size_t i = 0; i < n3; ++i) {
                        mbuf[i] = rgb[i] * inv + mbuf[i] * s;
                    }
                }
                const float invP = 1.0f - protect;
                for (std::size_t i = 0; i < n3; ++i) {
                    mbuf[i] = rgb[i] * protect + mbuf[i] * invP;
                }
                br = mbuf;
            }
        } else {
            if (s >= 1.0f) {
                br = mbuf;
            } else if (s <= 0.0f) {
                br = rgb;
            } else {
                const float inv = 1.0f - s;
                for (std::size_t i = 0; i < n3; ++i) {
                    mbuf[i] = rgb[i] * inv + mbuf[i] * s;
                }
                br = mbuf;
            }
        }
        for (std::size_t p = 0; p < n; ++p) {
            const std::size_t i = 3 * p;
            fused[i] += wk[p] * br[i];
            fused[i + 1] += wk[p] * br[i + 1];
            fused[i + 2] += wk[p] * br[i + 2];
        }
    }

    // 亮度策略 global：融合结果只留 a/b，L 取全局映射（皮肤区排除在替换外）
    ct::rgbToLab(globalS, labRef, n);
    ct::rgbToLab(fused, labOut, n);
    for (std::size_t p = 0; p < n; ++p) {
        const float lw = wSkin[p];
        const std::size_t i = 3 * p;
        labOut[i] = labRef[i] * (1.0f - lw) + labOut[i] * lw;
    }
    ct::labToRgb(labOut, labRef, n);                    // res → 复用 labRef 缓冲

    // 外层 strength：整段结果 ↔ 原图线性混合（Kotlin withStrength 三分支）
    if (strength <= 0.0f) {
        std::memcpy(out, rgb, n3 * sizeof(float));
    } else if (strength >= 1.0f) {
        std::memcpy(out, labRef, n3 * sizeof(float));
    } else {
        const float inv = 1.0f - strength;
        for (std::size_t i = 0; i < n3; ++i) {
            out[i] = rgb[i] * inv + labRef[i] * strength;
        }
    }
}

void regionStatBandCore(const float* rgb, const float* mapped,
                        const float* wSkin, const float* wHair,
                        const float* wCloth, const float* wBg,
                        const float* prm, float protect, float strength,
                        float* out, std::size_t n) {
    ct::parallelFor(n, 32768, [&](std::size_t b, std::size_t e) {
        regionStatBandRange(rgb, mapped, wSkin, wHair, wCloth, wBg,
                            prm, protect, strength, out, b, e);
    });
}

// ---------------------------------------------------------------------------
// chroma 锚定整段（2026-10-02 性能轮）：SkinWeight.of（肤色隶属度：sqrt+atan2
// 的 double 语义照 Kotlin——double 计算再截 float）+ anchorChroma（a/b 按
// w·protect 锚回原片）+ LAB 往返合成一趟。
// 调用约定照 TiledPipeline.applyBandCore 的 chroma 分支：**无 protect<=0 早退**
// （protect=0 输出 = labToRgb∘rgbToLab(globalS) 的往返值，Kotlin 同）——调用方
// 有早退语义的（applyChroma / EngineRepository.chromaAnchor / fallbackChroma）
// 自行在 protect>0 时才进来。
// ---------------------------------------------------------------------------
inline float angularDistanceF(float deg, float center) {
    float m = std::fmod((deg - center) + 180.0f, 360.0f);
    if (m < 0.0f) m += 360.0f;
    return std::fabs(m - 180.0f);
}

void chromaAnchorRange(const float* rgb, const float* globalOut, float protect,
                       float* out, std::size_t begin, std::size_t end) {
    const std::size_t n = end - begin;
    const std::size_t n3 = n * 3;
    std::vector<float>& buf = scratch();
    buf.resize(n3 * 2 + n);          // [labIn][labOut][w]
    float* labIn = buf.data();
    float* labOut = labIn + n3;
    float* w = labOut + n3;

    ct::rgbToLab(rgb + 3 * begin, labIn, n);
    ct::rgbToLab(globalOut + 3 * begin, labOut, n);

    // SkinWeight.of：HUE_CENTER=45 / HUE_HALF=26 / HUE_SOFT=24 / C_LO=6 / C_HI=14
    // / C_FALL_LO=32 / C_FALL_HI=46（SkinWeight.kt 常量）。
    for (std::size_t p = 0; p < n; ++p) {
        const float a = labIn[3 * p + 1];
        const float b = labIn[3 * p + 2];
        const float chroma = static_cast<float>(
            std::sqrt(static_cast<double>(a) * a + static_cast<double>(b) * b));
        const double hueD = std::atan2(static_cast<double>(b),
                                       static_cast<double>(a));
        const float hue = static_cast<float>((hueD * 180.0) / 3.141592653589793);
        const float wHue = 1.0f - smoothstepF(26.0f, 50.0f,
                                              angularDistanceF(hue, 45.0f));
        float wC = smoothstepF(6.0f, 14.0f, chroma);
        wC *= 1.0f - smoothstepF(32.0f, 46.0f, chroma);
        w[p] = wHue * wC;
    }

    // anchorChroma（skin_match=0 分支）：out=labOut 原地锚定 a/b
    for (std::size_t p = 0; p < n; ++p) {
        const float we = w[p] * protect;
        if (we > 0.0f) {
            for (int c = 1; c <= 2; ++c) {
                const float o = labOut[3 * p + c];
                labOut[3 * p + c] = o - we * (o - labIn[3 * p + c]);
            }
        }
    }
    ct::labToRgb(labOut, out + 3 * begin, n);
}

void chromaAnchorCore(const float* rgb, const float* globalOut, float protect,
                      float* out, std::size_t n) {
    ct::parallelFor(n, 32768, [&](std::size_t b, std::size_t e) {
        chromaAnchorRange(rgb, globalOut, protect, out, b, e);
    });
}

// ---------------------------------------------------------------------------
// RegionEngine.applyBandMapped（RegionEngine.kt）的整段等价实现：
//   withStrength → 四区软融合（skin→hair→cloth→bg）→ L 替换（global 亮度策略）
//   → labToRgb
// ---------------------------------------------------------------------------
void fuseBandRange(const float* rgb, const float* mapped, const float* wSkin,
                   const float* wHair, const float* wCloth, const float* wBg,
                   float protect, float strength, float* out,
                   std::size_t begin, std::size_t end) {
    const std::size_t n = end - begin;
    const std::size_t n3 = n * 3;
    rgb += 3 * begin; mapped += 3 * begin; out += 3 * begin;
    wSkin += begin; wHair += begin; wCloth += begin; wBg += begin;

    std::vector<float>& buf = scratch();    // thread_local ⇒ 并行区间互不踩
    buf.resize(n3 * 4);                 // [globalS][fused][labRef][labOut]
    float* pGlobal = buf.data();
    float* fused = pGlobal + n3;
    float* labRef = fused + n3;
    float* labOut = labRef + n3;

    // withStrength：同 EncoderEngine.withStrength（<=0 原图、>=1 映射、之间线性混合）
    const float* globalS;
    if (strength <= 0.0f) {
        globalS = rgb;
    } else if (strength >= 1.0f) {
        globalS = mapped;
    } else {
        const float s = strength;
        const float inv = 1.0f - s;
        for (std::size_t i = 0; i < n3; ++i) {
            pGlobal[i] = rgb[i] * inv + mapped[i] * s;
        }
        globalS = pGlobal;
    }

    if (protect <= 0.0f) {              // 桌面/Kotlin 同：直接返回全局结果
        std::memcpy(out, globalS, n3 * sizeof(float));
        return;
    }

    // 四区软融合。Kotlin 是 r 外层 / p 内层，对**固定像素**的加法顺序同为
    // skin → hair → cloth → bg；这里改成 p 外层（顺序不变）顺带省掉 skinBranch
    // 与 branches 两个中间数组。非肤三区共享同一次全局映射。
    const bool skinIsRaw = (protect >= 0.999f);
    const float invProtect = 1.0f - protect;
    for (std::size_t p = 0; p < n; ++p) {
        const std::size_t i = 3 * p;
        float b0;
        float b1;
        float b2;
        if (skinIsRaw) {                // Kotlin: protect>=0.999 时 skinBranch 就是原图
            b0 = rgb[i];
            b1 = rgb[i + 1];
            b2 = rgb[i + 2];
        } else {
            b0 = rgb[i] * protect + globalS[i] * invProtect;
            b1 = rgb[i + 1] * protect + globalS[i + 1] * invProtect;
            b2 = rgb[i + 2] * protect + globalS[i + 2] * invProtect;
        }
        const float w0 = wSkin[p];
        float fr = w0 * b0;
        float fg = w0 * b1;
        float fb = w0 * b2;
        const float wh = wHair[p];
        fr += wh * globalS[i];
        fg += wh * globalS[i + 1];
        fb += wh * globalS[i + 2];
        const float wc = wCloth[p];
        fr += wc * globalS[i];
        fg += wc * globalS[i + 1];
        fb += wc * globalS[i + 2];
        const float wb = wBg[p];
        fr += wb * globalS[i];
        fg += wb * globalS[i + 1];
        fb += wb * globalS[i + 2];
        fused[i] = fr;
        fused[i + 1] = fg;
        fused[i + 2] = fb;
    }

    // 亮度策略 global：融合结果只留 a/b，L 取全局映射（皮肤区排除在替换外）
    ct::rgbToLab(globalS, labRef, n);
    ct::rgbToLab(fused, labOut, n);
    for (std::size_t p = 0; p < n; ++p) {
        const float lw = wSkin[p];
        const std::size_t i = 3 * p;
        labOut[i] = labRef[i] * (1.0f - lw) + labOut[i] * lw;
    }
    ct::labToRgb(labOut, out, n);
}

// 像素间独立 ⇒ 按区间并行；区间内部（含 rgbToLab/labToRgb）一律单线程，
// 防嵌套线程爆炸（见 parallel.h 顶部约定）。
void fuseBandCore(const float* rgb, const float* mapped, const float* wSkin,
                  const float* wHair, const float* wCloth, const float* wBg,
                  float protect, float strength, float* out, std::size_t n) {
    ct::parallelFor(n, 32768, [&](std::size_t b, std::size_t e) {
        fuseBandRange(rgb, mapped, wSkin, wHair, wCloth, wBg,
                      protect, strength, out, b, e);
    });
}

// ---------------------------------------------------------------------------
// 统计类方法（reinhard/ot）的内容统计归约（2026-10-02 性能轮）：原先在
// TiledPipeline.migrate 里是 Kotlin 单线程逐像素累加（ot 档真机 src+stat
// ~2.5s/张）。这里改分块并行：固定块边界、块内顺序累加、块间按块序合并
// ⇒ 端内确定；与旧顺序版的差 ~1e-12（远低于统计类金标 u8 ≤2/255 容差，
// 桌面 numpy 的 pairwise 归约本就与顺序版不同序）。
// ---------------------------------------------------------------------------
void statSumRgbCore(const float* rgb, std::size_t n, double* out3) {
    const std::size_t grain = 32768;
    const std::size_t blocks = (n + grain - 1) / grain;
    std::vector<double> partial(blocks * 3, 0.0);
    ct::parallelFor(n, grain, [&](std::size_t b, std::size_t e) {
        double s0 = 0.0, s1 = 0.0, s2 = 0.0;
        for (std::size_t p = b; p < e; ++p) {
            s0 += rgb[3 * p]; s1 += rgb[3 * p + 1]; s2 += rgb[3 * p + 2];
        }
        const std::size_t k = (b / grain) * 3;
        partial[k] = s0; partial[k + 1] = s1; partial[k + 2] = s2;
    });
    for (std::size_t k = 0; k < blocks * 3; k += 3) {
        out3[0] += partial[k]; out3[1] += partial[k + 1]; out3[2] += partial[k + 2];
    }
}

// ot 协方差第二遍：d = x − mu 的上三角 Σ d⊗d（out6 = [00,01,02,11,12,22]）。
void statOuterRgbCore(const float* rgb, std::size_t n, const double* mu,
                      double* out6) {
    const std::size_t grain = 32768;
    const std::size_t blocks = (n + grain - 1) / grain;
    std::vector<double> partial(blocks * 6, 0.0);
    ct::parallelFor(n, grain, [&](std::size_t b, std::size_t e) {
        double o00 = 0.0, o01 = 0.0, o02 = 0.0;
        double o11 = 0.0, o12 = 0.0, o22 = 0.0;
        for (std::size_t p = b; p < e; ++p) {
            const double d0 = rgb[3 * p] - mu[0];
            const double d1 = rgb[3 * p + 1] - mu[1];
            const double d2 = rgb[3 * p + 2] - mu[2];
            o00 += d0 * d0; o01 += d0 * d1; o02 += d0 * d2;
            o11 += d1 * d1; o12 += d1 * d2; o22 += d2 * d2;
        }
        const std::size_t k = (b / grain) * 6;
        partial[k] = o00; partial[k + 1] = o01; partial[k + 2] = o02;
        partial[k + 3] = o11; partial[k + 4] = o12; partial[k + 5] = o22;
    });
    for (std::size_t k = 0; k < blocks * 6; k += 6) {
        out6[0] += partial[k]; out6[1] += partial[k + 1];
        out6[2] += partial[k + 2]; out6[3] += partial[k + 3];
        out6[4] += partial[k + 4]; out6[5] += partial[k + 5];
    }
}

// reinhard：RGB → LAB（ct:: 同源）+ Σ 与 Σ²（out6 = [ΣL,Σa,Σb, ΣL²,Σa²,Σb²]），
// 一趟替代 Kotlin 的"每段 rgbToLab 新数组 + 双循环累加"。
void statSumLabCore(const float* rgb, std::size_t n, double* out6) {
    const std::size_t grain = 32768;
    const std::size_t blocks = (n + grain - 1) / grain;
    std::vector<double> partial(blocks * 6, 0.0);
    ct::parallelFor(n, grain, [&](std::size_t b, std::size_t e) {
        std::vector<float> lab((e - b) * 3);
        ct::rgbToLab(rgb + 3 * b, lab.data(), e - b);
        double s0 = 0.0, s1 = 0.0, s2 = 0.0;
        double q0 = 0.0, q1 = 0.0, q2 = 0.0;
        for (std::size_t p = 0; p < e - b; ++p) {
            const double l = lab[3 * p];
            const double a = lab[3 * p + 1];
            const double bb = lab[3 * p + 2];
            s0 += l; s1 += a; s2 += bb;
            q0 += l * l; q1 += a * a; q2 += bb * bb;
        }
        const std::size_t k = (b / grain) * 6;
        partial[k] = s0; partial[k + 1] = s1; partial[k + 2] = s2;
        partial[k + 3] = q0; partial[k + 4] = q1; partial[k + 5] = q2;
    });
    for (std::size_t k = 0; k < blocks * 6; k += 6) {
        out6[0] += partial[k]; out6[1] += partial[k + 1];
        out6[2] += partial[k + 2]; out6[3] += partial[k + 3];
        out6[4] += partial[k + 4]; out6[5] += partial[k + 5];
    }
}

// ---- JNI 侧薄封装：数组取用/释放与异常兜底（失败一律返回 false → Kotlin 回退）----
// Critical 版取指针：段数组全是 MB 级（ART 不可移动堆）⇒ 返回**真指针零拷贝**
// （Get*ArrayElements 的拷贝语义在批量/胶片段级曾是 ~GB 级/张 的内存搬运）。
// 区间内只做纯 C++ 计算、不调其他 JNI；多数组同时持有（如 fuseBand 7 个）在
// ART 上实践安全，全部内核有逐位对拍关卡兜底。
struct FloatArr {
    JNIEnv* env = nullptr;
    jfloatArray arr = nullptr;
    jfloat* ptr = nullptr;

    bool get(JNIEnv* e, jfloatArray a) {
        env = e;
        arr = a;
        ptr = (a != nullptr)
            ? static_cast<jfloat*>(e->GetPrimitiveArrayCritical(a, nullptr))
            : nullptr;
        return ptr != nullptr;
    }
    void release(jint mode) {
        if (ptr != nullptr) env->ReleasePrimitiveArrayCritical(arr, ptr, mode);
        ptr = nullptr;
    }
};

struct DoubleArr {
    JNIEnv* env = nullptr;
    jdoubleArray arr = nullptr;
    jdouble* ptr = nullptr;

    bool get(JNIEnv* e, jdoubleArray a) {
        env = e;
        arr = a;
        ptr = (a != nullptr)
            ? static_cast<jdouble*>(e->GetPrimitiveArrayCritical(a, nullptr))
            : nullptr;
        return ptr != nullptr;
    }
    void release(jint mode) {
        if (ptr != nullptr) env->ReleasePrimitiveArrayCritical(arr, ptr, mode);
        ptr = nullptr;
    }
};

}  // namespace

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_colortrace_engine_NativeKernels_lutApply(JNIEnv* env, jobject /* this */,
                                                 jdoubleArray jtable, jint size,
                                                 jfloatArray jrgb, jfloatArray jout) {
    if (jtable == nullptr || jrgb == nullptr || jout == nullptr || size < 2) return JNI_FALSE;
    const jsize len = env->GetArrayLength(jrgb);
    if (len <= 0 || len % 3 != 0) return JNI_FALSE;
    if (env->GetArrayLength(jout) != len) return JNI_FALSE;
    if (env->GetArrayLength(jtable) != static_cast<jsize>(size) * size * size * 3) {
        return JNI_FALSE;
    }
    try {
        auto* table = static_cast<jdouble*>(env->GetPrimitiveArrayCritical(jtable, nullptr));
        if (table == nullptr) return JNI_FALSE;
        FloatArr in;
        if (!in.get(env, jrgb)) {
            env->ReleasePrimitiveArrayCritical(jtable, table, JNI_ABORT);
            return JNI_FALSE;
        }
        FloatArr dst;
        if (!dst.get(env, jout)) {
            in.release(JNI_ABORT);
            env->ReleasePrimitiveArrayCritical(jtable, table, JNI_ABORT);
            return JNI_FALSE;
        }
        bool ok;
        try {
            lutApplyCore(table, size, in.ptr, dst.ptr, static_cast<std::size_t>(len) / 3);
            ok = true;
        } catch (...) {
            ok = false;                 // 失败时丢弃半成品，交给 Kotlin 回退重算
        }
        dst.release(ok ? 0 : JNI_ABORT);
        in.release(JNI_ABORT);
        env->ReleasePrimitiveArrayCritical(jtable, table, JNI_ABORT);
        return ok ? JNI_TRUE : JNI_FALSE;
    } catch (...) {
        return JNI_FALSE;
    }
}

JNIEXPORT jboolean JNICALL
Java_com_colortrace_engine_NativeKernels_fuseBand(
        JNIEnv* env, jobject /* this */, jfloatArray jrgb, jfloatArray jmapped,
        jfloatArray jwSkin, jfloatArray jwHair, jfloatArray jwCloth, jfloatArray jwBg,
        jfloat protect, jfloat strength, jfloatArray jout) {
    const jsize len = (jrgb != nullptr) ? env->GetArrayLength(jrgb) : 0;
    if (len <= 0 || len % 3 != 0) return JNI_FALSE;
    const jsize n = len / 3;
    if (env->GetArrayLength(jmapped) != len || env->GetArrayLength(jwSkin) != n ||
        env->GetArrayLength(jwHair) != n || env->GetArrayLength(jwCloth) != n ||
        env->GetArrayLength(jwBg) != n || env->GetArrayLength(jout) != len) {
        return JNI_FALSE;
    }
    FloatArr rgb;
    FloatArr mapped;
    FloatArr wSkin;
    FloatArr wHair;
    FloatArr wCloth;
    FloatArr wBg;
    FloatArr out;
    try {
        if (!rgb.get(env, jrgb) || !mapped.get(env, jmapped) || !wSkin.get(env, jwSkin) ||
            !wHair.get(env, jwHair) || !wCloth.get(env, jwCloth) || !wBg.get(env, jwBg) ||
            !out.get(env, jout)) {
            rgb.release(JNI_ABORT);
            mapped.release(JNI_ABORT);
            wSkin.release(JNI_ABORT);
            wHair.release(JNI_ABORT);
            wCloth.release(JNI_ABORT);
            wBg.release(JNI_ABORT);
            out.release(JNI_ABORT);
            return JNI_FALSE;
        }
        bool ok;
        try {
            fuseBandCore(rgb.ptr, mapped.ptr, wSkin.ptr, wHair.ptr, wCloth.ptr, wBg.ptr,
                         protect, strength, out.ptr, static_cast<std::size_t>(n));
            ok = true;
        } catch (...) {
            ok = false;                 // 同上：丢弃半成品，Kotlin 回退重算
        }
        rgb.release(JNI_ABORT);
        mapped.release(JNI_ABORT);
        wSkin.release(JNI_ABORT);
        wHair.release(JNI_ABORT);
        wCloth.release(JNI_ABORT);
        wBg.release(JNI_ABORT);
        out.release(ok ? 0 : JNI_ABORT);
        return ok ? JNI_TRUE : JNI_FALSE;
    } catch (...) {
        return JNI_FALSE;
    }
}

JNIEXPORT jboolean JNICALL
Java_com_colortrace_engine_NativeKernels_developApply(
        JNIEnv* env, jobject /* this */, jfloatArray jrgb, jfloatArray jparams,
        jfloatArray jout) {
    if (jrgb == nullptr || jparams == nullptr || jout == nullptr) return JNI_FALSE;
    const jsize len = env->GetArrayLength(jrgb);
    if (len <= 0 || len % 3 != 0) return JNI_FALSE;
    if (env->GetArrayLength(jout) != len) return JNI_FALSE;
    if (env->GetArrayLength(jparams) != 17) return JNI_FALSE;
    auto* prm = static_cast<jfloat*>(env->GetPrimitiveArrayCritical(jparams, nullptr));
    if (prm == nullptr) return JNI_FALSE;
    FloatArr rgb;
    FloatArr out;
    try {
        if (!rgb.get(env, jrgb) || !out.get(env, jout)) {
            rgb.release(JNI_ABORT);
            out.release(JNI_ABORT);
            env->ReleasePrimitiveArrayCritical(jparams, prm, JNI_ABORT);
            return JNI_FALSE;
        }
        bool ok;
        try {
            developCore(rgb.ptr, prm, out.ptr, static_cast<std::size_t>(len) / 3);
            ok = true;
        } catch (...) {
            ok = false;                 // 丢弃半成品，Kotlin 回退重算
        }
        rgb.release(JNI_ABORT);
        out.release(ok ? 0 : JNI_ABORT);
        env->ReleasePrimitiveArrayCritical(jparams, prm, JNI_ABORT);
        return ok ? JNI_TRUE : JNI_FALSE;
    } catch (...) {
        env->ReleasePrimitiveArrayCritical(jparams, prm, JNI_ABORT);
        return JNI_FALSE;
    }
}

JNIEXPORT jboolean JNICALL
Java_com_colortrace_engine_NativeKernels_rgbToBgrU8(
        JNIEnv* env, jobject /* this */, jfloatArray jrgb, jbyteArray jout) {
    if (jrgb == nullptr || jout == nullptr) return JNI_FALSE;
    const jsize len = env->GetArrayLength(jrgb);
    if (len <= 0 || len % 3 != 0) return JNI_FALSE;
    if (env->GetArrayLength(jout) != len) return JNI_FALSE;
    auto* rgb = static_cast<jfloat*>(env->GetPrimitiveArrayCritical(jrgb, nullptr));
    if (rgb == nullptr) return JNI_FALSE;
    auto* out = static_cast<jbyte*>(env->GetPrimitiveArrayCritical(jout, nullptr));
    if (out == nullptr) {
        env->ReleasePrimitiveArrayCritical(jrgb, rgb, JNI_ABORT);
        return JNI_FALSE;
    }
    try {
        rgbToBgrU8Core(rgb, reinterpret_cast<std::uint8_t*>(out),
                       static_cast<std::size_t>(len) / 3);
    } catch (...) {
        env->ReleasePrimitiveArrayCritical(jrgb, rgb, JNI_ABORT);
        env->ReleasePrimitiveArrayCritical(jout, out, JNI_ABORT);
        return JNI_FALSE;
    }
    env->ReleasePrimitiveArrayCritical(jrgb, rgb, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(jout, out, 0);
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_com_colortrace_engine_NativeKernels_argbToRgbF32(
        JNIEnv* env, jobject /* this */, jintArray jargb, jint n,
        jfloatArray jrgb, jbyteArray jrgbU8) {
    if (jargb == nullptr || jrgb == nullptr || n <= 0) return JNI_FALSE;
    if (env->GetArrayLength(jargb) < n) return JNI_FALSE;
    if (env->GetArrayLength(jrgb) < n * 3) return JNI_FALSE;
    if (jrgbU8 != nullptr && env->GetArrayLength(jrgbU8) < n * 3) return JNI_FALSE;
    auto* argb = static_cast<jint*>(env->GetPrimitiveArrayCritical(jargb, nullptr));
    if (argb == nullptr) return JNI_FALSE;
    auto* rgb = static_cast<jfloat*>(env->GetPrimitiveArrayCritical(jrgb, nullptr));
    if (rgb == nullptr) {
        env->ReleasePrimitiveArrayCritical(jargb, argb, JNI_ABORT);
        return JNI_FALSE;
    }
    auto* u8 = (jrgbU8 != nullptr)
        ? static_cast<jbyte*>(env->GetPrimitiveArrayCritical(jrgbU8, nullptr))
        : nullptr;
    try {
        argbToRgbF32Core(reinterpret_cast<std::uint32_t*>(argb),
                         static_cast<std::size_t>(n),
                         rgb, reinterpret_cast<std::uint8_t*>(u8));
    } catch (...) {
        env->ReleasePrimitiveArrayCritical(jargb, argb, JNI_ABORT);
        env->ReleasePrimitiveArrayCritical(jrgb, rgb, JNI_ABORT);
        if (u8 != nullptr) env->ReleasePrimitiveArrayCritical(jrgbU8, u8, JNI_ABORT);
        return JNI_FALSE;
    }
    env->ReleasePrimitiveArrayCritical(jargb, argb, JNI_ABORT);   // 只读，不写回
    env->ReleasePrimitiveArrayCritical(jrgb, rgb, 0);
    if (u8 != nullptr) env->ReleasePrimitiveArrayCritical(jrgbU8, u8, 0);
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_com_colortrace_engine_NativeKernels_strengthMix(
        JNIEnv* env, jobject /* this */, jfloatArray jrgb, jfloatArray jmapped,
        jfloat s, jfloatArray jout) {
    if (jrgb == nullptr || jmapped == nullptr || jout == nullptr) return JNI_FALSE;
    const jsize len = env->GetArrayLength(jrgb);
    if (len <= 0 || len % 3 != 0) return JNI_FALSE;
    if (env->GetArrayLength(jmapped) != len || env->GetArrayLength(jout) != len) {
        return JNI_FALSE;
    }
    FloatArr rgb;
    FloatArr mapped;
    FloatArr out;
    if (!rgb.get(env, jrgb) || !mapped.get(env, jmapped) || !out.get(env, jout)) {
        rgb.release(JNI_ABORT);
        mapped.release(JNI_ABORT);
        out.release(JNI_ABORT);
        return JNI_FALSE;
    }
    bool ok;
    try {
        strengthMixCore(rgb.ptr, mapped.ptr, s, out.ptr,
                        static_cast<std::size_t>(len) / 3);
        ok = true;
    } catch (...) {
        ok = false;                     // 丢弃半成品，Kotlin 回退重算
    }
    rgb.release(JNI_ABORT);
    mapped.release(JNI_ABORT);
    out.release(ok ? 0 : JNI_ABORT);
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_colortrace_engine_NativeKernels_statSumRgb(
        JNIEnv* env, jobject /* this */, jfloatArray jrgb, jdoubleArray jout) {
    if (jrgb == nullptr || jout == nullptr) return JNI_FALSE;
    const jsize len = env->GetArrayLength(jrgb);
    if (len <= 0 || len % 3 != 0) return JNI_FALSE;
    if (env->GetArrayLength(jout) != 3) return JNI_FALSE;
    auto* rgb = static_cast<jfloat*>(env->GetPrimitiveArrayCritical(jrgb, nullptr));
    if (rgb == nullptr) return JNI_FALSE;
    auto* out = static_cast<jdouble*>(env->GetPrimitiveArrayCritical(jout, nullptr));
    if (out == nullptr) {
        env->ReleasePrimitiveArrayCritical(jrgb, rgb, JNI_ABORT);
        return JNI_FALSE;
    }
    bool ok;
    try {
        statSumRgbCore(rgb, static_cast<std::size_t>(len) / 3, out);
        ok = true;
    } catch (...) {
        ok = false;
    }
    env->ReleasePrimitiveArrayCritical(jrgb, rgb, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(jout, out, ok ? 0 : JNI_ABORT);
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_colortrace_engine_NativeKernels_statOuterRgb(
        JNIEnv* env, jobject /* this */, jfloatArray jrgb, jdoubleArray jmu,
        jdoubleArray jout) {
    if (jrgb == nullptr || jmu == nullptr || jout == nullptr) return JNI_FALSE;
    const jsize len = env->GetArrayLength(jrgb);
    if (len <= 0 || len % 3 != 0) return JNI_FALSE;
    if (env->GetArrayLength(jmu) != 3 || env->GetArrayLength(jout) != 6)
        return JNI_FALSE;
    auto* rgb = static_cast<jfloat*>(env->GetPrimitiveArrayCritical(jrgb, nullptr));
    if (rgb == nullptr) return JNI_FALSE;
    auto* mu = static_cast<jdouble*>(env->GetPrimitiveArrayCritical(jmu, nullptr));
    if (mu == nullptr) {
        env->ReleasePrimitiveArrayCritical(jrgb, rgb, JNI_ABORT);
        return JNI_FALSE;
    }
    auto* out = static_cast<jdouble*>(env->GetPrimitiveArrayCritical(jout, nullptr));
    if (out == nullptr) {
        env->ReleasePrimitiveArrayCritical(jrgb, rgb, JNI_ABORT);
        env->ReleasePrimitiveArrayCritical(jmu, mu, JNI_ABORT);
        return JNI_FALSE;
    }
    bool ok;
    try {
        statOuterRgbCore(rgb, static_cast<std::size_t>(len) / 3, mu, out);
        ok = true;
    } catch (...) {
        ok = false;
    }
    env->ReleasePrimitiveArrayCritical(jrgb, rgb, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(jmu, mu, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(jout, out, ok ? 0 : JNI_ABORT);
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_colortrace_engine_NativeKernels_statSumLab(
        JNIEnv* env, jobject /* this */, jfloatArray jrgb, jdoubleArray jout) {
    if (jrgb == nullptr || jout == nullptr) return JNI_FALSE;
    const jsize len = env->GetArrayLength(jrgb);
    if (len <= 0 || len % 3 != 0) return JNI_FALSE;
    if (env->GetArrayLength(jout) != 6) return JNI_FALSE;
    auto* rgb = static_cast<jfloat*>(env->GetPrimitiveArrayCritical(jrgb, nullptr));
    if (rgb == nullptr) return JNI_FALSE;
    auto* out = static_cast<jdouble*>(env->GetPrimitiveArrayCritical(jout, nullptr));
    if (out == nullptr) {
        env->ReleasePrimitiveArrayCritical(jrgb, rgb, JNI_ABORT);
        return JNI_FALSE;
    }
    bool ok;
    try {
        statSumLabCore(rgb, static_cast<std::size_t>(len) / 3, out);
        ok = true;
    } catch (...) {
        ok = false;
    }
    env->ReleasePrimitiveArrayCritical(jrgb, rgb, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(jout, out, ok ? 0 : JNI_ABORT);
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_colortrace_engine_NativeKernels_flowProbes(
        JNIEnv* env, jobject /* this */, jdoubleArray jprobes, jdoubleArray jwc,
        jdoubleArray jws, jint hidden, jdouble strength, jint steps) {
    if (jprobes == nullptr || jwc == nullptr || jws == nullptr) return JNI_FALSE;
    if (hidden <= 0 || steps < 0) return JNI_FALSE;
    const jsize len = env->GetArrayLength(jprobes);
    if (len <= 0 || len % 3 != 0) return JNI_FALSE;
    if (env->GetArrayLength(jwc) != 8 * hidden + 3 ||
        env->GetArrayLength(jws) != 8 * hidden + 3) {
        return JNI_FALSE;
    }
    DoubleArr probes;
    DoubleArr wc;
    DoubleArr ws;
    try {
        if (!probes.get(env, jprobes) || !wc.get(env, jwc) || !ws.get(env, jws)) {
            probes.release(JNI_ABORT);
            wc.release(JNI_ABORT);
            ws.release(JNI_ABORT);
            return JNI_FALSE;
        }
        bool ok;
        try {
            flowProbesCore(probes.ptr, static_cast<std::size_t>(len) / 3,
                           wc.ptr, ws.ptr, hidden, strength, steps);
            ok = true;
        } catch (...) {
            ok = false;                     // 丢弃半成品，Kotlin 回退重算
        }
        probes.release(ok ? 0 : JNI_ABORT); // probes 原地写回
        wc.release(JNI_ABORT);
        ws.release(JNI_ABORT);
        return ok ? JNI_TRUE : JNI_FALSE;
    } catch (...) {
        return JNI_FALSE;
    }
}

JNIEXPORT jboolean JNICALL
Java_com_colortrace_engine_NativeKernels_regionStatBand(
        JNIEnv* env, jobject /* this */, jfloatArray jrgb, jfloatArray jmapped,
        jfloatArray jwSkin, jfloatArray jwHair, jfloatArray jwCloth,
        jfloatArray jwBg, jfloatArray jprm, jfloat protect, jfloat strength,
        jfloatArray jout) {
    const jsize len = (jrgb != nullptr) ? env->GetArrayLength(jrgb) : 0;
    if (len <= 0 || len % 3 != 0) return JNI_FALSE;
    const jsize n = len / 3;
    if (jprm == nullptr || env->GetArrayLength(jprm) < 30) return JNI_FALSE;
    // prm[0] 决定打包长度（lab=30 / ot=66），先取元素再校验
    jfloat isLab = 0.0f;
    env->GetFloatArrayRegion(jprm, 0, 1, &isLab);
    const jsize prmLen = (isLab != 0.0f) ? 30 : 66;
    if (env->GetArrayLength(jprm) != prmLen) return JNI_FALSE;
    if (env->GetArrayLength(jmapped) != len || env->GetArrayLength(jwSkin) != n ||
        env->GetArrayLength(jwHair) != n || env->GetArrayLength(jwCloth) != n ||
        env->GetArrayLength(jwBg) != n || env->GetArrayLength(jout) != len) {
        return JNI_FALSE;
    }
    FloatArr rgb;
    FloatArr mapped;
    FloatArr wSkin;
    FloatArr wHair;
    FloatArr wCloth;
    FloatArr wBg;
    FloatArr prm;
    FloatArr out;
    try {
        if (!rgb.get(env, jrgb) || !mapped.get(env, jmapped) ||
            !wSkin.get(env, jwSkin) || !wHair.get(env, jwHair) ||
            !wCloth.get(env, jwCloth) || !wBg.get(env, jwBg) ||
            !prm.get(env, jprm) || !out.get(env, jout)) {
            rgb.release(JNI_ABORT);
            mapped.release(JNI_ABORT);
            wSkin.release(JNI_ABORT);
            wHair.release(JNI_ABORT);
            wCloth.release(JNI_ABORT);
            wBg.release(JNI_ABORT);
            prm.release(JNI_ABORT);
            out.release(JNI_ABORT);
            return JNI_FALSE;
        }
        bool ok;
        try {
            regionStatBandCore(rgb.ptr, mapped.ptr, wSkin.ptr, wHair.ptr,
                               wCloth.ptr, wBg.ptr, prm.ptr, protect, strength,
                               out.ptr, static_cast<std::size_t>(n));
            ok = true;
        } catch (...) {
            ok = false;
        }
        rgb.release(JNI_ABORT);
        mapped.release(JNI_ABORT);
        wSkin.release(JNI_ABORT);
        wHair.release(JNI_ABORT);
        wCloth.release(JNI_ABORT);
        wBg.release(JNI_ABORT);
        prm.release(JNI_ABORT);
        out.release(ok ? 0 : JNI_ABORT);
        return ok ? JNI_TRUE : JNI_FALSE;
    } catch (...) {
        return JNI_FALSE;
    }
}

JNIEXPORT jboolean JNICALL
Java_com_colortrace_engine_NativeKernels_chromaAnchor(
        JNIEnv* env, jobject /* this */, jfloatArray jrgb,
        jfloatArray jglobalOut, jfloat protect, jfloatArray jout) {
    if (jrgb == nullptr || jglobalOut == nullptr || jout == nullptr) return JNI_FALSE;
    const jsize len = env->GetArrayLength(jrgb);
    if (len <= 0 || len % 3 != 0) return JNI_FALSE;
    if (env->GetArrayLength(jglobalOut) != len || env->GetArrayLength(jout) != len) {
        return JNI_FALSE;
    }
    FloatArr rgb;
    FloatArr globalOut;
    FloatArr out;
    try {
        if (!rgb.get(env, jrgb) || !globalOut.get(env, jglobalOut) ||
            !out.get(env, jout)) {
            rgb.release(JNI_ABORT);
            globalOut.release(JNI_ABORT);
            out.release(JNI_ABORT);
            return JNI_FALSE;
        }
        bool ok;
        try {
            chromaAnchorCore(rgb.ptr, globalOut.ptr, protect, out.ptr,
                             static_cast<std::size_t>(len) / 3);
            ok = true;
        } catch (...) {
            ok = false;
        }
        rgb.release(JNI_ABORT);
        globalOut.release(JNI_ABORT);
        out.release(ok ? 0 : JNI_ABORT);
        return ok ? JNI_TRUE : JNI_FALSE;
    } catch (...) {
        return JNI_FALSE;
    }
}

}  // extern "C"