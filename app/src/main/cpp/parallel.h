// 逐像素内核的并行骨架（2026-10-01 性能轮）。
//
// 逐位纪律：**只并行"像素间独立"的循环**——每个像素的输入、输出与像素内的
// 累加序在并行化后与单线程逐位一致（不引入任何跨像素归约，不改任何单条
// 浮点运算，-ffp-contract=off 语义不变）⇒ NativeKernelsTest 的 native↔Kotlin
// 逐位对拍在多线程下依旧 0 失配。
//
// 调用约定：
//  - fn(begin, end) 处理像素区间 [begin, end)，像素数 n 按 grain 切块；
//  - 线程数 = min(硬件核数, 8)（big.LITTLE 的小核也算数；8 封顶防小图核风暴）；
//    块数 < 2 或核数 < 2 时**原地单线程**跑——小段（如 128² 缩略图）不值得
//    每次花 ~0.5ms 创建线程组；
//  - 动态领取（atomic 计数）而非静态均分：小核/大核速度差 2~3×，静态均分
//    会被最慢的核拖住；
//  - worker 内异常（几乎只会是 vector resize 的 bad_alloc）被捕获、join 后
//    在主线程重抛——JNI 层的 try/catch 原样接住并回退 Kotlin。
//
// 嵌套禁止：ct::rgbToLab/ct::labToRgb 等"被并行区间内部调用"的函数**不得**
// 再自己 parallelFor（会 8×8 线程爆炸）。当前分工：
//   并行入口 = 各 JNI 核（fuseBandCore / lutApplyCore / developCore /
//   rgbToBgrU8Core / argbToRgbF32Core / lab_conv 的 run 包装）；
//   区间内部直接调 ct:: 单线程版本。

#ifndef COLORTRACE_PARALLEL_H
#define COLORTRACE_PARALLEL_H

#include <algorithm>
#include <atomic>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace ct {

template <typename Fn>
inline void parallelFor(std::size_t n, std::size_t grain, Fn fn) {
    if (n == 0) return;
    const std::size_t hw = std::thread::hardware_concurrency();
    const std::size_t threads = std::min(hw == 0 ? std::size_t{1} : hw,
                                         std::size_t{8});
    const std::size_t nChunks = (n + grain - 1) / grain;
    if (threads < 2 || nChunks < 2) {
        fn(0, n);
        return;
    }
    const std::size_t nWorkers = std::min(threads, nChunks) - 1;  // 主线程算一份

    std::atomic<std::size_t> next{0};
    std::mutex errMu;
    std::exception_ptr err;
    auto worker = [&]() {
        try {
            for (;;) {
                const std::size_t c = next.fetch_add(1, std::memory_order_relaxed);
                const std::size_t b = c * grain;
                if (b >= n) break;
                fn(b, std::min(b + grain, n));
            }
        } catch (...) {
            const std::lock_guard<std::mutex> lk(errMu);
            if (!err) err = std::current_exception();
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(nWorkers);
    for (std::size_t i = 0; i < nWorkers; ++i) pool.emplace_back(worker);
    worker();                                   // 主线程也领活
    for (auto& t : pool) t.join();
    if (err) std::rethrow_exception(err);
}

}  // namespace ct

#endif  // COLORTRACE_PARALLEL_H
