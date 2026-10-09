package com.colortrace.poc

import android.content.Context
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers

/**
 * 进程级引擎单例（P2.18 批量前台服务化）：Activity 与 [BatchExportService] 共享
 * 同一份 [EngineRepository] 与同一条单车道调度器。
 *
 * 为什么必须单例：① 引擎非线程安全（LUT 缓存），Activity 预览与服务批量必须
 * 排队互斥；② 模型 ≈23MB + ONNX 网解析是按 SHA-256 跨实例共享的（P2.12）——
 * 第二个实例纯属浪费；③ 前台服务跑在**同进程**（不设 android:process），
 * 模型加载一次、批量与预览天然共存。
 */
object EngineBox {
    /** 引擎非线程安全：所有引擎调用走这条单车道调度器。 */
    val engineDispatcher: CoroutineDispatcher = Dispatchers.Default.limitedParallelism(1)

    @Volatile
    private var repo: EngineRepository? = null

    fun repository(ctx: Context): EngineRepository =
        repo ?: synchronized(this) {
            repo ?: EngineRepository(ctx.applicationContext).also { repo = it }
        }
}

/** 批量导出进度（同进程共享：Service 写、UI 订阅；null = 当前无批量）。 */
object BatchBus {
    data class State(
        val running: Boolean,
        val total: Int,
        val done: Int,
        val text: String,
    )

    val state = kotlinx.coroutines.flow.MutableStateFlow<State?>(null)
}
