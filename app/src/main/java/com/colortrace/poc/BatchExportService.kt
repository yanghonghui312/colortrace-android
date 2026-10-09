package com.colortrace.poc

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.net.Uri
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import com.colortrace.DebugLog
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * 批量导出前台服务（P2.18，用户拍板"切后台批量照常跑，通知栏进度条"）：
 * 批量执行权从 Activity 移交到这里——FGS 把进程提到前台级，切后台/锁屏后
 * OriginOS 不再冻结/杀掉宿主进程；通知栏实时进度 + 「停止」按钮。
 *
 * 设计要点：
 *  - **同进程**（不设 android:process）：引擎经 [EngineBox] 单例共享——模型不重复
 *    加载、与 Activity 预览在同一条单车道调度器上排队互斥；
 *  - **会话按快照重建**（与续传同一逻辑：`fitSample(region = mode==REGION)` /
 *    `loadPreset`）——E2E 已证重建会话产物与原会话 MD5 逐字节一致，不依赖 Activity
 *    的活会话；
 *  - **续传机制原样**：`BatchResumeStore` 标记/清理在 [runBatchCore] 里——强杀/失败
 *    后重启仍走既有续传对话框，服务只是把"能跑完"变成常态；
 *  - **类型 `dataSync`**：Android 15+ 有 6h/24h 配额，分钟级批次无虞（targetSdk 34
 *    无需 onTimeout）。不用 WorkManager（可延期语义不符，
 *    进度/续传自有，零新依赖）。
 */
class BatchExportService : Service() {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        ensureChannel(this)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_CANCEL) {
            DebugLog.i("批量导出：通知栏请求停止（已完成下标保留，下次启动可续传）")
            scope.cancel()
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
            return START_NOT_STICKY
        }
        // startForegroundService 启动后必须立刻进前台态（5s 内）
        startForeground(
            NOTIF_ID, buildNotification("准备中…", 0, 0, indeterminate = true),
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q)
                ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC else 0)
        if (BatchBus.state.value?.running == true) {
            DebugLog.i("批量导出：已有批量在跑，忽略重复启动")
            return START_NOT_STICKY
        }
        scope.launch { run() }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        scope.cancel()
        super.onDestroy()
    }

    private suspend fun run() {
        val repo = EngineBox.repository(this)
        val ctx = this
        var statusText = ""
        try {
            val job0 = withContext(Dispatchers.IO) { BatchResumeStore.load(ctx) }
            if (job0 == null || job0.pending.isEmpty()) {
                DebugLog.i("批量导出：无待跑任务，服务退出")
                BatchBus.state.value = null
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf()
                return
            }
            val total = job0.uris.size
            BatchBus.state.value = BatchBus.State(
                running = true, total = total, done = job0.done.size, text = "准备中…")

            // 会话按快照重建（与续传同逻辑；不用 Activity 的活会话——服务可独立于
            // Activity 存活，且重建等价性已被强杀 E2E 的 MD5 复现证明）
            val session = withContext(EngineBox.engineDispatcher) {
                if (job0.presetJson != null) repo.loadPreset(job0.presetJson)
                else {
                    val bmp = withContext(Dispatchers.IO) {
                        repo.decodeCapped(Uri.parse(job0.sampleUri!!),
                                          EngineRepository.PROC_MAX)
                    }
                    repo.fitSample(bmp, job0.method,
                                   region = job0.mode == ProtectMode.REGION.name)
                }
            }

            var lastNotifMs = 0L
            val cb = object : BatchCallbacks {
                override fun onPhotoStart(idx: Int, total: Int, uri: Uri) {
                    updateNotification("第 ${idx + 1}/$total 张", idx, total, force = true)
                    BatchBus.state.value = BatchBus.State(
                        running = true, total = total, done = idx,
                        text = "批量导出 ${idx + 1}/$total")
                }
                override fun onStep(idx: Int, total: Int, d: Int, n: Int) {
                    val pct = if (n > 0) d * 100 / n else 0
                    updateNotification(
                        "第 ${idx + 1}/$total 张 · $pct%", idx, total, force = false)
                    // App 内文案与通知同格式（busy 覆盖层镜像）
                    BatchBus.state.value = BatchBus.State(
                        running = true, total = total, done = idx,
                        text = "批量导出 ${idx + 1}/$total · $pct%")
                }
                override fun onPhotoDone(idx: Int, total: Int, w: Int, h: Int, ms: Long) {
                    updateNotification("已完成 ${idx + 1}/$total", idx + 1, total, force = true)
                    BatchBus.state.value = BatchBus.State(
                        running = true, total = total, done = idx + 1,
                        text = "批量导出 ${idx + 1}/$total")
                }
                override fun onStatus(text: String) { statusText = text }
            }

            // resume 只表示"这是接着上次没跑完的跑"（有已完成项）——它只影响汇总/日志
            // 文案。P2.21 修正：此前写死 true，导致全新批次也说"本次…全部保存成功"。
            val job = runBatchCore(ctx, repo, EngineBox.engineDispatcher, session,
                                   job0, job0.pending,
                                   resume = job0.done.isNotEmpty(), cb = cb)

            BatchBus.state.value = BatchBus.State(
                running = false, total = total, done = job.done.size, text = statusText)
            // 结束通知（非 ongoing，可滑掉）
            getSystemService(NotificationManager::class.java).notify(
                NOTIF_ID + 1, buildNotification(statusText, job.done.size, total,
                                                indeterminate = false, ongoing = false))
            DebugLog.i("批量导出服务结束 done=${job.done.size}/$total")
        } catch (e: CancellationException) {
            BatchBus.state.value = BatchBus.State(
                running = false, total = 0, done = 0, text = "批量导出已停止（下次可继续）")
            throw e
        } catch (e: Exception) {
            DebugLog.e("批量导出服务失败", e)
            BatchBus.state.value = BatchBus.State(
                running = false, total = 0, done = 0, text = "批量导出失败")
            getSystemService(NotificationManager::class.java).notify(
                NOTIF_ID + 1, buildNotification("批量导出失败（下次可继续）",
                    0, 0, indeterminate = false, ongoing = false))
        } finally {
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
        }
    }

    private fun updateNotification(text: String, done: Int, total: Int, force: Boolean) {
        val now = System.currentTimeMillis()
        if (!force && now - lastNotifMs < NOTIF_MIN_INTERVAL_MS) return
        lastNotifMs = now
        getSystemService(NotificationManager::class.java)
            .notify(NOTIF_ID, buildNotification(text, done, total, indeterminate = false))
    }
    private var lastNotifMs = 0L

    // ---- 通知 ----

    private fun buildNotification(text: String, progress: Int, max: Int,
                                  indeterminate: Boolean,
                                  ongoing: Boolean = true): Notification {
        val b = NotificationCompat.Builder(this, CHANNEL_ID)
            // 通知小图标只取 alpha 并被系统染色 ⇒ 不能用彩色启动图标（会糊成灰块），
            // 用纯白剪影 ic_stat_colortrace（与启动图标单色层同形）。
            // 历史教训（P2.18）：当时 manifest 尚无 android:icon ⇒ setSmallIcon(0)
            // 在真机上 "Bad notification for startForeground" 闪退——小图标必须显式指定。
            .setSmallIcon(R.drawable.ic_stat_colortrace)
            .setContentTitle("追色引擎 · 批量导出")
            .setContentText(text)
            .setOnlyAlertOnce(true)
            .setSilent(true)
        if (max > 0) b.setProgress(max, progress, indeterminate)
        else b.setProgress(0, 0, indeterminate)
        if (ongoing) {
            b.setOngoing(true)
            b.addAction(0, "停止", PendingIntent.getService(
                this, 0,
                Intent(this, BatchExportService::class.java).setAction(ACTION_CANCEL),
                PendingIntent.FLAG_IMMUTABLE))
        }
        return b.build()
    }

    companion object {
        private const val CHANNEL_ID = "batch_export"
        const val NOTIF_ID = 41
        const val ACTION_CANCEL = "com.colortrace.poc.batch.CANCEL"
        private const val NOTIF_MIN_INTERVAL_MS = 600L

        /** 启动/续传批量（调用方先把快照落盘；服务从快照自举）。 */
        fun start(ctx: Context) {
            ctx.startForegroundService(
                Intent(ctx, BatchExportService::class.java))
        }

        fun ensureChannel(ctx: Context) {
            val mgr = ctx.getSystemService(NotificationManager::class.java)
            if (mgr.getNotificationChannel(CHANNEL_ID) == null) {
                mgr.createNotificationChannel(
                    NotificationChannel(CHANNEL_ID, "批量导出",
                                        NotificationManager.IMPORTANCE_LOW).apply {
                        description = "批量导出的进度与结果"
                    })
            }
        }
    }
}
