package com.colortrace

import android.content.Context
import android.os.Build
import android.util.Log
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * 轻量文件日志（vivo logd 限流的兜底）：与 logcat 同 tag 同内容，额外追加到
 * 应用私有目录 `files/colortrace.log`——OriginOS 会抑制前台应用的 logd 输出
 * （`log.ratelimit.level=3`，工程菜单才能关），文件不受影响，可用
 * `adb shell run-as com.colortrace.poc cat files/colortrace.log` 拉取。
 *
 * 能力：
 *  - [i] 阶段耗时等常规信息；[e] **带完整堆栈**的错误（诊断必须）；
 *  - 启动横幅：设备型号 / API / App 版本 / OpenCV 版本（对齐版本差异）；
 *  - 未捕获异常兜底（含 OOM）：写入文件后再交给系统默认处理（App 仍会崩）；
 *  - 超 512KB 保留**后半段**（不是清空），避免一次溢出丢掉全部现场。
 *
 * 放在 `com.colortrace` 根包而非 `poc`：引擎层（engine 包）的静默降级
 * （分割模型加载/推理失败）也要落盘，不能让报错消失。
 * 调用频率为阶段级/错误级（无逐像素），synchronized 单锁。
 */
object DebugLog {
    private const val TAG = "colortrace"
    private const val FILE_NAME = "colortrace.log"
    private const val MAX_BYTES = 512 * 1024
    private const val KEEP_BYTES = 256 * 1024
    private val TIME = SimpleDateFormat("MM-dd HH:mm:ss.SSS", Locale.US)

    private var file: File? = null

    /**
     * 启动横幅原文（机型 / API / App 版本 / OpenCV / native）。帮助页的「关于」直接显示它
     * ——排查版本差异时用户能自己念出来，不必再教他跑 adb。
     */
    var startBanner: String = ""
        private set

    fun init(context: Context) {
        if (file != null) return
        file = File(context.filesDir, FILE_NAME)
        val version = runCatching {
            val pi = context.packageManager.getPackageInfo(context.packageName, 0)
            "v${pi.versionName}(${pi.longVersionCode})"
        }.getOrElse { "v?" }
        val opencv = runCatching { org.opencv.core.Core.VERSION }.getOrElse { "?" }
        // native 版本是第一次排查"快/慢、逐位与否"必须先看的一项（加载失败会是 n/a）
        val native = runCatching { com.colortrace.engine.NativeLab.version }.getOrElse { "?" }
        startBanner = "${Build.MODEL} / API ${Build.VERSION.SDK_INT} / " +
                "app $version / opencv $opencv / native $native"
        i("──── app start ──── $startBanner")
        installCrashHandler()
    }

    /**
     * 读日志尾部（帮助页「查看运行日志」用）：最多 [maxBytes] 字节，从下一个换行边界
     * 开始，避免首行被截成半条。读不到时返回一句说明而**不抛异常**——帮助页不该因为
     * 日志读失败而崩掉。
     */
    fun readTail(maxBytes: Int = 128 * 1024): String {
        val f = file ?: return "（日志尚未初始化）"
        return try {
            if (!f.exists()) return "（暂无日志）"
            val len = f.length()
            if (len <= maxBytes) f.readText()
            else {
                val bytes = f.readBytes()
                val tail = bytes.copyOfRange((len - maxBytes).toInt(), len.toInt())
                val start = tail.indexOfFirst { it == '\n'.code.toByte() }
                String(tail, Charsets.UTF_8).substring(if (start >= 0) start + 1 else 0)
            }
        } catch (e: Exception) {
            "（读取日志失败：${e.message}）"
        }
    }

    /** 常规信息（logcat + 文件）。 */
    fun i(msg: String) = write('I', msg, null)

    /** 错误：写完整堆栈（诊断报错必须带栈，否则只知道"失败了"不知道在哪）。 */
    fun e(msg: String, t: Throwable? = null) = write('E', msg, t)

    private fun installCrashHandler() {
        val prev = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread, throwable ->
            write('E', "UNCAUGHT on ${thread.name}", throwable)
            prev?.uncaughtException(thread, throwable)
        }
    }

    @Synchronized
    private fun write(level: Char, msg: String, t: Throwable?) {
        if (t == null) {
            if (level == 'E') Log.e(TAG, msg) else Log.i(TAG, msg)
        } else {
            Log.e(TAG, msg, t)
        }
        val f = file ?: return
        try {
            rotateIfNeeded(f)
            val sb = StringBuilder()
            sb.append(TIME.format(Date())).append(' ').append(level).append(' ')
                .append(msg).append('\n')
            if (t != null) {
                sb.append(t.stackTraceToString())
                var cause = t.cause
                var depth = 0
                while (cause != null && depth < 5) {          // 链式 cause
                    sb.append("Caused by: ").append(cause.stackTraceToString())
                    cause = cause.cause; depth++
                }
            }
            f.appendText(sb.toString())
        } catch (_: Exception) {
            // 日志失败不影响主流程
        }
    }

    /** 超限时保留后半段（丢最旧），而不是清空——一次溢出不该丢掉全部现场。 */
    private fun rotateIfNeeded(f: File) {
        if (f.length() <= MAX_BYTES) return
        val bytes = f.readBytes()
        val tail = bytes.copyOfRange(bytes.size - KEEP_BYTES, bytes.size)
        // 从下一个换行开始，避免首行被截成半条
        val start = tail.indexOfFirst { it == '\n'.code.toByte() }
        f.writeBytes(tail.copyOfRange(if (start >= 0) start + 1 else 0, tail.size))
    }
}