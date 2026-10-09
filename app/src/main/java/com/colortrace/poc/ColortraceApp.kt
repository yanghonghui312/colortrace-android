package com.colortrace.poc

import android.app.Application
import com.colortrace.DebugLog
import org.opencv.android.OpenCVLoader

class ColortraceApp : Application() {
    override fun onCreate() {
        super.onCreate()
        // P0 已证：本地初始化 + 5.0.0 AAR 与桌面 opencv-python 对拍一致
        check(OpenCVLoader.initLocal()) { "OpenCV native 加载失败" }
        // 文件日志（vivo logd 限流兜底）：files/colortrace.log
        DebugLog.init(this)
    }
}
