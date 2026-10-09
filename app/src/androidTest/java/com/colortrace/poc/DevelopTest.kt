package com.colortrace.poc

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.colortrace.engine.DevelopSpec
import com.colortrace.engine.DevelopTransform
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader

/**
 * develop（手动微调）移植对拍（P2.6）：桌面金标 develop_vectors.json，
 * 4 组参数案例（basic/color/split/sink 全 17 参数）。
 *
 * 容差协议：u8 max|Δ| ≤ 1/255 且 >1/255 的像素为 0——Kotlin 的 pow/sin/cos
 * 走 double 再截 float，与 numpy float32 的 powf/sinf 差 1 ulp 级，u8 边界
 * 舍入允许 ±1；与 P1Test e2e 的 region 关卡同协议。
 */
@RunWith(AndroidJUnit4::class)
class DevelopTest {

    private lateinit var report: JSONObject

    @Before
    fun setUp() {
        assertTrue("OpenCV native 加载失败", OpenCVLoader.initLocal())
        report = JSONObject()
        report.put("device", android.os.Build.MODEL)
        report.put("api", android.os.Build.VERSION.SDK_INT)
    }

    private fun u8Trunc(img: FloatArray): ByteArray {
        val out = ByteArray(img.size)
        for (i in out.indices) {
            out[i] = ((img[i].coerceIn(0f, 1f) * 255f + 0.5f).toInt()).toByte()
        }
        return out
    }

    @Test
    fun matchesDesktopGoldens() {
        val g = JSONObject(TestIO.assetText("goldens/develop_vectors.json"))
        val rgb = g.getString("input_rgb_f32").toF32()
        val cases = g.getJSONObject("cases")
        val details = JSONObject()

        val keys = cases.keys()
        while (keys.hasNext()) {
            val name = keys.next()
            val c = cases.getJSONObject(name)
            val paramsJson = c.getJSONObject("params")
            val params = HashMap<String, Float>()
            for (k in paramsJson.keys()) params[k] =
                paramsJson.getDouble(k).toFloat()

            val dev = DevelopTransform(params)
            val gotU8 = u8Trunc(dev.apply(rgb))
            val golden = java.util.Base64.getDecoder()
                .decode(c.getString("output_u8"))
            val maxD = TestIO.maxAbsDiff(gotU8, golden)
            val over1 = TestIO.countDiffOver(gotU8, golden, 1)
            details.put(name, JSONObject().put("max_abs", maxD)
                .put("px_over_1", over1))
            println("[develop] $name: max|Δ|=${maxD}/255 pxOver1=$over1/${golden.size}")
            assertTrue("$name max|Δ|=$maxD 超 1/255", maxD <= 1)
            assertTrue("$name 有 $over1 个像素差 >1/255（须为 0）", over1 == 0)
        }
        report.put("goldens", details)

        // identity 短路：全默认参数应原样返回（不进 LAB 往返）
        val identity = DevelopTransform(DevelopSpec.DEFAULTS)
        assertTrue("identity 未被识别", identity.isIdentity())
        val probe = FloatArray(300) { (it % 255) / 255f }
        val out = identity.apply(probe)
        assertTrue("identity apply 改变了输入",
                   out.contentEquals(probe))
        saveReport()
    }

    private fun saveReport() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        ctx.filesDir.mkdirs()
        ctx.openFileOutput("develop_report.json", android.content.Context.MODE_PRIVATE)
            .use { it.write(report.toString().toByteArray()) }
    }
}
