package com.colortrace.poc

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.After
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.dnn.Dnn
import org.opencv.dnn.Net
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Base64
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max

/**
 * 安卓 P0 对拍 PoC（2026-09-28）：cv2.dnn 在安卓上跑 encoder / selfie 两个
 * ONNX，与桌面金标对拍。金标由 `scripts/export_android_poc_vectors.py`
 * 生成（桌面 opencv 5.0.0 / numpy 2.5.3），打包在本测试 APK 的
 * assets 下（goldens 目录的 JSON + 仓库根 .models 的 ONNX）。
 *
 * 对拍协议（与生成器同源）：
 *  - 输入 blob 直接读金标字节——隔离预处理差异（预处理对拍是 P1）；
 *  - encoder：前向 flat (2054,)——报告 max|Δ| 与 max 相对差，容差断言；
 *  - selfie：softmax/argmax 在移动端复现（与 semantic.py 同式），
 *    类图必须**逐位一致**；32² 采样概率与 6 类概率和给容差。
 *
 * 数值同时 println 与写 targetContext 外部文件 poc_report.json
 * （adb pull 出报告）。容差常数按首跑实测值收紧（首跑上限留裕量）。
 */
@RunWith(AndroidJUnit4::class)
class PocTest {

    private lateinit var report: JSONObject

    @Before
    fun setUp() {
        assertTrue("OpenCV native 加载失败", OpenCVLoader.initLocal())
        report = JSONObject()
        report.put("opencv", OpenCVLoader.OPENCV_VERSION)
        report.put("device", android.os.Build.MODEL)
        report.put("api", android.os.Build.VERSION.SDK_INT)
    }

    // ---- 工具：assets 与字节解码 ----

    private fun assetText(name: String): String =
        InstrumentationRegistry.getInstrumentation().context.assets
            .open(name).bufferedReader().use { it.readText() }

    /** ONNX 打包在 assets 根（.models srcDir）——读字节经 MatOfByte 构网，
     *  完全不落盘（插桩上下文的数据目录初始化不可靠，ENOSPC/ENOENT 坑）。 */
    private fun readNetFromAsset(name: String): Net {
        val ctx = InstrumentationRegistry.getInstrumentation().context
        val bytes = ctx.assets.open(name).use { it.readBytes() }
        val m = Mat(1, bytes.size, CvType.CV_8U)
        m.put(0, 0, bytes)
        return Dnn.readNetFromONNX(MatOfByte(m))
    }

    private fun String.toF32(): FloatArray {
        val bytes = Base64.getDecoder().decode(this)
        val fb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(bytes.size / 4).also { fb.get(it) }
    }

    private fun String.toU8(): ByteArray = Base64.getDecoder().decode(this)

    private fun String.toF64(): DoubleArray {
        val bytes = Base64.getDecoder().decode(this)
        val db = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asDoubleBuffer()
        return DoubleArray(bytes.size / 8).also { db.get(it) }
    }

    private fun maxAbsDiff(a: FloatArray, b: FloatArray): Float =
        a.indices.maxOf { abs(a[it] - b[it]) }

    // ---- encoder：前向 flat 对拍（P0 核心） ----

    @Test
    fun encoderForwardMatchesDesktop() {
        val g = JSONObject(assetText("goldens/encoder_poc.json"))
        val input = g.getString("input_blob_f32").toF32()
        val golden = g.getString("output_flat_f32").toF32()
        assertEquals(golden.size, g.getJSONObject("meta").getInt("flat_size"))

        // blob 构造用「2D 连续写入 + reshape(int[],) 视图」：4D Mat 上直接
        // put(0,0) 的寻址不可靠（实测输入被放歪、前向完全对不上），2D put
        // 与 reshape 均只共享内存、不改数据。
        val flatIn = Mat(1, input.size, CvType.CV_32F)
        flatIn.put(0, 0, input)
        val net = readNetFromAsset("encoder_cnn128.onnx")
        net.setInput(flatIn.reshape(1, intArrayOf(1, 3, 128, 128)))
        val out = net.forward().reshape(1, 1)
        val got = FloatArray(out.cols())
        out.get(0, 0, got)

        assertEquals("输出元素数不符", golden.size, got.size)
        val maxAbs = maxAbsDiff(got, golden)
        val maxRel = got.indices.maxOf {
            abs(got[it] - golden[it]) / max(1e-9f, abs(golden[it]))
        }
        report.put("encoder_max_abs", maxAbs.toDouble())
        report.put("encoder_max_rel", maxRel.toDouble())
        println("[poc] encoder forward: max|Δ|=$maxAbs maxRel=$maxRel")
        // 首跑实测（AEHD x86_64 模拟器）：max|Δ|=8.9e-08 —— 1e-5 留 100× 裕量。
        // maxRel 在权重近 0 的元素上会放大（~3.6e-4），不作断言口径。
        assertTrue("encoder 前向偏差过大 max|Δ|=$maxAbs", maxAbs < 1e-5f)
    }

    // ---- selfie：类图逐位 + 采样概率 + 概率和 ----

    @Test
    fun selfieSegmentationMatchesDesktop() {
        val g = JSONObject(assetText("goldens/selfie_poc.json"))
        val input = g.getString("input_png_u8").toU8()   // 256×256×3 u8 (RGB)
        val goldenCls = g.getString("cls_map_u8").toU8()
        val ys = g.getJSONArray("sample_ys").let { arr ->
            IntArray(arr.length()) { arr.getInt(it) }
        }
        val goldenProbs = g.getString("sample_probs_f32").toF32() // 32×32×6
        val goldenSums = g.getString("class_prob_sums_f64").toF64()
        val n = 256 * 256
        val nGrid = ys.size * ys.size

        // 与 semantic.py 同式：u8 → /255 → (1,256,256,3) NHWC float32
        // （256×256 的 CV_32FC3 Mat 内存布局与之相同）
        val f32 = FloatArray(n * 3)
        for (i in 0 until n) {
            f32[3 * i] = (input[3 * i].toInt() and 0xFF) / 255f
            f32[3 * i + 1] = (input[3 * i + 1].toInt() and 0xFF) / 255f
            f32[3 * i + 2] = (input[3 * i + 2].toInt() and 0xFF) / 255f
        }
        // NHWC 4D blob（与桌面 (1,256,256,3) 同构）：2D 连续写入 + reshape 视图
        val flatIn = Mat(1, f32.size, CvType.CV_32F)
        flatIn.put(0, 0, f32)
        val net = readNetFromAsset("selfie_multiclass.onnx")
        net.setInput(flatIn.reshape(1, intArrayOf(1, 256, 256, 3)))
        val out = net.forward().reshape(1, 256)          // 256 行 × (256*6) 列
        val logits = FloatArray(out.rows() * out.cols())
        out.get(0, 0, logits)
        assertEquals(256 * 256 * 6, logits.size)

        // softmax + argmax（与 semantic.py 逐式对应：减最大值 → exp → 归一；
        // argmax 平票取低类号——与 numpy argmax 同规则）。采样行按网格顺序
        // 写 gotGrid（行主序 yy 大、xx 小，与金标 np.ix_ 一致）。
        val ysSet = ys.toSet()
        val cls = ByteArray(n)
        val gotGrid = FloatArray(nGrid * 6)
        val sums = DoubleArray(6)
        val e = FloatArray(6)
        var k = 0
        for (row in 0 until 256) {
            val rowSampled = row in ysSet
            val yy = if (rowSampled) ys.indexOf(row) else -1
            for (col in 0 until 256) {
                var m = Float.NEGATIVE_INFINITY
                for (c in 0 until 6) m = max(m, logits[k + c])
                var s = 0f
                for (c in 0 until 6) { e[c] = exp(logits[k + c] - m); s += e[c] }
                var best = 0
                var bv = 0f
                for (c in 0 until 6) {
                    val pr = e[c] / s
                    sums[c] += pr.toDouble()
                    if (pr > bv) { bv = pr; best = c }
                }
                cls[row * 256 + col] = best.toByte()
                if (rowSampled && col in ysSet) {
                    val xx = ys.indexOf(col)
                    for (c in 0 until 6) gotGrid[(yy * ys.size + xx) * 6 + c] = e[c] / s
                }
                k += 6
            }
        }

        val clsMismatch = cls.indices.count { cls[it] != goldenCls[it] }
        val probsMaxAbs = maxAbsDiff(gotGrid, goldenProbs)
        val sumsRel = sums.indices.maxOf {
            abs(sums[it] - goldenSums[it]) / max(1e-9, abs(goldenSums[it]))
        }
        report.put("selfie_cls_mismatch", clsMismatch)
        report.put("selfie_probs_max_abs", probsMaxAbs.toDouble())
        report.put("selfie_sums_max_rel", sumsRel)
        println("[poc] selfie: clsMismatch=$clsMismatch/$n " +
                "probsMax|Δ|=$probsMaxAbs sumsMaxRel=$sumsRel")

        assertArrayEquals("类图必须与桌面逐位一致", goldenCls, cls)
        // 首跑实测：probsMax|Δ|=1.79e-06 ⇒ 1e-4 留 50× 裕量。
        // sums：金标 float64 求和后实测相对差 9.0e-08——这是 fp32 前向噪声
        // 经 65536 像素求和的 √N 放大（逐像素 ~1e-6 → 和 ~1e-7），不可再低；
        // 1e-6 留 10× 裕量，仍足以抓住类别互换/归一化错误这类真 bug。
        assertTrue("采样概率偏差过大 max|Δ|=$probsMaxAbs", probsMaxAbs < 1e-4f)
        assertTrue("类概率和相对偏差过大 $sumsRel", sumsRel < 1e-6)
    }

    @After
    fun tearDown() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        runCatching {
            val dir = ctx.filesDir                  // 外部目录在 headless 下不可靠
            File(dir, "poc_report.json").writeText(report.toString(2))
        }
        Log.i("PocTest", "report=$report")
    }
}
