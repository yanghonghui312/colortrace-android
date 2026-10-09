package com.colortrace.poc

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.colortrace.engine.EncoderEngine
import com.colortrace.engine.EncoderPreset
import com.colortrace.engine.FlowWeights
import com.colortrace.engine.LabConv
import com.colortrace.engine.RegionEngine
import com.colortrace.engine.transfer
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertTrue
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

/**
 * P1 端到端对拍（2026-09-28）：Kotlin 引擎移植 ↔ 桌面金标。
 *
 * 分层锚：LAB（16³ 网格）、flow 传输（P0 金标的 16³ 探针，双侧 float64）；
 * 端到端：桌面真实 encoder 预设 × 256² 真实照片，三条产品路径的 u8 输出：
 *   plain / chroma(protect=1, 0.5) / region(protect=1, luma=global)。
 * 容差协议：u8 逐像素 max|Δ| ≤ 2/255（audit_equiv_check 同源关卡），
 * ≤1/255 的像素数同步报告；阈值首跑定标后收紧。
 */
@RunWith(AndroidJUnit4::class)
class P1Test {

    private lateinit var report: JSONObject

    @Before
    fun setUp() {
        assertTrue("OpenCV native 加载失败", OpenCVLoader.initLocal())
        report = JSONObject()
        report.put("opencv", OpenCVLoader.OPENCV_VERSION)
        report.put("device", android.os.Build.MODEL)
        report.put("api", android.os.Build.VERSION.SDK_INT)
    }

    // ---- 分层锚 1：LAB ----

    @Test
    fun labMatchesDesktop() {
        val g = JSONObject(TestIO.assetText("goldens/p1_lab.json"))
        val rgb = g.getString("rgb_f32").toF32()
        val golden = g.getString("lab_f32").toF32()
        val got = LabConv.rgbToLab(rgb)
        val maxAbs = LabConv.maxAbsDiff(got, golden)
        report.put("lab_max_abs", maxAbs.toDouble())
        println("[p1] LAB grid: max|Δ|=$maxAbs")
        // 幂/cbrt 走 double 再取 float 与 numpy float32 powf 差 ~1 ulp
        // （首跑实测 9.2e-05）；5e-4 留 5× 裕量。
        assertTrue("LAB 偏差过大 max|Δ|=$maxAbs", maxAbs < 5e-4f)
    }

    // ---- 分层锚 2：flow 传输（双侧 float64） ----

    @Test
    fun flowTransferMatchesDesktop() {
        val g = JSONObject(TestIO.assetText("goldens/encoder_poc.json"))
        val flat = g.getString("output_flat_f32").toF32()
        val probes = g.getString("transfer_probes_f32").toF32()
        val golden = g.getString("transfer_out_f32").toF32()
        val hidden = g.getJSONObject("meta").getInt("hidden")
        val flatD = DoubleArray(flat.size) { flat[it].toDouble() }
        val half = flatD.size / 2
        val wC = FlowWeights.fromFlat(flatD.copyOfRange(0, half), hidden)
        val wS = FlowWeights.fromFlat(flatD.copyOfRange(half, flatD.size), hidden)
        val got = FloatArray(golden.size)
        for (p in 0 until golden.size / 3) {
            val x = doubleArrayOf(probes[3 * p].toDouble(),
                                  probes[3 * p + 1].toDouble(),
                                  probes[3 * p + 2].toDouble())
            transfer(x, wC, wS, 1.0, 1)
            for (c in 0 until 3) got[3 * p + c] = x[c].toFloat()
        }
        val maxAbs = LabConv.maxAbsDiff(got, golden)
        report.put("flow_max_abs", maxAbs.toDouble())
        println("[p1] flow transfer: max|Δ|=$maxAbs")
        // 双侧都是 float64 单步 Euler → 残差应为 1e-15 量级（矩阵乘累加序差异）
        assertTrue("flow 传输偏差过大 max|Δ|=$maxAbs", maxAbs < 1e-9)
    }

    // ---- 端到端：三条产品路径 ----

    @Test
    fun endToEndMatchesDesktop() {
        val g = JSONObject(TestIO.assetText("goldens/p1_e2e.json"))
        val side = g.getInt("side")
        val preset = EncoderPreset(TestIO.assetText("goldens/encoder_preset.json"))
        val engine = EncoderEngine(preset, TestIO.assetBytes(preset.onnxName))
        val selfie = readNetFromAsset("selfie_multiclass.onnx")
        val (content, w) = TestIO.decodePngAsset("goldens/p1_content.png")
        require(w == side)

        // 分层锚：内容缩略图（Θc 的输入，桌面 INTER_AREA 应逐位一致）
        val thumb = engine.thumbOf(content, side, side)
        val thumbGolden = g.getString("thumb_f32").toF32()
        val thumbMax = LabConv.maxAbsDiff(thumb, thumbGolden)
        report.put("thumb_max_abs", thumbMax.toDouble())
        println("[p1] thumb: max|Δ|=$thumbMax")
        assertTrue("内容缩略图偏差过大 max|Δ|=$thumbMax", thumbMax < 1e-6f)

        val region = RegionEngine(engine, selfie, 1.0f)
        val cases = mapOf(
            "plain" to engine.apply(content, side, side),
            "chroma100" to engine.applyChroma(content, side, side, 1.0f),
            "chroma50" to engine.applyChroma(content, side, side, 0.5f),
            "region100" to region.apply(content, side, side),
        )
        val details = JSONObject()
        for ((name, img) in cases) {
            val golden = TestIO.pngAssetToU8("goldens/p1_out_${name}.png")
            val got = TestIO.toU8Round(img)
            val maxD = TestIO.maxAbsDiff(got, golden)
            val over1 = TestIO.countDiffOver(got, golden, 1)
            details.put(name, JSONObject()
                .put("max_abs", maxD)
                .put("px_over_1", over1))
            println("[p1] e2e $name: max|Δ|=${maxD}/255 pxOver1=$over1/${golden.size}")
            // 首跑实测：plain/chroma50/chroma100 全部 0/255（逐位），region100
            // max=1（分割概率 fp32 噪声的 u8 边界舍入）、无像素超 1 ⇒
            // 收紧为 max ≤ 1 且 0 像素 >1。
            assertTrue("$name max|Δ|=$maxD 超 1/255", maxD <= 1)
            assertTrue("$name 有 $over1 个像素差 >1/255（须为 0）", over1 == 0)
        }
        report.put("e2e", details)
    }

    private fun readNetFromAsset(name: String): Net {
        val bytes = TestIO.assetBytes(name)
        val m = Mat(1, bytes.size, CvType.CV_8U)
        m.put(0, 0, bytes)
        return Dnn.readNetFromONNX(MatOfByte(m))
    }

    @After
    fun tearDown() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        runCatching {
            File(ctx.filesDir, "p1_report.json").writeText(report.toString(2))
        }
        Log.i("P1Test", "report=$report")
    }
}
