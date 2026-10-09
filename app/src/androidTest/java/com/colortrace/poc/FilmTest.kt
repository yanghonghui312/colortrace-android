package com.colortrace.poc

import android.graphics.Bitmap
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.colortrace.engine.FilmLayer
import com.colortrace.engine.NativeKernels
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.core.CvType
import org.opencv.core.Mat

/**
 * 胶片质感层（FilmLayer + film_kernels.cpp）的关卡（2026-10-02）。
 *
 * 口径（用户拍板）：柔光/光晕与桌面同式 ⇒ u8 ≤1/255 金标对拍（film_out_*.png，
 * 桌面 `scripts/export_android_film_vectors.py` 产）；**颗粒不与桌面逐位**
 * （两端各自确定性、算法同构）⇒ 端内锚：同参逐位可复现、幅度 = σ·数量、
 * 大小参数单调改变噪声尺度。
 */
@RunWith(AndroidJUnit4::class)
class FilmTest {

    @Before
    fun setUp() {
        assertTrue("OpenCV native 加载失败", OpenCVLoader.initLocal())
    }

    /** 内容图 → f32 Mat（CV_32FC3 RGB）。 */
    private fun contentMat(): Pair<Mat, Int> {
        val (f32, side) = TestIO.decodePngAsset("goldens/film_content.png")
        val m = Mat(side, side, CvType.CV_32FC3)
        m.put(0, 0, f32)
        return Pair(m, side)
    }

    private fun matToU8(m: Mat): ByteArray {
        val f = FloatArray(m.cols() * m.rows() * 3)
        m.get(0, 0, f)
        return TestIO.toU8Round(f)
    }

    /** 模糊并行性探针：1920×1080 f32 σ=25 全图 GaussianBlur 计时（单次+3 次均值）。
     *  若 >1s 说明 OpenCV Android 的 blur 未并行（单线程），决定优化路线。 */
    @Test
    fun blurTimingProbe() {
        val w = 1920
        val h = 1080
        val m = Mat(h, w, CvType.CV_32FC3)
        val buf = FloatArray(w * h * 3)
        java.util.Random(7L).let { r -> for (i in buf.indices) buf[i] = r.nextFloat() }
        m.put(0, 0, buf)
        val dst = Mat()
        org.opencv.imgproc.Imgproc.GaussianBlur(m, dst,
            org.opencv.core.Size(0.0, 0.0), 25.0, 25.0)
        val t = System.nanoTime()
        repeat(3) {
            org.opencv.imgproc.Imgproc.GaussianBlur(m, dst,
                org.opencv.core.Size(0.0, 0.0), 25.0, 25.0)
        }
        val ms = (System.nanoTime() - t) / 3e6
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        java.io.File(ctx.filesDir, "film_probe.txt")
            .writeText("blur 1920x1080 s=25: ${"%.0f".format(ms)}ms/次")
        dst.release(); m.release()
    }

    /**
     * 半分辨率模糊 A/B 探针（2026-10-02 决策用，不进任何断言）：
     * 预压缩 2880×1920 与原始 4096×3072 两档，对比
     *   A 全分辨率 GaussianBlur（现状）
     *   B 1/2：INTER_AREA 缩半 → blur(σ/2) → INTER_LINEAR 放回（耗时 + 与 A 的差异）
     *   C 1/4：同构
     * 结果写 files/film_probe.txt。输出与 A 不同（低频近似）——只测耗时与差量，
     * 是否采用待用户拍板（需桌面端同步改口径 + 金标重锚）。
     */
    @Test
    fun halfResBlurProbe() {
        val sb = StringBuilder()
        for ((w, h) in listOf(2880 to 1920, 4096 to 3072)) {
            val m = Mat(h, w, CvType.CV_32FC3)
            val buf = FloatArray(w * h * 3)
            val r = java.util.Random(9L)
            for (i in buf.indices) buf[i] = r.nextFloat()
            m.put(0, 0, buf)
            val sigma = minOf(w, h) * 0.013      // 柔光默认档（radius 50）
            val dst = Mat()

            fun bench(name: String, times: Int, block: () -> Unit): Double {
                block()
                val t = System.nanoTime()
                repeat(times) { block() }
                return (System.nanoTime() - t) / 1e6 / times
            }

            val full = bench("full", 3) {
                org.opencv.imgproc.Imgproc.GaussianBlur(
                    m, dst, org.opencv.core.Size(0.0, 0.0), sigma, sigma)
            }
            val refMat = dst.clone()          // 差异基准留 native 内存（Java 堆放不下全图数组）

            val half = bench("half", 3) {
                val sm = Mat(); val bd = Mat()
                org.opencv.imgproc.Imgproc.resize(
                    m, sm, org.opencv.core.Size(0.0, 0.0), 0.5, 0.5,
                    org.opencv.imgproc.Imgproc.INTER_AREA)
                org.opencv.imgproc.Imgproc.GaussianBlur(
                    sm, bd, org.opencv.core.Size(0.0, 0.0), sigma / 2, sigma / 2)
                org.opencv.imgproc.Imgproc.resize(
                    bd, dst, org.opencv.core.Size(w.toDouble(), h.toDouble()),
                    0.0, 0.0, org.opencv.imgproc.Imgproc.INTER_LINEAR)
                sm.release(); bd.release()
            }
            var sd = 0.0; var mx = 0f; var cnt = 0L
            run {
                val rows = 256
                val b1 = FloatArray(w * rows * 3)
                val b2 = FloatArray(w * rows * 3)
                for (y0 in 0 until h step rows) {
                    val r = minOf(rows, h - y0)
                    val n = w * r * 3
                    refMat.rowRange(y0, y0 + r).get(0, 0, b1)
                    dst.rowRange(y0, y0 + r).get(0, 0, b2)
                    for (i in 0 until n) {
                        val d = kotlin.math.abs(b1[i] - b2[i])
                        sd += d; if (d > mx) mx = d
                    }
                    cnt += n
                }
            }
            val meanHalf = sd / cnt

            val quarter = bench("quarter", 3) {
                val sm = Mat(); val bd = Mat()
                org.opencv.imgproc.Imgproc.resize(
                    m, sm, org.opencv.core.Size(0.0, 0.0), 0.25, 0.25,
                    org.opencv.imgproc.Imgproc.INTER_AREA)
                org.opencv.imgproc.Imgproc.GaussianBlur(
                    sm, bd, org.opencv.core.Size(0.0, 0.0), sigma / 4, sigma / 4)
                org.opencv.imgproc.Imgproc.resize(
                    bd, dst, org.opencv.core.Size(w.toDouble(), h.toDouble()),
                    0.0, 0.0, org.opencv.imgproc.Imgproc.INTER_LINEAR)
                sm.release(); bd.release()
            }
            sd = 0.0; mx = 0f
            run {
                val rows = 256
                val b1 = FloatArray(w * rows * 3)
                val b2 = FloatArray(w * rows * 3)
                for (y0 in 0 until h step rows) {
                    val r = minOf(rows, h - y0)
                    val n = w * r * 3
                    refMat.rowRange(y0, y0 + r).get(0, 0, b1)
                    dst.rowRange(y0, y0 + r).get(0, 0, b2)
                    for (i in 0 until n) {
                        val d = kotlin.math.abs(b1[i] - b2[i])
                        sd += d; if (d > mx) mx = d
                    }
                    cnt += n
                }
            }
            refMat.release()
            sb.append("${w}x${h} sigma=${"%.1f".format(sigma)}: " +
                    "full=${"%.0f".format(full)}ms " +
                    "half=${"%.0f".format(half)}ms(meanD=${"%.5f".format(meanHalf)} maxD=$mx) " +
                    "quarter=${"%.0f".format(quarter)}ms\n")
            dst.release(); m.release()
            dst.release(); m.release()
        }
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        java.io.File(ctx.filesDir, "film_probe.txt").writeText(sb.toString())
    }

    /** 柔光/光晕与桌面金标逐像素对拍（全部档；u8 ≤1/255）。 */
    @Test
    fun softBloomMatchesDesktop() {
        val g = JSONObject(TestIO.assetText("goldens/film_soft_bloom.json"))
        val (src, side) = contentMat()
        src.release()
        val fails = mutableListOf<String>()
        for (name in g.getJSONObject("cases").keys()) {
            val params = mutableMapOf<String, Float>()
            val cj = g.getJSONObject("cases").getJSONObject(name)
            for (k in cj.keys()) params[k] = cj.getDouble(k).toFloat()
            params["grain_on"] = 0f
            val (input, _) = contentMat()
            assertTrue("FilmLayer.apply 失败 ($name)", FilmLayer.apply(input, params))
            val got = matToU8(input)
            input.release()
            val want = TestIO.pngAssetToU8("goldens/film_out_$name.png")
            val d = TestIO.maxAbsDiff(got, want)
            val over = TestIO.countDiffOver(got, want, 1)
            val wm = want.map { it.toInt() and 0xFF }.average()
            var first = -1
            var zeros = 0
            for (i in got.indices) {
                val g = got[i].toInt() and 0xFF
                if (g == 0) zeros++
                if (first < 0 && kotlin.math.abs(got[i] - want[i]) > 1) first = i
            }
            println("[film] $name max|Δ|=$d 超容差=$over/${got.size} " +
                    "want均值=${"%.2f".format(wm)} first=$first " +
                    "got=${if (first >= 0) got[first].toInt() and 0xFF else 0} " +
                    "want=${if (first >= 0) want[first].toInt() and 0xFF else 0} " +
                    "零值=$zeros")
            if (d > 1) fails.add("$name max|Δ|=$d 超=$over")
        }
        assertTrue("金标对拍失败: $fails", fails.isEmpty())
    }

    /** 颗粒端内锚：同图同参**逐位可复现**；幅度 = σ·数量；大小参数单调改尺度。 */
    @Test
    fun grainDeterministicAndStats() {
        val side = 128
        val params = FilmLayer.DEFAULTS +
                mapOf("grain_on" to 1f, "grain_amount" to 35f,
                      "grain_rough" to 45f, "grain_size" to 30f)
        fun run(): Pair<FloatArray, FloatArray> {
            val (m, _) = contentMat()
            val f = FloatArray(side * side * 3).also { m.get(0, 0, it) }
            m.release()
            // 用小图独立构造（128²），避免依赖金标内容图尺寸
            val mm = Mat(side, side, CvType.CV_32FC3)
            mm.put(0, 0, f)
            assertTrue(FilmLayer.apply(mm, params))
            val out = FloatArray(side * side * 3).also { mm.get(0, 0, it) }
            mm.release()
            return Pair(f, out)
        }
        val (_, out1) = run()
        val (_, out2) = run()
        // 端内确定性：噪声种子由 (h,w,粗糙,大小) 派生 ⇒ 同参两次逐位一致
        assertEquals("颗粒必须端内逐位可复现", 0, (0 until out1.size).count { out1[it] != out2[it] })

        // 幅度锚：native 噪声序列经归一后 std ≈ σ·数量（σ=2.2，数量 0.35）
        val n = 1 shl 16
        val noise = FloatArray(n)
        NativeKernels.filmGrainBase(42L, n, noise)
        NativeKernels.filmGrainNormalize(noise, 2.2f * 0.35f)
        var s = 0.0
        for (v in noise) s += v.toDouble()
        val mean = s / n
        var sq = 0.0
        for (v in noise) { val d = v.toDouble() - mean; sq += d * d }
        val std = kotlin.math.sqrt(sq / n)
        println("[film] grain std=${"%.4f".format(std)} 期望=${"%.4f".format(2.2f * 0.35f)}")
        assertEquals("归一后幅度必须 = σ·数量",
                     2.2 * 0.35, std, 0.01)

        // 大小单调：size 越大（粗网格越粗）→ 相邻像素越相关
        fun corr(size: Float): Double {
            val k = 1.0 + 3.0 * (size / 100.0) * (size / 100.0)
            val hk = maxOf(2, kotlin.math.ceil(side / k).toInt())
            val wk = maxOf(2, kotlin.math.ceil(side / k).toInt())
            val base = FloatArray(hk * wk)
            NativeKernels.filmGrainBase(7L, base.size, base)
            // 近似用 base 的行内相关（上采样 INTER_LINEAR 保线性插值 ⇒ 相关方向一致）
            var c00 = 0.0; var c01 = 0.0; var a0 = 0.0; var a1 = 0.0
            for (y in 0 until hk) for (x in 0 until wk - 1) {
                val v0 = base[y * wk + x].toDouble()
                val v1 = base[y * wk + x + 1].toDouble()
                c00 += v0 * v0; c01 += v0 * v1; a0 += v0; a1 += v1
            }
            val cnt = (hk * (wk - 1)).toDouble()
            val m0 = a0 / cnt
            val cov = c01 / cnt - m0 * (a1 / cnt)
            val v0 = c00 / cnt - m0 * m0
            return if (v0 <= 0.0) 0.0 else cov / kotlin.math.sqrt(v0 * v0)
        }
        val fine = corr(10f)
        val coarse = corr(100f)
        println("[film] 行内相关 fine=$fine coarse=$coarse")
        assertTrue("大小必须单调改变噪声尺度", coarse > fine)
    }

    /** film 关闭（全默认）时渲染路径零变化：render(film=null) ≡ render(全默认)。 */
    @Test
    fun filmOffPathUnchanged() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val repo = EngineRepository(ctx)
        val bmpIn = TestIO.decodePngAsset("goldens/film_content.png").let { (f, side) ->
            val px = IntArray(side * side)
            for (p in px.indices) {
                px[p] = (0xFF shl 24) or ((f[3 * p] * 255f + 0.5f).toInt() shl 16) or
                        ((f[3 * p + 1] * 255f + 0.5f).toInt() shl 8) or
                        (f[3 * p + 2] * 255f + 0.5f).toInt()
            }
            Bitmap.createBitmap(px, side, side, Bitmap.Config.ARGB_8888)
        }
        val session = repo.fitSample(bmpIn, "reinhard")
        val pipe = TiledPipeline(session.model, null)
        val m = pipe.migrate(bmpIn, ProtectMode.OFF)
        val off = pipe.renderResult(m, 1f)
        val def = pipe.renderResult(m, 1f, film = FilmLayer.DEFAULTS)
        // render 状态链路：film 关闭（或成功）时 filmApplied 必须为 true
        assertTrue("关闭路径 filmApplied 必须为 true",
                   off.filmApplied && def.filmApplied)
        assertEquals("关闭路径 w 不变", off.bitmap.width, def.bitmap.width)
        assertEquals("关闭路径 h 不变", off.bitmap.height, def.bitmap.height)
        val a = IntArray(off.bitmap.width * off.bitmap.height)
        off.bitmap.getPixels(a, 0, off.bitmap.width, 0, 0, off.bitmap.width, off.bitmap.height)
        val b = IntArray(def.bitmap.width * def.bitmap.height)
        def.bitmap.getPixels(b, 0, def.bitmap.width, 0, 0, def.bitmap.width, def.bitmap.height)
        assertEquals("全默认（开关全关）必须与 null 路径逐位一致", 0,
                     (0 until a.size).count { a[it] != b[it] })
        m.release()
    }

    /** E2E 冒烟：三效果全开导出成功、尺寸正确、时长日志可读。 */
    @Test
    fun filmFullPipelineSmoke() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val repo = EngineRepository(ctx)
        val side = 256
        val bmpIn = TestIO.decodePngAsset("goldens/film_content.png").let { (f, s0) ->
            val px = IntArray(s0 * s0)
            for (p in px.indices) {
                px[p] = (0xFF shl 24) or ((f[3 * p] * 255f + 0.5f).toInt() shl 16) or
                        ((f[3 * p + 1] * 255f + 0.5f).toInt() shl 8) or
                        (f[3 * p + 2] * 255f + 0.5f).toInt()
            }
            Bitmap.createBitmap(px, s0, s0, Bitmap.Config.ARGB_8888)
        }
        val session = repo.fitSample(bmpIn, "reinhard")
        val full = FilmLayer.DEFAULTS + mapOf(
            "soft_on" to 1f, "bloom_on" to 1f, "grain_on" to 1f)
        val ex = TiledPipeline(session.model, null).processToJpeg(
            bmpIn, ProtectMode.OFF, 1f, null, 1f, film = full)
        assertEquals("导出宽不变", side, ex.w)
        assertEquals("导出高不变", side, ex.h)
        assertTrue("JPEG 字节非空", ex.bytes.size > 1000)
        assertTrue("三效果全开必须应用（JpegExport.filmApplied 状态链路）", ex.filmApplied)
        println("[film] E2E 导出 ${ex.w}x${ex.h} ${ex.bytes.size}B")
    }

    /**
     * Mat 数据指针路径（2026-10-03 生产路径）与数组路径（金标基准）**逐位**等价。
     * 金标继续锁数组版，这里锁指针版到数组版 ⇒ 指针版传递等价于桌面。每个
     * 指针版内核都在**带行偏移**的窗口上对拍（覆盖 y0/a 行窗算术与就地写回）。
     */
    @Test
    fun filmMatWrappersMatchArray() {
        val rnd = java.util.Random(21L)
        fun randF(sz: Int) = FloatArray(sz).also {
            for (i in it.indices) it[i] = rnd.nextFloat() * 0.95f + 0.025f
        }
        fun assertBitsEq(name: String, want: FloatArray, got: FloatArray) {
            var bad = -1
            for (i in want.indices) {
                if (java.lang.Float.floatToRawIntBits(want[i]) !=
                    java.lang.Float.floatToRawIntBits(got[i])) { bad = i; break }
            }
            assertTrue("$name 指针版与数组版逐位不一致 first=$bad", bad < 0)
        }

        // —— 柔光 / 光晕·混合：就地写回 + (y0, a) 双行偏移 ——
        for (case in listOf("soft", "bloomApply")) {
            val w = 61; val rows = 37; val y0 = 13; val a = 5
            val mat = Mat(rows + y0, w, CvType.CV_32FC3)
            mat.put(0, 0, randF((rows + y0) * w * 3))
            val blur = Mat(rows + a, w, CvType.CV_32FC3)
            blur.put(0, 0, randF((rows + a) * w * 3))
            val matArr = FloatArray((rows + y0) * w * 3).also { mat.get(0, 0, it) }
            val blurArr = FloatArray((rows + a) * w * 3).also { blur.get(0, 0, it) }
            val rgbA = matArr.copyOfRange(y0 * w * 3, matArr.size)
            val blurA = blurArr.copyOfRange(a * w * 3, blurArr.size)
            val want = FloatArray(rows * w * 3)
            if (case == "soft") {
                NativeKernels.filmSoftBand(rgbA, blurA, 0.55f, 0.31f, 0.18f, want)
                NativeKernels.filmSoftBandMat(mat.dataAddr(), y0, blur.dataAddr(), a,
                                              rows, w, 0.55f, 0.31f, 0.18f)
            } else {
                NativeKernels.filmBloomApply(rgbA, blurA, 0.6f, 0.4f, want)
                NativeKernels.filmBloomApplyMat(mat.dataAddr(), y0, blur.dataAddr(), a,
                                                rows, w, 0.6f, 0.4f)
            }
            val got = FloatArray((rows + y0) * w * 3).also { mat.get(0, 0, it) }
            assertBitsEq(case, want, got.copyOfRange(y0 * w * 3, got.size))
            mat.release(); blur.release()
        }

        // —— 光晕·高光提取：写 masked/mask，yA 偏移 ——
        run {
            val w = 41; val rows = 23; val yA = 11
            val rgb = Mat(rows + yA, w, CvType.CV_32FC3)
            rgb.put(0, 0, randF((rows + yA) * w * 3))
            val rgbArr = FloatArray((rows + yA) * w * 3).also { rgb.get(0, 0, it) }
            val sub = rgbArr.copyOfRange(yA * w * 3, rgbArr.size)
            val wantMasked = FloatArray(rows * w * 3)
            val wantMask = FloatArray(rows * w)
            NativeKernels.filmBloomMask(sub, wantMasked, wantMask)
            val masked = Mat(rows, w, CvType.CV_32FC3)
            val mask = Mat(rows, w, CvType.CV_32FC1)
            NativeKernels.filmBloomMaskMat(rgb.dataAddr(), yA, masked.dataAddr(),
                                           mask.dataAddr(), rows, w)
            assertBitsEq("bloomMask.masked", wantMasked,
                         FloatArray(rows * w * 3).also { masked.get(0, 0, it) })
            assertBitsEq("bloomMask.mask", wantMask,
                         FloatArray(rows * w).also { mask.get(0, 0, it) })
            rgb.release(); masked.release(); mask.release()
        }

        // —— 颗粒 base / scale / bands / normalize：同参数两版逐位 ——
        run {
            val hk = 9; val wk = 13
            val want = FloatArray(hk * wk)
            NativeKernels.filmGrainBase(12345L, hk * wk, want)
            val bm = Mat(hk, wk, CvType.CV_32FC1)
            NativeKernels.filmGrainBaseMat(12345L, bm.dataAddr(), hk * wk)
            assertBitsEq("grainBase", want, FloatArray(hk * wk).also { bm.get(0, 0, it) })
            bm.release()

            val sn = 1 shl 14
            val s1 = randF(sn)
            val s1m = Mat(1, sn, CvType.CV_32FC1).also { it.put(0, 0, s1.copyOf()) }
            val wantS = s1.copyOf()
            for (i in 0 until sn) wantS[i] *= 0.65f
            NativeKernels.filmGrainScaleMat(s1m.dataAddr(), sn, 0.65f)
            assertBitsEq("grainScale", wantS, FloatArray(sn).also { s1m.get(0, 0, it) })
            s1m.release()

            val bw = 97; val bh = 2048; val fine = 0.35f * 0.45f
            val b1 = randF(bw * bh)
            val b2m = Mat(bh, bw, CvType.CV_32FC1).also { it.put(0, 0, b1.copyOf()) }
            NativeKernels.filmGrainBands(777L, 2, 1024, bw, bh, fine, b1)
            NativeKernels.filmGrainBandsMat(777L, 2, 1024, bw, bh, fine, b2m.dataAddr())
            assertBitsEq("grainBands", b1, FloatArray(bw * bh).also { b2m.get(0, 0, it) })
            b2m.release()

            val nz = randF(1 shl 16)
            val nzm = Mat(1, nz.size, CvType.CV_32FC1).also { it.put(0, 0, nz.copyOf()) }
            NativeKernels.filmGrainNormalize(nz, 0.77f)
            NativeKernels.filmGrainNormalizeMat(nzm.dataAddr(), nz.size, 0.77f)
            assertBitsEq("grainNormalize", nz,
                         FloatArray(nz.size).also { nzm.get(0, 0, it) })
            nzm.release()
        }

        // —— 颗粒段：就地写回 + y0 行偏移 ——
        run {
            val w = 41; val rows = 23; val y0 = 7
            val rgb = Mat(rows + y0, w, CvType.CV_32FC3)
            rgb.put(0, 0, randF((rows + y0) * w * 3))
            val noise = Mat(rows + y0, w, CvType.CV_32FC1)
            noise.put(0, 0, randF((rows + y0) * w))
            val rgbArr = FloatArray((rows + y0) * w * 3).also { rgb.get(0, 0, it) }
            val noiseArr = FloatArray((rows + y0) * w).also { noise.get(0, 0, it) }
            val want = FloatArray(rows * w * 3)
            NativeKernels.filmGrainBand(
                rgbArr.copyOfRange(y0 * w * 3, rgbArr.size),
                noiseArr.copyOfRange(y0 * w, noiseArr.size), want)
            NativeKernels.filmGrainBandMat(rgb.dataAddr(), y0, noise.dataAddr(), rows, w)
            val got = FloatArray((rows + y0) * w * 3).also { rgb.get(0, 0, it) }
            assertBitsEq("grainBand", want, got.copyOfRange(y0 * w * 3, got.size))
            rgb.release(); noise.release()
        }
    }

    /**
     * 全尺寸导出内存回归（2026-10-03 OOM 修复；真机 5011×3341 三连「静默跳过」
     * 的关卡）：全图 f32 + 三效果全开必须**完成**，且 Java 堆不得出现全图级
     * 分配——旧实现的段缓冲/噪声数组在 256MB 堆上限下必 OOM → 整层被跳过、
     * 出图静默丢掉柔光/光晕/颗粒。中间数据现已全部 native（指针版内核）。
     */
    @Test
    fun filmFullSizeNoOomNoJavaHeapSpike() {
        val w = 5011
        val h = 3341
        val m = Mat(h, w, CvType.CV_32FC3)
        m.setTo(org.opencv.core.Scalar(0.45, 0.50, 0.55))
        m.rowRange(200, 500).setTo(org.opencv.core.Scalar(0.95, 0.92, 0.88))
        m.colRange(600, 900).setTo(org.opencv.core.Scalar(0.08, 0.12, 0.18))
        val params = FilmLayer.DEFAULTS + mapOf(
            "soft_on" to 1f, "bloom_on" to 1f, "grain_on" to 1f)
        fun heapUsed(): Long {
            System.gc()
            val rt = Runtime.getRuntime()
            return rt.totalMemory() - rt.freeMemory()
        }
        val heap0 = heapUsed()
        val t0 = System.nanoTime()
        assertTrue("全尺寸 film 必须成功（OOM 回归）", FilmLayer.apply(m, params))
        val ms = (System.nanoTime() - t0) / 1e6
        val heapDelta = heapUsed() - heap0
        println("[film] 全尺寸 ${w}x${h} film=${"%.0f".format(ms)}ms " +
                "Java堆增量=${"%.1f".format(heapDelta / 1e6)}MB")
        assertTrue(
            "Java 堆增量必须 <16MB（实际 ${"%.1f".format(heapDelta / 1e6)}MB）——出现全图级分配",
            heapDelta < 16_000_000)
        m.release()
    }
}
