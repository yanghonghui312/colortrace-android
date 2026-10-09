package com.colortrace.poc.ui

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.graphics.ClipOp
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.clipPath
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import kotlin.math.roundToInt

/**
 * 前后对比画布（桌面 _CompareView 的 Compose 版，img-comparison-slider 范式）：
 * 可拖竖线分割（左原图右追色后）、圆钮手柄带 ‹ › 暗示、双击切换
 * 只看原图 → 只看结果 → 回到分割。拖到任意位置都跟手（不必捏着手柄）。
 *
 * 动效（docs/ui-redesign-research-2026-09-28.md）：
 * - 拖动 1:1 跟手（snap() 不插值——交互跟手优先，与桌面一致）；
 * - 新结果到位时 250ms 淡入（M3 medium duration）；
 * - 双击的模式切换用弹簧（M3 Expressive 的 spring 手感）。
 */
@Composable
fun CompareCanvas(
    before: ImageBitmap?,
    after: ImageBitmap?,
    modifier: Modifier = Modifier,
    overlay: ImageBitmap? = null,
    onTap: (() -> Unit)? = null,
) {
    var mode by remember { mutableIntStateOf(0) }   // 0=分割 1=只看原图 2=只看结果
    var canvasWidth by remember { mutableIntStateOf(0) }
    // 结果刷新 → 250ms 淡入（首帧 snap，避免开屏空闪）
    val afterAlpha = remember { Animatable(1f) }
    LaunchedEffect(after) {
        if (after != null) {
            afterAlpha.snapTo(0f)
            afterAlpha.animateTo(1f, tween(Motion.CONTENT_MS + 50))
        }
    }
    // 绘制分割位置：**拖动直写 dragFrac**——此前经 animateFloatAsState(snap) 中转，
    // 每次拖动都要"写状态 → 重组 → 动画值 → 再重组"多一帧，真机上就是不跟手。
    // 双击的模式切换/切回用弹簧（modeAnim.value 本身是状态，逐帧驱动重组）；
    // 切回分割（2→0）时从当前位置弹回用户拖动位，完成后无缝交还给 dragFrac。
    // 初始位置 = 1/3（用户拍板：左原图 1/3 · 右成片 2/3，视觉重心给修改图）
    var dragFrac by remember { mutableFloatStateOf(1f / 3f) }
    val modeAnim = remember { Animatable(1f / 3f) }
    var returnAnim by remember { mutableStateOf(false) }
    LaunchedEffect(mode) {
        when (mode) {
            1 -> { modeAnim.snapTo(dragFrac); modeAnim.animateTo(1f, spring(dampingRatio = 0.9f)) }
            2 -> { modeAnim.snapTo(dragFrac); modeAnim.animateTo(0f, spring(dampingRatio = 0.9f)) }
            else -> {
                returnAnim = true
                modeAnim.animateTo(dragFrac, spring(dampingRatio = 0.9f))
                returnAnim = false
            }
        }
    }
    val drawnFrac = if (mode == 0 && !returnAnim) dragFrac else modeAnim.value

    val dividerColor = Color.White.copy(alpha = 0.85f)
    val knobFill = MaterialTheme.colorScheme.surface
    val knobBorder = MaterialTheme.colorScheme.primary

    Canvas(
        modifier = modifier
            .semantics { contentDescription = "前后对比画布" }
            .onSizeChanged { canvasWidth = it.width }
            .pointerInput(Unit) {
                detectDragGestures { change, _ ->
                    change.consume()
                    if (canvasWidth > 0) {
                        mode = 0                       // 一拖就回到分割模式
                        returnAnim = false             // 弹回动画作废，直接跟手
                        dragFrac = (change.position.x / canvasWidth)
                            .coerceIn(0.02f, 0.98f)
                    }
                }
            }
            .pointerInput(Unit) {
                // 单触 = 沉浸开关（收起/弹出下方控制页，Lightroom mobile 同款手势）；
                // 双触 = 三种对比模式循环。同时注册时 onTap 会等双触超时（~300ms）
                // 才触发——这是 Compose 的固有语义，换来的是一次手势同时支持两者。
                detectTapGestures(
                    onTap = { onTap?.invoke() },
                    onDoubleTap = { mode = (mode + 1) % 3 },
                )
            },
    ) {
        if (before == null) return@Canvas
        val s = minOf(size.width / before.width, size.height / before.height)
        val dw = before.width * s
        val dh = before.height * s
        val ox = (size.width - dw) / 2f
        val oy = (size.height - dh) / 2f
        val dstOff = IntOffset(ox.roundToInt(), oy.roundToInt())
        val dstSize = IntSize(dw.roundToInt(), dh.roundToInt())

        // before 全画；after 裁到分割线右侧（带淡入 alpha）
        drawImage(before, dstOffset = dstOff, dstSize = dstSize)
        // 防御：尺寸不匹配（旧结果残留）时不画——宁可只看原图，也不张冠李戴
        val afterOk = after != null && after.width == before.width &&
                after.height == before.height
        if (afterOk) {
            val cutoff = drawnFrac.coerceIn(0f, 1f)
            if (cutoff < 1f && afterAlpha.value > 0f) {
                val x = ox + dw * cutoff
                val right = Path().apply { addRect(Rect(x, oy, ox + dw, oy + dh)) }
                clipPath(right, ClipOp.Intersect) {
                    drawImage(after!!, dstOffset = dstOff, dstSize = dstSize,
                              alpha = afterAlpha.value)
                    // 保护区域叠加（透明绿，alpha 已逐像素编码在叠加位图里）：
                    // 只影响显示，保存路径用的是未叠加的 after
                    if (overlay != null && overlay.width == before.width &&
                            overlay.height == before.height) {
                        drawImage(overlay, dstOffset = dstOff, dstSize = dstSize)
                    }
                }
            }
        }

        // 分割线 + 圆钮手柄
        if (mode == 0) {
            val x = ox + dw * drawnFrac
            val cy = oy + dh / 2f
            drawLine(dividerColor, Offset(x, oy), Offset(x, oy + dh), 2f)
            val r = 14.dp.toPx()
            drawCircle(knobFill, r, Offset(x, cy))
            drawCircle(knobBorder, r, Offset(x, cy), style = Stroke(2.5f))
            // ‹ › 暗示可拖
            fun arrow(dir: Float) {
                val tip = Offset(x + dir * r * 0.55f, cy)
                drawLine(dividerColor,
                         Offset(tip.x - dir * 4.dp.toPx(), cy - 4.dp.toPx()), tip, 2f)
                drawLine(dividerColor,
                         Offset(tip.x - dir * 4.dp.toPx(), cy + 4.dp.toPx()), tip, 2f)
            }
            arrow(-1f)
            arrow(1f)
        }
    }
}
