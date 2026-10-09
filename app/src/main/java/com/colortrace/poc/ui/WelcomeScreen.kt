package com.colortrace.poc.ui

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.MutableTransitionState
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.scaleIn
import androidx.compose.animation.slideInVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp

/**
 * 欢迎屏：**两个入口**——选样片（设备端 fit，默认 encoder 档）或打开**预设库**
 * （P2.14：列表 / 应用 / 删除 / 从文件导入）。
 *
 * 追色方法的选择移到**编辑页顶栏正中**（P2.14 用户反馈：入口集中在编辑页，
 * 主界面不再放方法档——fit 默认走 encoder，要换在编辑页换）。
 * 入场动效：hero 卡片 scaleIn + 内容错落渐入（一次性，M3 emphasized）。
 *
 * P2.20：多一张**「继续上次编辑」卡片**（[resumeDetail] 非空时出现）——覆盖安装 /
 * 被杀 / 系统回收后回来接着改，不必重选参考图重调参数。恢复要做 refit（1~2s），
 * 所以刻意做成**用户点一下**才发生，不压在冷启动上。
 */
@Composable
fun WelcomeScreen(
    busy: Boolean,
    busyText: String,
    error: String?,
    resumeDetail: String?,
    canDiscardResume: Boolean,
    onResume: () -> Unit,
    onDiscardResume: () -> Unit,
    onPickSample: () -> Unit,
    onOpenLibrary: () -> Unit,
) {
    val shown = remember { MutableTransitionState(false).apply { targetState = true } }
    Column(
        modifier = Modifier.fillMaxSize()
            .statusBarsPadding()
            .navigationBarsPadding()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Spacer(Modifier.weight(1f))

        // Hero 卡片：amber 渐变（品牌色唯一"装饰"出场，其余界面保持克制）
        AnimatedVisibility(
            shown,
            enter = scaleIn(tween(Motion.PAGE_MS, easing = Motion.Emphasized),
                initialScale = 0.92f) +
                    fadeIn(tween(Motion.PAGE_MS)),
        ) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(210.dp)
                    .background(
                        Brush.linearGradient(
                            listOf(MaterialTheme.colorScheme.primary,
                                   MaterialTheme.colorScheme.primaryContainer)),
                        MaterialTheme.shapes.extraLarge,
                    )
                    .padding(24.dp),
            ) {
                Column {
                    Text("colortrace",
                         style = MaterialTheme.typography.headlineMedium,
                         color = MaterialTheme.colorScheme.onPrimary)
                    Text("追色引擎",
                         style = MaterialTheme.typography.titleMedium,
                         color = MaterialTheme.colorScheme.onPrimary)
                    Spacer(Modifier.height(10.dp))
                    Text("把参考图的光影色调，\n迁移到自己的照片上。",
                         style = MaterialTheme.typography.bodyLarge,
                         color = MaterialTheme.colorScheme.onPrimaryContainer)
                }
            }
        }

        Spacer(Modifier.height(20.dp))

        // 三步提示：错落渐入。
        // 宽度收窄到 86% 并由父 Column 居中它（父是 CenterHorizontally）——竖排块
        // **整体居中、块内左对齐**才稳：直接 fillMaxWidth 会把圆点顶到屏幕最左
        // （用户反馈"太靠左、视觉上不稳定"），而只按内容宽度居中又会让圆点随文字
        // 长短漂移（"三点没对齐"）。
        Column(
            modifier = Modifier.fillMaxWidth(0.86f),
            horizontalAlignment = Alignment.Start,
        ) {
            listOf("① 选一张参考图（喜欢的色调）", "② 加入自己的照片，自动追色",
                   "③ 精修满意后保存到相册").forEachIndexed { i, step ->
                AnimatedVisibility(
                    shown,
                    enter = fadeIn(tween(Motion.PAGE_MS, delayMillis = 120 + i * 80,
                                         easing = Motion.Emphasized)) +
                            slideInVertically(tween(Motion.PAGE_MS, delayMillis = 120 + i * 80,
                                                    easing = Motion.Emphasized)) { it / 4 },
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.fillMaxWidth().padding(vertical = 6.dp)) {
                        Box(
                            Modifier.size(8.dp).background(
                                MaterialTheme.colorScheme.primary, CircleShape),
                        )
                        Text(step, modifier = Modifier.padding(start = 10.dp),
                             style = MaterialTheme.typography.bodyMedium,
                             color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }
            }
        }

        Spacer(Modifier.weight(1f))

        if (error != null) {
            Text(error, color = MaterialTheme.colorScheme.error,
                 style = MaterialTheme.typography.bodySmall)
            Spacer(Modifier.height(8.dp))
        }

        // 「继续上次编辑」：一行卡片（标题 + 方法/保护/张数），右侧 × 放弃。
        // 只在此处出现——用户明确点它才付恢复代价；× 只在现场**不在内存**时给
        //（在内存里放弃会把状态变成够不着的孤儿）
        if (resumeDetail != null) {
            OutlinedButton(
                onClick = onResume,
                enabled = !busy,
                modifier = Modifier.fillMaxWidth(),
                contentPadding = PaddingValues(horizontal = 8.dp, vertical = 8.dp),
            ) {
                // 文案按**卡片整宽居中**（用户反馈"太靠左"）：× 用 Box 叠在右端、
                // 不参与排版——若把 × 放进 Row 里，文本块会以"卡片宽 −×宽"为基准居中，
                // 看起来仍然偏左。左右各留 48dp 是给 × 与视觉呼吸的**对称**余量
                //（副行过长由省略号收尾，不会钻到 × 下面）
                Box(Modifier.fillMaxWidth()) {
                    Column(
                        modifier = Modifier.align(Alignment.Center)
                            .padding(horizontal = 48.dp),
                        horizontalAlignment = Alignment.CenterHorizontally,
                    ) {
                        Text("继续上次编辑",
                             style = MaterialTheme.typography.titleSmall)
                        Text(resumeDetail,
                             style = MaterialTheme.typography.bodySmall,
                             color = MaterialTheme.colorScheme.onSurfaceVariant,
                             maxLines = 1, overflow = TextOverflow.Ellipsis)
                    }
                    if (canDiscardResume) {
                        IconButton(onClick = onDiscardResume,
                                   modifier = Modifier.align(Alignment.CenterEnd)) {
                            Icon(Icons.Filled.Close, contentDescription = "放弃上次会话")
                        }
                    }
                }
            }
            Spacer(Modifier.height(10.dp))
        }

        Button(
            onClick = onPickSample,
            enabled = !busy,
            modifier = Modifier.fillMaxWidth().height(52.dp),
        ) {
            if (busy) {
                CircularProgressIndicator(
                    modifier = Modifier.size(18.dp), strokeWidth = 2.dp,
                    color = MaterialTheme.colorScheme.onPrimary)
                Spacer(Modifier.size(10.dp))
                Text(busyText)
            } else {
                Text("选择参考图", style = MaterialTheme.typography.titleMedium)
            }
        }
        TextButton(onClick = onOpenLibrary, enabled = !busy,
                   modifier = Modifier.padding(top = 4.dp)) {
            Text("打开预设库")
        }
    }
}
