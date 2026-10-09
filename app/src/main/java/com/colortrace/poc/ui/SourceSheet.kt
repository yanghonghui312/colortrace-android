package com.colortrace.poc.ui

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.MutableTransitionState
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.slideInVertically
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Star
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp

/**
 * 参考来源面板（P2.14 用户反馈：**不必退回主界面**才能换参考）。
 *
 * 编辑页顶栏右侧的「参考图 / 预设」pill 打开本面板——展示**当前来源**并在原地
 * 更换：换参考图（重选样片重新 fit，照片列表与精调保留）或换预设（打开预设库）。
 *
 * 动效（对齐 `docs/ui-redesign-research-2026-09-28.md` §2 原则 3）：
 * ModalBottomSheet 的 M3 进场 + 内容**一次性错落渐入**（60ms 步进 / 250ms，
 * 与欢迎屏同一套语言）；来源标签切换由调用侧的 AnimatedContent 承担。
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SourceSheet(
    isReference: Boolean,
    detail: String,
    sampleThumb: ImageBitmap?,
    canSavePreset: Boolean,
    onSwitchSample: () -> Unit,
    onSwitchPreset: () -> Unit,
    onSavePreset: () -> Unit,
    onDismiss: () -> Unit,
) {
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
    val shown = remember { MutableTransitionState(false).apply { targetState = true } }
    ModalBottomSheet(onDismissRequest = onDismiss, sheetState = sheetState) {
        Column(
            Modifier.fillMaxWidth().navigationBarsPadding()
                .padding(horizontal = 20.dp),
        ) {
            Text("参考图与预设", style = MaterialTheme.typography.titleLarge)

            Spacer(Modifier.height(12.dp))

            // 当前来源（缩略图 / 图标 + 类型 + 细节）
            Row(verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp)) {
                Box(
                    Modifier.size(44.dp).clip(MaterialTheme.shapes.small)
                        .background(MaterialTheme.colorScheme.surfaceContainerHighest),
                    contentAlignment = Alignment.Center,
                ) {
                    if (isReference && sampleThumb != null) {
                        Image(sampleThumb, contentDescription = null,
                              modifier = Modifier.fillMaxSize(),
                              contentScale = ContentScale.Crop)
                    } else {
                        Icon(if (isReference) Icons.Filled.Refresh else Icons.Filled.Star,
                             contentDescription = null,
                             tint = MaterialTheme.colorScheme.primary,
                             modifier = Modifier.size(20.dp))
                    }
                }
                Spacer(Modifier.width(12.dp))
                Column(Modifier.weight(1f)) {
                    Text(if (isReference) "当前：参考图" else "当前：预设",
                         style = MaterialTheme.typography.bodyLarge)
                    Text(detail, style = MaterialTheme.typography.bodySmall,
                         color = MaterialTheme.colorScheme.onSurfaceVariant,
                         maxLines = 2, overflow = TextOverflow.Ellipsis)
                }
            }

            Spacer(Modifier.height(8.dp))

            StaggeredRow(shown, 0) {
                ActionRow(
                    icon = { Icon(Icons.Filled.Refresh, null, tint = MaterialTheme.colorScheme.primary,
                                  modifier = Modifier.size(20.dp)) },
                    title = "更换参考图",
                    subtitle = "重选一张参考图（照片与精修保留）",
                    onClick = onSwitchSample,
                )
            }
            StaggeredRow(shown, 1) {
                ActionRow(
                    icon = { Icon(Icons.Filled.Star, null, tint = MaterialTheme.colorScheme.primary,
                                  modifier = Modifier.size(20.dp)) },
                    title = "更换预设",
                    subtitle = "从预设库中选择",
                    onClick = onSwitchPreset,
                )
            }
            // 将当前参考图存入预设（P2.15 UI 用户指定：入口收进本面板的下拉里；
            // 像素蛋糕也把「保存预设」放在"更多功能"下拉 —— 同范式）。
            // 只在参考图会话出现：预设会话没有可 fit 的样片。
            if (canSavePreset) {
                StaggeredRow(shown, 2) {
                    ActionRow(
                        icon = { Icon(Icons.Filled.Check, null,
                                      tint = MaterialTheme.colorScheme.primary,
                                      modifier = Modifier.size(20.dp)) },
                        title = "将当前参考图存入预设",
                        subtitle = "保存的是参考图解析出的色彩映射，与照片、精修无关",
                        onClick = onSavePreset,
                    )
                }
            }

            Spacer(Modifier.height(16.dp))
        }
    }
}

/** 一次性错落渐入（60ms 步进 / 250ms / emphasized，与欢迎屏同款）。 */
@Composable
private fun StaggeredRow(
    shown: MutableTransitionState<Boolean>,
    index: Int,
    content: @Composable () -> Unit,
) {
    AnimatedVisibility(
        shown,
        enter = fadeIn(tween(Motion.CONTENT_MS, delayMillis = 60 * index,
                            easing = Motion.Emphasized)) +
                slideInVertically(tween(Motion.CONTENT_MS, delayMillis = 60 * index,
                                        easing = Motion.Emphasized)) { it / 5 },
    ) { content() }
}

@Composable
private fun ActionRow(
    icon: @Composable () -> Unit,
    title: String,
    subtitle: String,
    onClick: () -> Unit,
) {
    Surface(
        shape = MaterialTheme.shapes.medium,
        color = MaterialTheme.colorScheme.surfaceContainer,
        modifier = Modifier.fillMaxWidth().padding(vertical = 5.dp)
            .clickable(onClick = onClick),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.padding(horizontal = 14.dp, vertical = 14.dp)) {
            Box(
                Modifier.size(34.dp).clip(CircleShape)
                    .background(MaterialTheme.colorScheme.primaryContainer),
                contentAlignment = Alignment.Center,
            ) { icon() }
            Spacer(Modifier.width(12.dp))
            Column(Modifier.weight(1f)) {
                Text(title, style = MaterialTheme.typography.bodyLarge)
                Text(subtitle, style = MaterialTheme.typography.bodySmall,
                     color = MaterialTheme.colorScheme.onSurfaceVariant,
                     maxLines = 1, overflow = TextOverflow.Ellipsis)
            }
        }
    }
}