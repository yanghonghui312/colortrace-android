package com.colortrace.poc.ui

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Share
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.colortrace.poc.PresetStore
import com.colortrace.poc.methodLabelOf
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** 预设库条目里的档位（kind）→ 中文标签：统一走 Labels.kt 的 methodLabelOf。 */
fun presetMethodLabel(kind: String): String = methodLabelOf(kind)

/**
 * 预设库面板（P2.14）：列表 / 应用 / 删除 + 「从文件导入」。
 * P2.17：导入/导出统一在库里——每条目加**导出按钮**（SAF 存为 bundle JSON，
 * 含全部档 + 样片缩略图），「从文件导入」升级为识别 bundle/单档预设两种格式。
 *
 * 存储与格式见 [PresetStore]：一个条目可含多档（同一张样片 fit 出的 encoder/
 * reinhard/ot…），应用时取默认档（encoder 优先，其次按名字典序）。
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PresetLibrarySheet(
    entries: List<PresetStore.Entry>,
    thumbs: Map<String, ImageBitmap?>,
    onApply: (PresetStore.Entry) -> Unit,
    onDelete: (PresetStore.Entry) -> Unit,
    onExport: (PresetStore.Entry) -> Unit,
    onImport: () -> Unit,
    onDismiss: () -> Unit,
) {
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
    ModalBottomSheet(onDismissRequest = onDismiss, sheetState = sheetState) {
        Column(
            Modifier.fillMaxWidth().navigationBarsPadding()
                .padding(horizontal = 20.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("预设库", style = MaterialTheme.typography.titleLarge)
                Spacer(Modifier.weight(1f))
                TextButton(onClick = onImport) { Text("从文件导入") }
            }

            if (entries.isEmpty()) {
                Text(
                    "预设库还是空的——用参考图追色后点「存入预设库」，" +
                            "或从文件导入预设。",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(vertical = 24.dp),
                )
            } else {
                LazyColumn(Modifier.heightIn(max = 420.dp)) {
                    items(entries, key = { it.id }) { e ->
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.Start,
                            modifier = Modifier.fillMaxWidth()
                                .clickable { onApply(e) }
                                .padding(vertical = 8.dp),
                        ) {
                            Box(
                                Modifier.size(48.dp)
                                    .clip(RoundedCornerShape(8.dp))
                                    .background(MaterialTheme.colorScheme.surfaceContainerHighest),
                                contentAlignment = Alignment.Center,
                            ) {
                                val t = thumbs[e.id]
                                if (t != null) {
                                    Image(t, contentDescription = null,
                                          modifier = Modifier.fillMaxSize(),
                                          contentScale = ContentScale.Crop)
                                } else {
                                    // 统计类预设不含图像（只有统计量）——用档位占位
                                    Text(
                                        e.methods.firstOrNull()
                                            ?.let { presetMethodLabel(it) } ?: "预设",
                                        style = MaterialTheme.typography.labelSmall,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                                    )
                                }
                            }
                            Spacer(Modifier.width(12.dp))
                            Column(Modifier.weight(1f)) {
                                Text(e.name, style = MaterialTheme.typography.bodyLarge,
                                     maxLines = 1, overflow = TextOverflow.Ellipsis)
                                // 档名与日期**分成两行**：三档并列 + 日期挤在一行会在词中间
                                // 折行（实测"影调保 / 真"），拆开后两行各自完整、左对齐。
                                Text(
                                    e.methods.joinToString(" · ") { presetMethodLabel(it) },
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                    maxLines = 1, overflow = TextOverflow.Ellipsis,
                                    modifier = Modifier.padding(top = 2.dp),
                                )
                                Text(
                                    formatCreated(e.created),
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                    maxLines = 1,
                                )
                            }
                            IconButton(onClick = { onExport(e) }) {
                                Icon(Icons.Filled.Share, contentDescription = "导出预设")
                            }
                            IconButton(onClick = { onDelete(e) }) {
                                Icon(Icons.Filled.Delete, contentDescription = "删除预设")
                            }
                        }
                    }
                }
            }
            Spacer(Modifier.height(16.dp))
        }
    }
}

private fun formatCreated(ms: Long): String =
    SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault()).format(Date(ms))