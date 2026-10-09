package com.colortrace.poc.ui

import android.graphics.Bitmap
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.scaleIn
import androidx.compose.animation.scaleOut
import androidx.compose.animation.togetherWith
import androidx.compose.animation.core.tween
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.pager.HorizontalPager
import androidx.compose.foundation.pager.rememberPagerState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.AddCircle
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Create
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Star
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.ScrollableTabRow
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Tab
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.graphics.BlendMode
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.CompositingStrategy
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.asAndroidBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.PointerEventPass
import androidx.compose.ui.input.pointer.changedToDown
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.colortrace.engine.DevelopSpec
import com.colortrace.poc.ExportSize
import com.colortrace.poc.ProtectMode
import com.colortrace.poc.methodDescOf
import com.colortrace.poc.methodLabelOf
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * 编辑屏：对比画布 + 强度/保护控件 + 照片条 + 保存/分享 + 手动精调面板。
 * 状态都在 MainActivity（v1 无 ViewModel——进程被杀重建会丢会话，可接受）。
 *
 * 精调（P2.11 改版）：入口从角落 TextButton 升级为高亮图标按钮（显眼、可切换
 * 开合）；面板从 ModalBottomSheet 改为**内嵌在控件区**——打开时替换下方控件堆，
 * 画布保持可见（精调必须人眼实时盯着图，抽屉拉长会挡画布——用户实测反馈）。
 */
@Composable
fun EditScreen(
    before: ImageBitmap?,
    after: ImageBitmap?,
    overlay: ImageBitmap?,
    busy: Boolean,
    busyText: String,
    strength: Float,
    protect: ProtectMode,
    protectStrength: Float,
    method: String,
    /** 方法 pill 可选档（P2.16）：设备端会话固定 3 档；预设会话 = 条目携带的档。 */
    methodOptions: List<String>,
    canSwitchMethod: Boolean,
    regionSupported: Boolean,
    showMask: Boolean,
    regionNoSkin: Boolean,
    photos: List<ImageBitmap>,
    photoKeys: List<Any>,
    activeIndex: Int,
    statusMsg: String?,
    canSave: Boolean,
    developEnabled: Boolean,
    devValues: Map<String, Float>,
    onDevelopToggle: (Boolean) -> Unit,
    onDevelopValue: (String, Float) -> Unit,
    filmValues: Map<String, Float>,
    onFilmValue: (String, Float) -> Unit,
    onBack: () -> Unit,
    onStrength: (Float) -> Unit,
    onMethod: (String) -> Unit,
    onProtect: (ProtectMode) -> Unit,
    onProtectStrength: (Float) -> Unit,
    onShowMask: (Boolean) -> Unit,
    onSelectPhoto: (Int) -> Unit,
    onRemovePhoto: (Int) -> Unit,
    onAddPhotos: () -> Unit,
    onSave: (ExportSize) -> Unit,
    onExportAll: (ExportSize) -> Unit,
    sourceIsReference: Boolean,
    sourceDetail: String,
    sourceThumb: ImageBitmap?,
    onOpenSource: () -> Unit,
    /** 画布手势一次性提示（2026-09-30 文案规范 C-1）：首次进入编辑页显示一次。 */
    showCanvasHint: Boolean,
    onCanvasHintShown: () -> Unit,
) {
    // 底部面板翻页（P2.15 UI）：0=调整（强度/保护/照片栏）1=精调
    var panelPage by androidx.compose.runtime.remember {
        androidx.compose.runtime.mutableIntStateOf(0)
    }
    // 沉浸：轻触画布收起/弹出下方控制页（Lightroom mobile「轻触隐藏界面」同款手势）
    var immersive by androidx.compose.runtime.remember {
        androidx.compose.runtime.mutableStateOf(false)
    }
    // 画布手势提示（2026-09-30 文案规范 C-1）：画布上"拖分割线 / 双击切换 / 轻触沉浸"
    // 三个手势此前没有任何文字线索。首次进入编辑页显示一次、4 秒后淡出，之后不再出现。
    // 帮助面板（2026-09-30 用户要求）：方法/保护说明与操作教程都收在这里。
    var showHelp by androidx.compose.runtime.remember {
        androidx.compose.runtime.mutableStateOf(false)
    }
    var canvasHintVisible by androidx.compose.runtime.remember {
        androidx.compose.runtime.mutableStateOf(showCanvasHint)
    }
    LaunchedEffect(showCanvasHint) {
        if (showCanvasHint) {
            delay(4000)
            canvasHintVisible = false
            onCanvasHintShown()
        }
    }
    if (showHelp) {
        HelpSheet(onDismiss = { showHelp = false })
    }
    // edge-to-edge 避让：不加这层，顶栏（微调/返回）会被状态栏盖住、
    // 点不到（vivo 真机实测——tap 落进系统状态栏被消费；用户目检同样报"被挡"）
    Column(Modifier.fillMaxSize()
        .statusBarsPadding()
        .navigationBarsPadding()) {
        // 顶栏（P2.14 重排）：左=返回；**正中=切换算法**；右=参考来源 + 精调。
        // 用 Box 让方法 pill 真正居中（Left/Center/Right 三个槽位互不影响）。
        Box(Modifier.fillMaxWidth().padding(horizontal = 6.dp, vertical = 4.dp)) {
            Row(modifier = Modifier.align(Alignment.CenterStart)) {
                IconButton(onClick = onBack) {
                    Icon(Icons.AutoMirrored.Filled.ArrowBack,
                         contentDescription = "返回",
                         tint = MaterialTheme.colorScheme.onSurface)
                }
                // 帮助入口（2026-09-30 用户要求）：放在返回键旁边，收纳方法/保护的
                // 详细说明与操作教程，主界面与下拉才能保持简约。
                IconButton(onClick = { showHelp = true }) {
                    Icon(Icons.Filled.Info,
                         contentDescription = "使用帮助",
                         tint = MaterialTheme.colorScheme.onSurface)
                }
            }

            // 方法档（2026-09-30 合并：**3 档算法**）——「用不用分区」由保护模式决定
            // （设备端会话切到分区保护会按同张样片现场升档）。重建会话后迁移缓存自动失效。
            // 预设会话不可换（统计/分区 fit 需要样片像素，预设里只有缩略图）。
            val methodLabel = methodLabelOf(method)
            var showMethodPicker by androidx.compose.runtime.remember {
                androidx.compose.runtime.mutableStateOf(false)
            }
            Surface(
                shape = CircleShape,
                color = if (canSwitchMethod) MaterialTheme.colorScheme.surfaceContainerHigh
                        else MaterialTheme.colorScheme.surfaceContainer,
                modifier = Modifier.align(Alignment.Center)
                    .clip(CircleShape)
                    .clickable(enabled = canSwitchMethod && !busy) { showMethodPicker = true },
            ) {
                Row(verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 7.dp)) {
                    AnimatedContent(
                        targetState = methodLabel,
                        transitionSpec = {
                            fadeIn(tween(Motion.FAST_MS, easing = Motion.Emphasized)) togetherWith
                                    fadeOut(tween(Motion.FAST_MS))
                        },
                        label = "methodLabel",
                    ) { label ->
                        Text(label, style = MaterialTheme.typography.labelLarge,
                             maxLines = 1,
                             color = if (canSwitchMethod) MaterialTheme.colorScheme.primary
                                     else MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                    Icon(Icons.Filled.ArrowDropDown, contentDescription = "切换追色方法",
                         modifier = Modifier.size(16.dp),
                         tint = if (canSwitchMethod) MaterialTheme.colorScheme.primary
                                else MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
            if (showMethodPicker) {
                androidx.compose.material3.AlertDialog(
                    onDismissRequest = { showMethodPicker = false },
                    title = { Text("切换追色方法") },
                    text = {
                        Column {
                            methodOptions.forEach { id ->
                                TextButton(
                                    onClick = { showMethodPicker = false; onMethod(id) },
                                    enabled = id != method,
                                    modifier = Modifier.fillMaxWidth(),
                                ) {
                                    // 档名 + 一句简介（2026-09-30 用户要求：让用户看懂各档区别）
                                    Column(Modifier.fillMaxWidth()) {
                                        Text(methodLabelOf(id),
                                             style = MaterialTheme.typography.bodyLarge,
                                             color = if (id == method)
                                                 MaterialTheme.colorScheme.primary
                                             else MaterialTheme.colorScheme.onSurface)
                                        Text(methodDescOf(id),
                                             style = MaterialTheme.typography.bodySmall,
                                             color = MaterialTheme.colorScheme.onSurfaceVariant)
                                    }
                                }
                            }
                        }
                    },
                    confirmButton = {},
                    dismissButton = {
                        TextButton(onClick = { showMethodPicker = false }) { Text("取消") }
                    },
                )
            }

            Row(verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.align(Alignment.CenterEnd)) {
                // 保存（P2.15 UI 用户指定）：原精调入口的位置换成**主操作「保存」**——
                // 与像素蛋糕「导出」在右上、LR「分享」在右上的范式一致；
                // 批量导出收进它的下拉（照片 >1 才出现），精调改为底部面板翻页
                var showSaveMenu by androidx.compose.runtime.remember {
                    androidx.compose.runtime.mutableStateOf(false)
                }
                Box {
                    Button(
                        onClick = { showSaveMenu = true },
                        enabled = !busy,
                        modifier = Modifier.height(38.dp),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = if (canSave) MaterialTheme.colorScheme.primary
                                             else MaterialTheme.colorScheme.surfaceContainerHighest,
                            contentColor = if (canSave) MaterialTheme.colorScheme.onPrimary
                                           else MaterialTheme.colorScheme.onSurfaceVariant,
                        ),
                        contentPadding = PaddingValues(start = 14.dp, end = 8.dp),
                    ) {
                        Text("保存", style = MaterialTheme.typography.labelLarge)
                        Icon(Icons.Filled.ArrowDropDown, contentDescription = "保存选项",
                             modifier = Modifier.size(16.dp))
                    }
                    DropdownMenu(expanded = showSaveMenu,
                                 onDismissRequest = { showSaveMenu = false }) {
                        // P2.21：尺寸分两档（原始 / 预压缩短边 1920），单张与批量各一项——
                        // 用户的心智模型就是"两种导出"，不做隐藏的尺寸开关（暗态会让人
                        // 导错档还不知道）。数字从 ExportSize 取，别在这里写死。
                        DropdownMenuItem(
                            text = { Text("保存到相册（${ExportSize.ORIGINAL.label}）") },
                            enabled = canSave && !busy,
                            onClick = { showSaveMenu = false; onSave(ExportSize.ORIGINAL) },
                        )
                        DropdownMenuItem(
                            text = { Text("保存到相册（${ExportSize.SOCIAL.label}）") },
                            enabled = canSave && !busy,
                            onClick = { showSaveMenu = false; onSave(ExportSize.SOCIAL) },
                        )
                        if (photos.size > 1) {
                            DropdownMenuItem(
                                text = {
                                    Text("批量导出全部（${photos.size} 张 · " +
                                            "${ExportSize.ORIGINAL.label}）")
                                },
                                enabled = !busy,
                                onClick = {
                                    showSaveMenu = false; onExportAll(ExportSize.ORIGINAL)
                                },
                            )
                            DropdownMenuItem(
                                text = {
                                    Text("批量导出全部（${photos.size} 张 · " +
                                            "${ExportSize.SOCIAL.label}）")
                                },
                                enabled = !busy,
                                onClick = {
                                    showSaveMenu = false; onExportAll(ExportSize.SOCIAL)
                                },
                            )
                        }
                    }
                }
            }
        }

        // 对比画布（含进行中的浮层）。
        // 沉浸（P2.15 UI 用户指定）：**轻触画布**收起下方控制页、再触弹出；
        // 收起时同时放大画布（去掉左右留白 + 圆角归零），与 Lightroom mobile
        // 「轻触照片隐藏命令与滑块 / 再轻触恢复」同款手势。
        val canvasPad by androidx.compose.animation.core.animateDpAsState(
            if (immersive) 0.dp else 12.dp,
            tween(Motion.CONTENT_MS, easing = Motion.Emphasized), label = "canvasPad")
        val canvasRadius by androidx.compose.animation.core.animateDpAsState(
            if (immersive) 0.dp else 16.dp,
            tween(Motion.CONTENT_MS, easing = Motion.Emphasized), label = "canvasRadius")
        var canvasSize by androidx.compose.runtime.remember {
            androidx.compose.runtime.mutableStateOf(androidx.compose.ui.unit.IntSize.Zero)
        }
        Box(Modifier.weight(1f).fillMaxWidth().padding(horizontal = canvasPad)
            .onSizeChanged { canvasSize = it }) {
            Surface(
                shape = androidx.compose.foundation.shape.RoundedCornerShape(canvasRadius),
                color = MaterialTheme.colorScheme.surfaceContainer,
                modifier = Modifier.fillMaxSize(),
            ) {
                CompareCanvas(before, after, Modifier.fillMaxSize(), overlay = overlay,
                              onTap = { immersive = !immersive })
            }
            // 画布直方图（2026-10-01 用户点单）：**左上角**小态（通常是人脸之外
            // 的天空/背景区，比顶部居中更少挡主体）；点击滑到**画布中央**放大。
            // 沉浸时随面板一起收起；点它只切大小——clickable 会消费事件，不会
            // 触发画布的沉浸轻触。
            if (!immersive && after != null) {
                HistogramOverlay(after, canvasSize)
            }
            // 进行中浮层：淡入淡出 + 轻微缩放（不遮挡全局判断，指示器居中）
            // 用全限定调用：ColumnScope 的同名扩展在此上下文会遮蔽顶层函数
            androidx.compose.animation.AnimatedVisibility(
                busy,
                enter = fadeIn(tween(Motion.CONTENT_MS)) +
                        scaleIn(tween(Motion.CONTENT_MS), initialScale = 0.95f),
                exit = fadeOut(tween(Motion.FAST_MS)) +
                        scaleOut(tween(Motion.FAST_MS), targetScale = 0.95f),
                modifier = Modifier.align(Alignment.BottomCenter).padding(16.dp),
            ) {
                Surface(shape = CircleShape,
                        color = MaterialTheme.colorScheme.surfaceContainerHigh,
                        tonalElevation = 4.dp) {
                    Row(modifier = Modifier.padding(horizontal = 16.dp, vertical = 10.dp),
                        verticalAlignment = Alignment.CenterVertically) {
                        LinearProgressIndicator(
                            modifier = Modifier.width(72.dp).height(4.dp),
                            color = MaterialTheme.colorScheme.primary)
                        Spacer(Modifier.width(10.dp))
                        Text(busyText, style = MaterialTheme.typography.bodySmall,
                             color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }
            }
            // 画布手势一次性提示（C-1）：三个手势此前无任何文字线索；与 busy 浮层
            // 同处画布底部，故 busy 时不显示（避免两块牌子叠在一起）。
            androidx.compose.animation.AnimatedVisibility(
                visible = canvasHintVisible && !busy,
                enter = fadeIn(tween(Motion.CONTENT_MS)),
                exit = fadeOut(tween(Motion.CONTENT_MS)),
                modifier = Modifier.align(Alignment.BottomCenter).padding(bottom = 14.dp),
            ) {
                Surface(shape = CircleShape,
                        color = MaterialTheme.colorScheme.surfaceContainerHigh.copy(alpha = 0.92f),
                        tonalElevation = 4.dp) {
                    Text("拖动分割线对比 · 双击切换 · 轻触隐藏面板",
                         style = MaterialTheme.typography.bodySmall,
                         color = MaterialTheme.colorScheme.onSurfaceVariant,
                         modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp))
                }
            }
        }

        if (statusMsg != null && !immersive) {
            Text(statusMsg,
                 style = MaterialTheme.typography.bodySmall,
                 color = MaterialTheme.colorScheme.onSurfaceVariant,
                 maxLines = 1, overflow = TextOverflow.Ellipsis,
                 modifier = Modifier.fillMaxWidth()
                     .padding(horizontal = 16.dp, vertical = 6.dp))
        }

        // ---- 底部控制面板（P2.15 UI 用户指定）----
        // · 圆角 + 半透明 + 顶部投影 = 悬浮"毛玻璃"观感（与 Lightroom mobile /
        //   像素蛋糕的浮层面板同构；真 backdrop blur 需 API 31+ RenderEffect 或
        //   自绘截屏模糊，成本/收益不划算——见 docs/ui-redesign-research §6）；
        // · 内容翻页：**调整**（强度/保护/照片栏）｜**精调**（原内嵌面板）；
        // · 沉浸（轻触画布）时整块滑出，画布 weight(1f) 随之变大。
        androidx.compose.animation.AnimatedVisibility(
            visible = !immersive,
            enter = androidx.compose.animation.expandVertically(
                tween(Motion.CONTENT_MS, easing = Motion.Emphasized),
                expandFrom = Alignment.Bottom) + fadeIn(tween(Motion.CONTENT_MS)),
            exit = androidx.compose.animation.shrinkVertically(
                tween(Motion.CONTENT_MS, easing = Motion.Emphasized),
                shrinkTowards = Alignment.Bottom) + fadeOut(tween(Motion.FAST_MS)),
        ) {
            Surface(
                shape = androidx.compose.foundation.shape.RoundedCornerShape(
                    topStart = 20.dp, topEnd = 20.dp),
                color = MaterialTheme.colorScheme.surfaceContainer.copy(alpha = 0.94f),
                tonalElevation = 6.dp,
                modifier = Modifier.fillMaxWidth(),
            ) {
                // 面板限高（屏高 56%）：保证画布至少留 ~1/3 屏——
                // 真机实测不限高时面板内容会溢出屏底（调整页 ≈6 行 + 照片条）
                val panelMax = (androidx.compose.ui.platform.LocalConfiguration.current
                    .screenHeightDp * 0.56f).dp
                Column(Modifier.heightIn(max = panelMax)) {
                    // 参考来源行（P2.14）：当前是参考图还是预设 + 细节，
                    // 点开在**原地**更换（存预设也在那个面板里）
                    SourceRow(
                        isReference = sourceIsReference,
                        detail = sourceDetail,
                        thumb = sourceThumb,
                        busy = busy,
                        onClick = onOpenSource,
                    )
                    // 翻页入口（用户指定：精调由"翻页"实现，不再是顶栏按钮）
                    PanelTabs(page = panelPage, developEnabled = developEnabled,
                              filmActive = com.colortrace.engine.FilmLayer
                                  .active(filmValues),
                              onPage = { panelPage = it })
                    androidx.compose.animation.AnimatedContent(
                        targetState = panelPage,
                        transitionSpec = {
                            (androidx.compose.animation.slideInHorizontally(
                                tween(Motion.CONTENT_MS, easing = Motion.Emphasized)) { it / 6 } +
                                    fadeIn(tween(Motion.CONTENT_MS))) togetherWith
                                    (androidx.compose.animation.slideOutHorizontally(
                                        tween(Motion.FAST_MS)) { -it / 6 } +
                                            fadeOut(tween(Motion.FAST_MS)))
                        },
                        label = "panelPage",
                    ) { page ->
                        if (page == 1) {
                            // ---- 精调页（原内嵌面板：滑杆拖即自动开总开关）----
                            DevelopPanel(
                                enabled = developEnabled,
                                values = devValues,
                                busy = busy,
                                onToggle = onDevelopToggle,
                                onValue = onDevelopValue,
                                onDismiss = { panelPage = 0 },
                            )
                        } else if (page == 2) {
                            // ---- 胶片感页（2026-10-02）：三组开关+滑杆，滚动列 ----
                            FilmPanel(
                                values = filmValues,
                                busy = busy,
                                onValue = onFilmValue,
                                onDismiss = { panelPage = 0 },
                            )
                        } else {
                            // 调整页可滚动：限高后内容可能超出（小屏/大字号），
                            // 滚动而不是裁切
                            Column(Modifier.verticalScroll(rememberScrollState())) {
            // 强度
            Row(
                modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 2.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text("追色强度", style = MaterialTheme.typography.bodyMedium)
                Slider(
                    value = strength,
                    onValueChange = onStrength,
                    colors = appSliderColors(),
                    modifier = Modifier.weight(1f).padding(horizontal = 12.dp),
                )
                Text("${(strength * 100).toInt()}%",
                     style = MaterialTheme.typography.bodyMedium,
                     color = MaterialTheme.colorScheme.primary,
                     modifier = Modifier.width(44.dp),
                     textAlign = TextAlign.End)
            }

            // 肤色保护三档（P2.13：chroma 对全部方法可用；REGION 对统计类
            // 暂不支持——桌面语义是各区独立统计映射，待后续轮，按钮置灰）
            SingleChoiceSegmentedButtonRow(
                Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 6.dp),
            ) {
                ProtectMode.entries.forEachIndexed { i, mode ->
                    SegmentedButton(
                        selected = protect == mode,
                        onClick = { onProtect(mode) },
                        enabled = mode != ProtectMode.REGION || regionSupported,
                        shape = SegmentedButtonDefaults.itemShape(index = i,
                                                                  count = ProtectMode.entries.size),
                    ) { Text(mode.label) }
                }
            }
            if (!regionSupported) {
                Text("当前预设不含分区数据，分区保护不可用",
                     style = MaterialTheme.typography.bodySmall,
                     color = MaterialTheme.colorScheme.onSurfaceVariant,
                     modifier = Modifier.fillMaxWidth()
                         .padding(horizontal = 16.dp, vertical = 1.dp))
            }

            // 分区保护没检测到人物（wSkin 峰值≈0）→ 给一键切换的提示（P2.11）
            if (protect == ProtectMode.REGION && regionNoSkin) {
                Surface(
                    shape = MaterialTheme.shapes.small,
                    color = MaterialTheme.colorScheme.surfaceContainerHighest,
                    modifier = Modifier.fillMaxWidth()
                        .padding(horizontal = 16.dp, vertical = 2.dp),
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.padding(start = 12.dp)) {
                        Text("未检测到人物，分区保护无效",
                             style = MaterialTheme.typography.bodySmall,
                             color = MaterialTheme.colorScheme.onSurfaceVariant,
                             modifier = Modifier.weight(1f))
                        TextButton(onClick = { onProtect(ProtectMode.CHROMA) },
                                   enabled = !busy) {
                            Text("改用肤色锁定",
                                 style = MaterialTheme.typography.labelLarge)
                        }
                    }
                }
            }

            // 保护强度（桌面「保护强度 N%」滑块同语义）：拖它**只重跑出图**，不重算迁移
            val protectOn = protect != ProtectMode.OFF
            Row(
                modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text("保护强度", style = MaterialTheme.typography.bodyMedium,
                     color = if (protectOn) MaterialTheme.colorScheme.onSurface
                             else MaterialTheme.colorScheme.onSurfaceVariant)
                Slider(
                    value = protectStrength,
                    onValueChange = onProtectStrength,
                    enabled = protectOn && !busy,
                    colors = appSliderColors(),
                    modifier = Modifier.weight(1f).padding(horizontal = 12.dp),
                )
                Text("${(protectStrength * 100).toInt()}%",
                     style = MaterialTheme.typography.bodyMedium,
                     color = if (protectOn) MaterialTheme.colorScheme.primary
                             else MaterialTheme.colorScheme.onSurfaceVariant,
                     modifier = Modifier.width(44.dp),
                     textAlign = TextAlign.End)
            }
            if (protectOn) {
                // 期望管理（P2.11 用户反馈"拖了没区别"）：保护只作用于皮肤、
                // 且可拉回的量 = 样片对肤色的改变量——样片肤色风格温和时差异天然小
                Text(
                    "皮肤保持原色的程度，主要作用于面部；保护强度越低，肤色受参考图影响越大",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.fillMaxWidth()
                        .padding(horizontal = 16.dp, vertical = 1.dp),
                )
            }

            // 显示保护区域（桌面 chk_mask 同款：结果侧绿色半透明叠加）
            Row(
                modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 2.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text("显示保护区域", style = MaterialTheme.typography.bodyMedium,
                     color = if (protectOn) MaterialTheme.colorScheme.onSurface
                             else MaterialTheme.colorScheme.onSurfaceVariant)
                Spacer(Modifier.weight(1f))
                Switch(checked = showMask && protectOn, onCheckedChange = onShowMask,
                       enabled = protectOn)
            }

            // 照片条（缩略图 + 右上角删除 + 末尾添加按钮）
            LazyRow(
                contentPadding = PaddingValues(horizontal = 12.dp, vertical = 8.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                // key 用照片 uri（不是下标）：删中间项时避免按位复用导致状态错位
                itemsIndexed(photos, key = { i, _ -> photoKeys.getOrElse(i) { i } }) { i, thumb ->
                    val selected = i == activeIndex
                    val borderWidth by animateDpAsState(
                        if (selected) 2.dp else 0.dp, label = "thumbBorder")
                    Box(Modifier.size(64.dp)) {
                        Image(
                            thumb,
                            contentDescription = null,
                            contentScale = ContentScale.Crop,
                            modifier = Modifier
                                .fillMaxSize()
                                // 先 clip 再 border：不加 clip 时图片按矩形绘制，
                                // 四角会顶出圆角边框（用户实测"尖角超出"）
                                .clip(MaterialTheme.shapes.small)
                                .border(borderWidth, MaterialTheme.colorScheme.primary,
                                        MaterialTheme.shapes.small)
                                .clickable { onSelectPhoto(i) },
                        )
                        // 删除（照片条右上角 ×，相册选择器惯例）
                        Surface(
                            shape = CircleShape,
                            color = MaterialTheme.colorScheme.surfaceContainerHighest,
                            modifier = Modifier.align(Alignment.TopEnd)
                                .padding(1.dp)
                                .size(20.dp)
                                .clickable(enabled = !busy) { onRemovePhoto(i) },
                        ) {
                            Box(contentAlignment = Alignment.Center) {
                                Icon(Icons.Filled.Close,
                                     contentDescription = "移除这张照片",
                                     tint = MaterialTheme.colorScheme.onSurfaceVariant,
                                     modifier = Modifier.size(13.dp))
                            }
                        }
                    }
                }
                item {
                    Surface(
                        shape = MaterialTheme.shapes.small,
                        color = MaterialTheme.colorScheme.surfaceContainerHigh,
                        modifier = Modifier.size(64.dp).clickable { onAddPhotos() },
                    ) {
                        Box(contentAlignment = Alignment.Center) {
                            Icon(Icons.Filled.AddCircle,
                                 contentDescription = "加入照片",
                                 tint = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                    }
                }
            }
                            }
                        }
                    }
                }
            }
        }
    }
}

/**
 * 控制面板的翻页入口（P2.15 UI）：调整｜精调。
 *
 * 用户指定「精调由翻页实现」——所以它不再是顶栏按钮，而是面板内的两个页签；
 * 选中页用**滑动指示条**（宽度动效）标记，精调页在总开关打开时带一个小圆点
 * （保留 P2.11「一眼可见精调在工作」的信息）。
 */
@Composable
private fun PanelTabs(page: Int, developEnabled: Boolean, filmActive: Boolean,
                      onPage: (Int) -> Unit) {
    Row(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 2.dp)) {
        listOf("调整" to 0, "精修" to 1, "胶片感" to 2).forEach { (label, idx) ->
            val sel = page == idx
            val frac by androidx.compose.animation.core.animateFloatAsState(
                if (sel) 1f / 3f else 0f,
                tween(Motion.CONTENT_MS, easing = Motion.Emphasized), label = "tabFrac")
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                modifier = Modifier.weight(1f).clip(MaterialTheme.shapes.small)
                    .clickable { onPage(idx) }.padding(vertical = 6.dp),
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(label, style = MaterialTheme.typography.labelLarge,
                         color = if (sel) MaterialTheme.colorScheme.primary
                                 else MaterialTheme.colorScheme.onSurfaceVariant)
                    if ((idx == 1 && developEnabled) || (idx == 2 && filmActive)) {
                        Box(Modifier.size(6.dp).background(
                            MaterialTheme.colorScheme.primary, CircleShape))
                    }
                }
                Spacer(Modifier.height(4.dp))
                Box(Modifier.height(2.dp).fillMaxWidth(frac)
                    .background(MaterialTheme.colorScheme.primary,
                                androidx.compose.foundation.shape.RoundedCornerShape(1.dp)))
            }
        }
    }
}

/**
 * 参考来源行（P2.14）：显示**当前来源**（参考图 / 预设 + 细节），点开原地更换。
 *
 * 为什么不在顶栏：顶栏要留给"切换算法居中"（用户指定），三件东西挤一行会让
 * 来源 pill 与方法 pill 贴在一起（真机截图实测，方法 caret 被挤没、也不居中）。
 * 放在画布下方既宽松又能放下"类型 + 细节"两行信息，精调面板打开时也够得着。
 */
@Composable
private fun SourceRow(
    isReference: Boolean,
    detail: String,
    thumb: ImageBitmap?,
    busy: Boolean,
    onClick: () -> Unit,
) {
    Surface(
        shape = MaterialTheme.shapes.small,
        color = MaterialTheme.colorScheme.surfaceContainer,
        modifier = Modifier.fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 2.dp)
            .clip(MaterialTheme.shapes.small)
            .clickable(enabled = !busy) { onClick() },
    ) {
        Row(verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 8.dp)) {
            Box(
                Modifier.size(26.dp).clip(MaterialTheme.shapes.extraSmall)
                    .background(MaterialTheme.colorScheme.surfaceContainerHighest),
                contentAlignment = Alignment.Center,
            ) {
                if (thumb != null) {
                    // 2026-10-01 用户点单：预设会话也显示样片缩略（与预设库一致），
                    // 不再只有通用星形图标；参考图行为不变
                    Image(thumb, contentDescription = null,
                          modifier = Modifier.fillMaxSize(),
                          contentScale = ContentScale.Crop)
                } else {
                    Icon(if (isReference) Icons.Filled.Refresh else Icons.Filled.Star,
                         contentDescription = null,
                         tint = MaterialTheme.colorScheme.primary,
                         modifier = Modifier.size(15.dp))
                }
            }
            Spacer(Modifier.width(10.dp))
            Column(Modifier.weight(1f)) {
                AnimatedContent(
                    targetState = if (isReference) "参考图" else "预设",
                    transitionSpec = {
                        fadeIn(tween(Motion.FAST_MS, easing = Motion.Emphasized)) togetherWith
                                fadeOut(tween(Motion.FAST_MS))
                    },
                    label = "sourceRowType",
                ) { label ->
                    Text(label, style = MaterialTheme.typography.labelLarge,
                         color = MaterialTheme.colorScheme.onSurface, maxLines = 1)
                }
                Text(detail, style = MaterialTheme.typography.bodySmall,
                     color = MaterialTheme.colorScheme.onSurfaceVariant,
                     maxLines = 1, overflow = TextOverflow.Ellipsis)
            }
            Icon(Icons.Filled.ArrowDropDown,
                 contentDescription = "更换参考图或预设",
                 tint = MaterialTheme.colorScheme.onSurfaceVariant,
                 modifier = Modifier.size(18.dp))
        }
    }
}

/**
 * 画布直方图（2026-10-01 用户点单，三轮定稿）：叠在画布上的 RGB 直方图，半透明
 * 不挡照片。**小态在画布左上角、更小**（100×46dp——顶部居中会挡主体，左上角
 * 通常是天空/背景）；点击**滑到画布上方居中**放大（208×100dp——首版验证过的
 * 位置与尺寸，LR mobile 的「点击切换大小」交互）。位置与尺寸用同一 motion token
 * 联动过渡。clickable 消费点按，不会触发画布的沉浸轻触。RGB 三通道
 * BlendMode.Lighten 叠加（重叠区取通道最大值的经典观感）；64 bin、从当前出图
 * 缩到 ≤256px 采样，Default 调度器上算——按 Bitmap 实例驱动（拖滑杆只在停手
 * 重渲染后才有新 Bitmap，天然防抖）。
 */
@Composable
private fun HistogramOverlay(after: ImageBitmap?, canvasSizePx: androidx.compose.ui.unit.IntSize,
                             modifier: Modifier = Modifier) {
    var expanded by androidx.compose.runtime.remember {
        androidx.compose.runtime.mutableStateOf(false)
    }
    var hist by androidx.compose.runtime.remember {
        androidx.compose.runtime.mutableStateOf<Array<IntArray>?>(null)
    }
    LaunchedEffect(after) {
        // runCatching：旧 Bitmap 可能恰在切图时被回收，别让统计崩了界面
        hist = withContext(Dispatchers.Default) {
            after?.let { ab -> runCatching { rgbHistogram(ab.asAndroidBitmap()) }.getOrNull() }
        }
    }
    val density = LocalDensity.current
    val canvasW = with(density) { canvasSizePx.width.toDp() }
    // **单进度动画**：位置/尺寸全部由同一个 progress 派生（目标 x 用最终宽度
    // 208dp 计算，不再追着动画中的 w 跑）——修掉二版的"末端往右挪一下"与
    // 追帧式卡顿感；一条曲线驱动，位置与尺寸天然同步。
    val progress by androidx.compose.animation.core.animateFloatAsState(
        if (expanded) 1f else 0f,
        tween(Motion.CONTENT_MS, easing = Motion.Emphasized), label = "histP")
    val w = androidx.compose.ui.unit.lerp(100.dp, 208.dp, progress)
    val h = androidx.compose.ui.unit.lerp(46.dp, 100.dp, progress)
    val x = androidx.compose.ui.unit.lerp(
        12.dp, ((canvasW - 208.dp) / 2).coerceAtLeast(12.dp), progress)
    val y = 10.dp
    Surface(
        shape = androidx.compose.foundation.shape.RoundedCornerShape(8.dp),
        color = Color.Transparent,
        border = BorderStroke(0.5.dp, Color.White.copy(alpha = 0.22f)),
        modifier = modifier.offset(x = x, y = y).width(w).height(h)
            .clip(androidx.compose.foundation.shape.RoundedCornerShape(8.dp))
            .clickable { expanded = !expanded },
    ) {
        // Offscreen 合成：Lighten 只在层内生效（不被照片干扰）；半透明底也画在
        // 层内，照片透过 Surface 露出——"带透明度、不完全遮挡"的诉求
        Canvas(Modifier.fillMaxSize()
            .graphicsLayer(compositingStrategy = CompositingStrategy.Offscreen)) {
            drawRoundRect(color = Color.Black.copy(alpha = 0.35f),
                          cornerRadius = CornerRadius(8.dp.toPx()))
            val hh = hist ?: return@Canvas
            val bins = hh[0].size
            var maxV = 0
            for (c in 0 until 3) for (b in 0 until bins) {
                if (hh[c][b] > maxV) maxV = hh[c][b]
            }
            if (maxV <= 0) return@Canvas
            val stepW = size.width / bins
            val pad = 2.dp.toPx()
            val plotH = size.height - 2 * pad
            val colors = listOf(Color(0xFFFF4D4D), Color(0xFF4DFF88), Color(0xFF4DA6FF))
            for (c in 0 until 3) {
                val path = Path()
                path.moveTo(0f, size.height)
                for (b in 0 until bins) {
                    path.lineTo(b * stepW,
                                size.height - pad - (hh[c][b].toFloat() / maxV) * plotH)
                }
                path.lineTo(size.width, size.height)
                path.close()
                // 三通道在低光区必然重叠，实心填充会被 Lighten 顶成一片白（首版
                // 实测挡照片）；2026-10-01 四轮微调：颜色整体再调淡（0.20/0.60），
                // 通透优先——曲线只做辅助读数
                drawPath(path, colors[c].copy(alpha = 0.20f),
                         blendMode = BlendMode.Lighten)
                drawPath(path, colors[c].copy(alpha = 0.60f),
                         style = Stroke(width = 1.2.dp.toPx()),
                         blendMode = BlendMode.Lighten)
            }
        }
    }
}

/** RGB 直方图（3×[bins] bin，Int 精确分桶）：缩到 ≤256px 采样统计（非主线程调用）。 */
private fun rgbHistogram(bmp: Bitmap, bins: Int = 64): Array<IntArray> {
    val scale = 256f / maxOf(bmp.width, bmp.height)
    val sample = if (scale < 1f)
        Bitmap.createScaledBitmap(bmp, (bmp.width * scale).toInt().coerceAtLeast(1),
                                  (bmp.height * scale).toInt().coerceAtLeast(1), true)
    else bmp
    val hist = Array(3) { IntArray(bins) }
    val px = IntArray(sample.width * sample.height)
    sample.getPixels(px, 0, sample.width, 0, 0, sample.width, sample.height)
    for (v in px) {
        hist[0][(v shr 16 and 0xFF) * bins / 256]++
        hist[1][(v shr 8 and 0xFF) * bins / 256]++
        hist[2][(v and 0xFF) * bins / 256]++
    }
    if (sample !== bmp) sample.recycle()
    return hist
}

/**
 * 手动精调面板（P2.11 起为**内嵌面板**，不再是 ModalBottomSheet）：
 * ScrollableTabRow（基础影调/色彩平衡/分离色调/高级）+ HorizontalPager 换页，
 * 每页该组的滑杆（参数名/数值双击回默认）。主开关关闭时整层不参与计算
 * （引擎 identity 短路，零开销）。右上 × 收起；打开时画布保持可见。
 */
@Composable
private fun DevelopPanel(
    enabled: Boolean,
    values: Map<String, Float>,
    busy: Boolean,
    onToggle: (Boolean) -> Unit,
    onValue: (String, Float) -> Unit,
    onDismiss: () -> Unit,
) {
    Surface(
        shape = MaterialTheme.shapes.large,
        color = MaterialTheme.colorScheme.surfaceContainer,
        tonalElevation = 2.dp,
        modifier = Modifier.fillMaxWidth()
            .padding(horizontal = 12.dp, vertical = 4.dp),
    ) {
        Column(Modifier.padding(horizontal = 16.dp, vertical = 4.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("精修", style = MaterialTheme.typography.titleMedium)
                Spacer(Modifier.weight(1f))
                // 用户反馈：这里原本是"重置全部"，语义上更像"启用/关闭"——
                // 改成状态文字（开关本体就是右侧 Switch）；复位下放到每个滑块的
                // **双击**（面板底部仍保留"重置本组"）
                Text(if (enabled) "已开启" else "已关闭",
                     style = MaterialTheme.typography.labelLarge,
                     color = if (enabled) MaterialTheme.colorScheme.primary
                             else MaterialTheme.colorScheme.onSurfaceVariant)
                Spacer(Modifier.width(6.dp))
                Switch(onCheckedChange = onToggle, checked = enabled)
                IconButton(onClick = onDismiss) {
                    Icon(Icons.Filled.Close,
                         contentDescription = "收起精修",
                         tint = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
            if (!enabled) {
                Text("开启后叠加在自动追色之上，调整实时生效",
                     style = MaterialTheme.typography.bodySmall,
                     color = MaterialTheme.colorScheme.onSurfaceVariant,
                     modifier = Modifier.padding(bottom = 4.dp))
            }

            val tabs = DevelopSpec.GROUPS
            val pagerState = rememberPagerState { tabs.size }
            val scope = androidx.compose.runtime.rememberCoroutineScope()
            // 模块栏（基础影调/色彩平衡/分离色调/高级）：包一层圆角容器——
            // 此前是裸 TabRow 直贴面板底，用户反馈"模块栏没做圆角"。容器色
            // surfaceContainerHigh 与面板 surfaceContainer 形成层次；TabRow 自身
            // 默认会铺 surface 底色，必须置为 Transparent 才不会盖掉圆角容器。
            Surface(
                shape = MaterialTheme.shapes.extraLarge,
                color = MaterialTheme.colorScheme.surfaceContainerHigh,
                modifier = Modifier.fillMaxWidth().padding(top = 6.dp),
            ) {
                ScrollableTabRow(
                    selectedTabIndex = pagerState.currentPage,
                    edgePadding = 12.dp,
                    divider = {},
                    containerColor = androidx.compose.ui.graphics.Color.Transparent,
                ) {
                    tabs.forEachIndexed { i, (title, keys) ->
                        val active = keys.any {
                            kotlin.math.abs((values[it] ?: DevelopSpec.BY_KEY[it]!!.def)
                                                - DevelopSpec.BY_KEY[it]!!.def) > 1e-6f
                        }
                        Tab(
                            selected = pagerState.currentPage == i,
                            onClick = {
                                scope.launch { pagerState.animateScrollToPage(i) }
                            },
                            text = {
                                Text(
                                    title,
                                    color = if (active && enabled)
                                        MaterialTheme.colorScheme.primary
                                    else androidx.compose.ui.graphics.Color.Unspecified,
                                )
                            },
                        )
                    }
                }
            }
            HorizontalPager(
                state = pagerState,
                modifier = Modifier.fillMaxWidth(),
            ) { page ->
                Column(
                    Modifier.fillMaxWidth().heightIn(max = 260.dp)
                        .verticalScroll(rememberScrollState())
                        .padding(top = 4.dp),
                ) {
                    tabs[page].second.forEach { key ->
                        val spec = DevelopSpec.BY_KEY[key]!!
                        val v = values[key] ?: spec.def
                        val text = "%.${spec.decimals}f".format(v)
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier.padding(top = 6.dp),
                        ) {
                            Text(
                                spec.cn,
                                style = MaterialTheme.typography.bodyMedium,
                                modifier = Modifier.weight(1f)
                                    .pointerInput(key) {
                                        detectTapGestures(onDoubleTap = {
                                            onValue(key, spec.def)
                                        })
                                    },
                            )
                            Text(
                                text,
                                style = MaterialTheme.typography.bodyMedium,
                                color = if (kotlin.math.abs(v - spec.def) > 1e-6f)
                                    MaterialTheme.colorScheme.primary
                                else MaterialTheme.colorScheme.onSurfaceVariant,
                                modifier = Modifier.pointerInput(key) {
                                    detectTapGestures(onDoubleTap = {
                                        onValue(key, spec.def)
                                    })
                                },
                            )
                        }
                        // 色调/色相类用渐变轨道（LR 式可视化，对齐桌面）；其余普通滑杆
                        val stops = DEVELOP_GRADIENT_STOPS[key]
                        if (stops != null) {
                            GradientSlider(
                                stops = stops,
                                value = v,
                                onValueChange = { onValue(key, it) },
                                valueRange = spec.min..spec.max,
                                enabled = !busy,
                                // 双击复位（用户指定）：手势挂到滑杆**自己的**
                                // modifier 上并监听 Initial 阶段——见 doubleTapToReset
                                modifier = Modifier.doubleTapToReset(key) {
                                    onValue(key, spec.def)
                                },
                            )
                        } else {
                            Slider(
                                value = v,
                                onValueChange = { onValue(key, it) },
                                valueRange = spec.min..spec.max,
                                enabled = !busy,
                                colors = appSliderColors(),
                                // 双击复位（用户指定）：手势挂到 Slider **自己的**
                                // modifier 上并监听 Initial 阶段——见 doubleTapToReset
                                modifier = Modifier.doubleTapToReset(key) {
                                    onValue(key, spec.def)
                                },
                            )
                        }
                    }
                    TextButton(
                        onClick = {
                            tabs[page].second.forEach { key ->
                                onValue(key, DevelopSpec.BY_KEY[key]!!.def)
                            }
                        },
                        enabled = !busy,
                    ) { Text("重置本组") }
                    Spacer(Modifier.height(8.dp))
                }
            }
        }
    }
}

/**
 * 胶片感面板（2026-10-02）：柔光 / 光晕 / 颗粒三组，各带**启用开关**（关 =
 * 该效果完全不参与出图、滑杆值保留）——与桌面 GUI「5 · 胶片质感」同构。
 * 无总开关：三开关就是门控。滑杆双击复位（同精修的 doubleTapToReset）。
 * 底部「全部重置」回默认（默认值 = 桌面用户标定档，开关默认关）。
 */
@Composable
private fun FilmPanel(values: Map<String, Float>, busy: Boolean,
                      onValue: (String, Float) -> Unit,
                      onDismiss: () -> Unit) {
    Surface(
        shape = MaterialTheme.shapes.large,
        color = MaterialTheme.colorScheme.surfaceContainer,
        tonalElevation = 2.dp,
        modifier = Modifier.fillMaxWidth()
            .padding(horizontal = 12.dp, vertical = 4.dp),
    ) {
        Column(Modifier.padding(horizontal = 16.dp, vertical = 4.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("胶片感", style = MaterialTheme.typography.titleMedium)
                Spacer(Modifier.weight(1f))
                Text(
                    if (com.colortrace.engine.FilmLayer.active(values)) "叠加中"
                    else "未启用",
                    style = MaterialTheme.typography.labelLarge,
                    color = if (com.colortrace.engine.FilmLayer.active(values))
                        MaterialTheme.colorScheme.primary
                    else MaterialTheme.colorScheme.onSurfaceVariant,
                )
                IconButton(onClick = onDismiss) {
                    Icon(Icons.Filled.Close,
                         contentDescription = "收起胶片感",
                         tint = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
            if (!com.colortrace.engine.FilmLayer.active(values)) {
                Text("柔光/光晕/颗粒各自开关，叠加在追色与精修之上",
                     style = MaterialTheme.typography.bodySmall,
                     color = MaterialTheme.colorScheme.onSurfaceVariant,
                     modifier = Modifier.padding(bottom = 4.dp))
            }
            Column(
                Modifier.fillMaxWidth().heightIn(max = 300.dp)
                    .verticalScroll(rememberScrollState()),
            ) {
                com.colortrace.engine.FilmLayer.GROUPS.forEach { (title, keys) ->
                    val onKey = keys.first()
                    val on = (values[onKey]
                        ?: com.colortrace.engine.FilmLayer.DEFAULTS[onKey]!!) >= 0.5f
                    Row(verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.padding(top = 6.dp)) {
                        Text(title, style = MaterialTheme.typography.labelLarge)
                        Spacer(Modifier.weight(1f))
                        Text(if (on) "已开启" else "已关闭",
                             style = MaterialTheme.typography.labelLarge,
                             color = if (on) MaterialTheme.colorScheme.primary
                             else MaterialTheme.colorScheme.onSurfaceVariant)
                        Spacer(Modifier.width(6.dp))
                        Switch(checked = on, enabled = !busy,
                               onCheckedChange = { onValue(onKey, if (it) 1f else 0f) })
                    }
                    keys.drop(1).forEach { key ->
                        val spec = com.colortrace.engine.FilmLayer.SPEC_BY_KEY[key]!!
                        val v = values[key] ?: spec.def
                        val reset = { onValue(key, spec.def) }
                        Row(verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier.padding(top = 4.dp)) {
                            Text(spec.cn, style = MaterialTheme.typography.bodyMedium,
                                 color = if (on) MaterialTheme.colorScheme.onSurface
                                 else MaterialTheme.colorScheme.onSurfaceVariant,
                                 modifier = Modifier.weight(1f)
                                     .pointerInput(key) {
                                         detectTapGestures(onDoubleTap = { reset() })
                                     })
                            Text("%.${spec.decimals}f".format(v),
                                 style = MaterialTheme.typography.bodyMedium,
                                 color = when {
                                     !on -> MaterialTheme.colorScheme.onSurfaceVariant
                                     kotlin.math.abs(v - spec.def) > 1e-6f ->
                                         MaterialTheme.colorScheme.primary
                                     else -> MaterialTheme.colorScheme.onSurfaceVariant
                                 },
                                 modifier = Modifier.pointerInput(key) {
                                     detectTapGestures(onDoubleTap = { reset() })
                                 })
                        }
                        // 光晕色调用渐变轨道（白→红橙，对齐桌面）；其余普通滑杆
                        val stops = FILM_GRADIENT_STOPS[key]
                        if (stops != null) {
                            GradientSlider(
                                stops = stops,
                                value = v,
                                onValueChange = { onValue(key, it) },
                                valueRange = spec.min..spec.max,
                                enabled = !busy && on,
                                modifier = Modifier.doubleTapToReset(key) { reset() },
                            )
                        } else {
                            Slider(
                                value = v,
                                onValueChange = { onValue(key, it) },
                                valueRange = spec.min..spec.max,
                                enabled = !busy && on,
                                colors = appSliderColors(),
                                modifier = Modifier.doubleTapToReset(key) { reset() },
                            )
                        }
                    }
                }
                TextButton(
                    onClick = {
                        com.colortrace.engine.FilmLayer.DEFAULTS.forEach { (k, v) ->
                            onValue(k, v)
                        }
                    },
                    enabled = !busy,
                ) { Text("全部重置") }
                Spacer(Modifier.height(8.dp))
            }
        }
    }
}

/**
 * 双击 → [onReset]，挂在 **Slider 自己的 modifier** 上。
 *
 * 为什么不用 `detectTapGestures`：M3 Slider 内部在 **Main** 阶段就消费掉 down，
 * 点按识别器若挂在父级（或链上游）拿不到事件——这正是"双击滑块没反应"的原因。
 * 这里改在 **Initial** 阶段观察（派发方向 root→leaf，早于 Slider 消费），
 * 只在判定为双击时才消费那一下，因此单选、拖动都照常工作。
 *
 * 只在"两次按下都落在同一处、且间隔在双击窗口内"时触发，避免"刚点完就拖动"
 * 被误判成双击。`key` 作为 pointerInput 的键，重组合不会打断手势状态。
 *
 * 判定为双击的那一次按下会被 **consume**：M3 Slider 自己也有"点按即定位"，
 * 不消费的话第二下抬手会把滑块又设回点按位置，复位等于白做（真机实测）。
 * 消费后 Slider 的 `awaitFirstDown(requireUnconsumed=true)` 直接跳过这一下。
 */
private fun Modifier.doubleTapToReset(key: Any?, onReset: () -> Unit): Modifier =
    pointerInput(key) {
        val timeout = viewConfiguration.doubleTapTimeoutMillis
        val slop = viewConfiguration.touchSlop * 4f
        var lastTime = 0L
        var lastPos = androidx.compose.ui.geometry.Offset.Zero
        awaitPointerEventScope {
            while (true) {
                val e = awaitPointerEvent(PointerEventPass.Initial)
                val c = e.changes.firstOrNull() ?: continue
                if (!c.changedToDown()) continue
                val dt = c.uptimeMillis - lastTime
                val dx = (c.position - lastPos).getDistance()
                if (lastTime != 0L && dt <= timeout && dx <= slop) {
                    lastTime = 0L
                    c.consume()          // 让 Slider 忽略这一下，否则复位被覆盖
                    onReset()
                } else {
                    lastTime = c.uptimeMillis
                    lastPos = c.position
                }
            }
        }
    }

// ---- LR 式可视化色块滑杆（2026-10-03 用户点单，对齐桌面 widgets.GradientSlider）----
//
// 轨道画成"数值 → 颜色"的渐变条，拖动时直接读出"往什么颜色偏"。键 → 渐变
// stops 与桌面 `_DEV_GRADIENTS` / `_FILM_GRADIENTS` 一一对应（色温蓝→黄 /
// 色调绿→品红 / 色相与分离色调 0~360° 色环 / 光晕白→红橙）。
//
// 颜色在桌面降饱和（S×0.5、V×0.86、α165）的基础上**再淡一档**（S×0.35、
// V×0.92）——用户反馈"颜色再淡一点"；暗色主题上低饱和 + 高明度读作"淡"，
// 与桌面浅色主题的 α 混合等效。中性灰（两端中点）保持灰不偏色。
//
// 实现：渐变画在下层 Box（轨道 4dp 高、2dp 圆角，与桌面 groove 同几何），
// M3 Slider 叠在上层、轨道/刻度全透明——拖动、点按定位、双击复位、无障碍
// 全部继承 M3，只换轨道配色；渐变两端内缩 thumb 半径，与手柄行程对齐。

/** 精修面板：键 → 轨道渐变（stops 比例对应 [spec.min, spec.max] 区间）。 */
private val DEVELOP_GRADIENT_STOPS: Map<String, List<Pair<Float, Color>>> = mapOf(
    // 色温：冷（蓝）→ 中性 → 暖（黄），0 在中间
    "temp" to listOf(
        0.0f to rgb(70, 120, 255), 0.5f to rgb(128, 128, 128),
        1.0f to rgb(255, 196, 40)),
    // 色调：绿 → 中性 → 品红
    "tint" to listOf(
        0.0f to rgb(40, 190, 90), 0.5f to rgb(128, 128, 128),
        1.0f to rgb(232, 70, 226)),
    // 色相偏移 ±180° 与分离色调色相 0~360°：全色环（两端同为红，环形语义）
    "hue_shift" to hueRingStops(),
    "split_high_hue" to hueRingStops(),
    "split_low_hue" to hueRingStops(),
)

/** 胶片感面板：光晕色调 白 → 红橙（halation，与桌面 _FILM_GRADIENTS 同款）。 */
private val FILM_GRADIENT_STOPS: Map<String, List<Pair<Float, Color>>> = mapOf(
    "bloom_tint" to listOf(0.0f to rgb(255, 255, 255), 1.0f to rgb(255, 96, 30)),
)

/** 0~255 整数 RGB → 不透明的 Compose Color（渐变 stops 书写用）。 */
private fun rgb(r: Int, g: Int, b: Int): Color =
    Color(0xFF000000.toInt() or (r shl 16) or (g shl 8) or b)

/**
 * 普通/渐变滑杆共用的"淡一档"主题色（2026-10-03 用户反馈"普通参数滑块也淡一点"，
 * 同桌面 10-02"手柄不刺眼/色块降饱和"的规矩）：主题 primary 的 HSV S×0.6、
 * V×0.95——柔化但不改色相；渐变滑杆的手柄用它，与普通滑杆完全同色。
 */
@Composable
private fun mutedPrimary(): Color {
    val p = MaterialTheme.colorScheme.primary
    val hsv = FloatArray(3)
    android.graphics.Color.RGBToHSV(
        (p.red * 255f).toInt(), (p.green * 255f).toInt(), (p.blue * 255f).toInt(), hsv)
    hsv[1] *= 0.6f
    hsv[2] *= 0.95f
    return Color(android.graphics.Color.HSVToColor(hsv))
}

/**
 * 全部滑杆统一用这份 colors（10-03 用户反馈"滑块粗细统一 + 普通滑块淡一点"）：
 * 手柄与 active 轨道 = [mutedPrimary]，inactive/disabled 保持 M3 默认（本就低调）。
 * 注意**只换颜色不换几何**——M3 默认轨道 16dp 高是全局唯一口径，渐变滑杆的
 * 渐变条也画 16dp，粗细天然一致（桌面 GradientSlider 同款纪律）。
 */
@Composable
private fun appSliderColors() = SliderDefaults.colors(
    thumbColor = mutedPrimary(),
    activeTrackColor = mutedPrimary(),
)

/** 色相环七等分取色（与桌面 `_hue_stops` 同参：HSL L=0.62 / S=0.78）。 */
private fun hueRingStops(): List<Pair<Float, Color>> =
    (0..6).map { i -> i / 6f to Color.hsl(i * 60f, 0.78f, 0.62f) }

/** 桌面 `_muted` 的"再淡一档"版：S×0.35、V×0.92；无色相的中性灰保持灰。 */
private fun mutedStop(c: Color): Color {
    if (c.red == c.green && c.green == c.blue) return c
    val hsv = FloatArray(3)
    android.graphics.Color.RGBToHSV(
        (c.red * 255f).toInt(), (c.green * 255f).toInt(), (c.blue * 255f).toInt(), hsv)
    hsv[1] *= 0.35f
    hsv[2] *= 0.92f
    return Color(android.graphics.Color.HSVToColor(hsv))
}

/**
 * 渐变轨道滑杆：外观 = 渐变条 + M3 手柄，交互与 [Slider] 完全一致
 * （[doubleTapToReset] 挂外层 Box，覆盖整个滑杆区域）。
 * **几何与普通滑杆完全一致**（10-03 用户反馈"滑块粗细统一"——桌面 GradientSlider
 * 同款纪律"只换颜色不换尺寸"）：渐变条与 M3 默认轨道同高 16dp、2dp 圆角；
 * 左右内缩 10dp（手柄半径）与手柄行程对齐；手柄用 [mutedPrimary] 与普通滑杆同色。
 */
@Composable
private fun GradientSlider(
    stops: List<Pair<Float, Color>>,
    value: Float,
    onValueChange: (Float) -> Unit,
    valueRange: ClosedFloatingPointRange<Float>,
    enabled: Boolean,
    modifier: Modifier = Modifier,
) {
    val brush = Brush.horizontalGradient(
        *stops.map { (p, c) -> p to mutedStop(c) }.toTypedArray())
    Box(modifier = modifier.fillMaxWidth(), contentAlignment = Alignment.CenterStart) {
        // 与 M3 默认轨道**同长**（10-03 用户反馈"长度没一致"）：M3 的轨道通到
        // 滑杆全宽、但绘制区左右各内缩 TrackInsideCornerSize(2dp)——量图证实
        // 普通轨道两端各比容器边界短 ~2dp，故渐变条补同样的 2dp 内缩
        Box(
            Modifier
                .padding(horizontal = 2.dp)
                .fillMaxWidth()
                .height(16.dp)
                // 端部全圆（pill）= M3 默认轨道的端部圆角（TrackHeight/2）
                .background(brush, RoundedCornerShape(percent = 50))
                .alpha(if (enabled) 1f else 0.38f),
        )
        Slider(
            value = value,
            onValueChange = onValueChange,
            valueRange = valueRange,
            enabled = enabled,
            colors = SliderDefaults.colors(
                thumbColor = mutedPrimary(),
                activeTrackColor = Color.Transparent,
                inactiveTrackColor = Color.Transparent,
                activeTickColor = Color.Transparent,
                inactiveTickColor = Color.Transparent,
            ),
            modifier = Modifier.fillMaxWidth(),
        )
    }
}
