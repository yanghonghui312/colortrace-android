package com.colortrace.poc

import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.OpenableColumns
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInHorizontally
import androidx.compose.animation.slideOutHorizontally
import androidx.compose.animation.togetherWith
import androidx.compose.animation.core.tween
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import com.colortrace.DebugLog
import com.colortrace.poc.ui.ColortraceTheme
import com.colortrace.poc.ui.EditScreen
import com.colortrace.poc.ui.Motion
import com.colortrace.poc.ui.PresetLibrarySheet
import com.colortrace.poc.ui.SourceSheet
import com.colortrace.poc.ui.WelcomeScreen
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            // 深色默认：调色工作区纪律（与桌面 GUI 同一条，见 docs/ui-redesign-research §2）
            ColortraceTheme(darkTheme = true) {
                Surface(Modifier.fillMaxSize(),
                        color = MaterialTheme.colorScheme.background) {
                    AppRoot()
                }
            }
        }
    }
}

/** 屏幕级过渡：forward 用强调缓动右滑进，back 反向（M3 shared-axis 手感）。
 *  在 AnimatedContent.transitionSpec 里调用（targetState 即 forward 与否）。 */
private fun pageTransition(forward: Boolean) = if (forward) {
    (fadeIn(tween(Motion.PAGE_MS, easing = Motion.Emphasized)) +
            slideInHorizontally(tween(Motion.PAGE_MS, easing = Motion.Emphasized)) { it / 8 }
            ) togetherWith
            (fadeOut(tween(Motion.CONTENT_MS)) +
                    slideOutHorizontally(tween(Motion.CONTENT_MS)) { -it / 12 })
} else {
    (fadeIn(tween(Motion.PAGE_MS, easing = Motion.Emphasized)) +
            slideInHorizontally(tween(Motion.PAGE_MS, easing = Motion.Emphasized)) { -it / 8 }
            ) togetherWith
            (fadeOut(tween(Motion.CONTENT_MS)) +
                    slideOutHorizontally(tween(Motion.CONTENT_MS)) { it / 12 })
}

/** 照片条一项：uri 与缩略图**绑成一项**——两个独立列表在解码失败时会错位。 */
private data class PhotoItem(val uri: Uri, val thumb: Bitmap)

// 方法 id → 中文标签统一走 Labels.kt 的 methodLabelOf（2026-09-30 文案规范）：
// 此前这里与 EditScreen、PresetLibrarySheet 各存一份，已出现"pill 与选择器文案不一致"。
// 预设 kind ↔ 方法 id 的映射（methodIdOfKind）同样在 Labels.kt——会话恢复也要用。

/** 载入/切换会话时的保护档收敛：分区档强制 REGION（预设本色，及其恒走分区语义）；
 *  不支持 REGION 的档（plain 统计）从 REGION 回落到 OFF（不静默降级）。 */
private fun adoptProtect(s: Session, current: ProtectMode): ProtectMode = when {
    s.isRegion -> ProtectMode.REGION
    current == ProtectMode.REGION && s.methodId != "encoder" -> ProtectMode.OFF
    else -> current
}

/** 会话**当前**是否带分区数据/天然支持分区（encoder 路线 0、或已是分区档）。
 *  设备端 fit 的 plain 统计档不在此列——但可在切到「分区保护」时按样片**现场升档**
 *  （见 applyProtectMode），所以 UI 另按 `sessionSampleUri != null` 放行。 */
private fun supportsRegion(s: Session?): Boolean =
    s == null || s.isRegion || s.methodId == "encoder"

/** 库条目的默认应用档：encoder 优先（现役主路径），否则取第一档。 */
private fun defaultMethodOf(methods: List<String>): String =
    if ("encoder" in methods) "encoder" else methods.first()

/** 文件选择器返回的显示名（去扩展名）。拿不到返回 null。 */
private fun displayNameOf(uri: Uri, ctx: android.content.Context): String? =
    runCatching {
        ctx.contentResolver.query(uri, null, null, null, null)?.use { c ->
            val i = c.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (i >= 0 && c.moveToFirst()) c.getString(i) else null
        }
    }.getOrNull()?.substringBeforeLast('.')

@Composable
private fun AppRoot() {
    val context = LocalContext.current
    // 引擎进程级单例（P2.18）：与 BatchExportService 共享同一份模型与单车道调度器
    val repo = EngineBox.repository(context)
    val engineDispatcher = EngineBox.engineDispatcher
    val scope = androidx.compose.runtime.rememberCoroutineScope()

    var showEdit by remember { mutableStateOf(false) }
    var decoding by remember { mutableStateOf(false) }   // 解码/fit 等前置 IO
    var applying by remember { mutableStateOf(false) }   // 引擎计算中
    var error by remember { mutableStateOf<String?>(null) }
    var statusMsg by remember { mutableStateOf<String?>(null) }
    var exportText by remember { mutableStateOf("") }   // 保存/导出阶段的进度文案
    // 画布手势提示只在本次 Activity 生命周期内显示一次（文案规范 C-1）
    var canvasHintShown by remember { mutableStateOf(false) }
    val busy = decoding || applying

    // 样片 / 会话（来源二选一记下来，断点续传靠它在新进程里重建引擎）
    var session by remember { mutableStateOf<Session?>(null) }
    var sessionSampleUri by remember { mutableStateOf<String?>(null) }
    var sessionPresetJson by remember { mutableStateOf<String?>(null) }
    // 设备端 fit 的样片缩略图 PNG（P2.14 存库用；载入预设的会话从 style_thumb 反解）
    var sampleThumbPng by remember { mutableStateOf<ByteArray?>(null) }
    // 预设会话来自哪个库条目（P2.20 会话快照用：恢复时按 id 把**全部方法档**一起带回来）。
    // 参考图会话为 null——来源二选一，与 sessionSampleUri / sessionPresetJson 同源
    var sessionPresetId by remember { mutableStateOf<String?>(null) }
    // 预设会话的样片缩略（2026-10-01 用户点单：来源卡与预设库一样显示缩略图，
    // 不再只有通用星形图标）。按库条目 id 现取（应用时/会话恢复时）；参考图会话为 null
    var presetThumbBmp by remember { mutableStateOf<androidx.compose.ui.graphics.ImageBitmap?>(null) }

    // 预设库（P2.14）：条目列表 + 缩略图 + 命名对话框
    var showLibrary by remember { mutableStateOf(false) }
    var libraryEntries by remember { mutableStateOf(listOf<PresetStore.Entry>()) }
    var libraryThumbs by remember { mutableStateOf(mapOf<String, ImageBitmap?>()) }
    var namingPreset by remember { mutableStateOf(false) }
    var presetNameInput by remember { mutableStateOf("") }
    // 参考来源面板（P2.14）：编辑页内原地更换参考图/预设，不退回主界面
    var showSourceSheet by remember { mutableStateOf(false) }
    // 当前会话的来源名（从预设库/文件导入时记下，供来源面板显示）
    var sessionSourceName by remember { mutableStateOf<String?>(null) }

    // 照片：**单一列表**（uri 与缩略图绑成一项）——曾经用两个独立列表，
// 缩略图解码失败时长度不一致会导致索引错位（删错/选错图）
    var photos by remember { mutableStateOf(listOf<PhotoItem>()) }
    var activeIndex by remember { mutableIntStateOf(-1) }
    var activeBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var beforeImg by remember { mutableStateOf<androidx.compose.ui.graphics.ImageBitmap?>(null) }
    var afterImg by remember { mutableStateOf<androidx.compose.ui.graphics.ImageBitmap?>(null) }
    var overlayImg by remember { mutableStateOf<androidx.compose.ui.graphics.ImageBitmap?>(null) }

    // 参数（保护默认关：不一定每张照片都有人——有人时用户自己开）
    var strength by remember { mutableStateOf(0.5f) }
    var protect by remember { mutableStateOf(ProtectMode.OFF) }
    var protectStrength by remember { mutableStateOf(1f) }   // 保护强度 0~1（桌面滑杆同语义）
    var showMask by remember { mutableStateOf(false) }       // 显示保护区域（叠加层）
    // 方法档（P2.13）：选样片设备端 fit 用——encoder / reinhard / ot
    var fitMethod by remember { mutableStateOf("encoder") }
    // REGION 档未检测到人物（wSkin 峰值≈0）→ 提示改用肤色锁定（P2.11）
    var regionNoSkin by remember { mutableStateOf(false) }
    // 手动微调（P2.6）：enabled 开关 + 参数表（全键，缺省取 DevelopSpec.DEFAULTS）
    var developEnabled by remember { mutableStateOf(false) }
    var devValues by remember {
        mutableStateOf(com.colortrace.engine.DevelopSpec.DEFAULTS)
    }
    fun devActive() = developEnabled &&
            !com.colortrace.engine.DevelopSpec.isIdentity(devValues)
    // 胶片感（2026-10-02）：全 11 键参数表（三个 *_on 开关在内）；不进预设
    // （用户拍板：预设只互通追色），会话快照与批量续传携带。
    var filmValues by remember { mutableStateOf(com.colortrace.engine.FilmLayer.DEFAULTS) }

    // ---- 预设库（P2.14） ----

    /** 重新读库（列表 + 缩略图解码都在 IO 线程）。 */
    fun refreshLibrary() {
        scope.launch {
            val pair = withContext(Dispatchers.IO) {
                val es = PresetStore.list(context)
                val ts = HashMap<String, ImageBitmap?>()
                for (e in es) ts[e.id] =
                    runCatching { PresetStore.thumbBitmap(context, e.id)?.asImageBitmap() }
                        .getOrNull()
                es to ts
            }
            libraryEntries = pair.first
            libraryThumbs = pair.second
        }
    }

    /** 应用库条目：取默认档 → 预设会话（与桌面预设载入同路径）。
     *  P2.16：条目的**全部方法档**随会话携带（bundleVariants）——预设会话无样片像素，
     *  换方法 = 从条目取档载入（不再置灰，多档条目才行）。 */
    fun applyLibraryEntry(e: PresetStore.Entry) {
        scope.launch {
            decoding = true
            error = null
            try {
                val variants = LinkedHashMap<String, String>()
                for (k in e.methods) {
                    val text = withContext(Dispatchers.IO) {
                        PresetStore.methodJson(context, e.id, k)
                    } ?: continue   // 缺档不拦（defaultMethodOf 会避开）
                    variants[methodIdOfKind(k)] = text
                }
                val kind = defaultMethodOf(e.methods)
                val text = variants[methodIdOfKind(kind)]
                    ?: throw IllegalArgumentException("条目缺少 $kind 档")
                val s = withContext(engineDispatcher) { repo.loadPreset(text, variants) }
                // 分区预设强制 REGION（预设本色）；其余档若当前 REGION 但会话不支持则回 OFF
                protect = adoptProtect(s, protect)
                session = s
                sessionPresetJson = text
                sessionPresetId = e.id
                sessionSampleUri = null
                sampleThumbPng = null
                presetThumbBmp = withContext(Dispatchers.IO) {
                    PresetStore.thumbBitmap(context, e.id)?.asImageBitmap()
                }
                sessionSourceName = e.name
                fitMethod = s.methodId   // 与当前方法 pill 保持一致
                showLibrary = false
                showSourceSheet = false
                showEdit = true
                statusMsg = "已应用预设：${e.name}"
            } catch (ex: Exception) {
                DebugLog.e("预设库应用失败 ${e.id}", ex)
                error = "预设应用失败，请重试"
            } finally {
                decoding = false
            }
        }
    }

    fun deleteLibraryEntry(e: PresetStore.Entry) {
        scope.launch {
            withContext(Dispatchers.IO) { PresetStore.delete(context, e.id) }
            statusMsg = "已删除预设：${e.name}"
            DebugLog.i("预设库删除 id=${e.id} name=${e.name}")
            refreshLibrary()
        }
    }

    /** 存库命名对话框的默认名：方法档 + 时间（同名条目会覆盖更新）。 */
    fun openSavePreset() {
        val s = session ?: return
        val stamp = SimpleDateFormat("MM-dd HH:mm", Locale.getDefault()).format(Date())
        presetNameInput = "${methodLabelOf(s.methodId)} $stamp"
        namingPreset = true
    }

    /**
     * 存库（P2.16 一次 fit 全算法存一档）：参考图会话**一次 fit 全部 3 档**写入
     * `methods` map——encoder plain + reinhard/ot 的 region 变体（⊇ plain，见
     * `EngineRepository.fitAllSamplePresets`）；分割只跑一次。应用时默认 encoder、
     * 编辑页方法 pill 可换档（无样片像素也成立）。
     * 预设会话没有样片像素（入口本就只对参考图会话出现）——此分支为兜底语义。
     */
    fun commitSavePreset(name: String) {
        val s = session ?: return
        val sampleUri = sessionSampleUri
        scope.launch {
            decoding = true   // 多档 fit ≈1-2s，给 busy 反馈
            try {
                val methods: Map<String, JSONObject> = if (sampleUri != null) {
                    val bmp = withContext(Dispatchers.IO) {
                        repo.decodeCapped(Uri.parse(sampleUri), EngineRepository.PROC_MAX)
                    }
                    withContext(engineDispatcher) {
                        repo.fitAllSamplePresets(bmp)
                            .mapValues { (_, ss) -> JSONObject(ss.toPresetJson()) }
                    }
                } else {
                    mapOf(s.methodKind to JSONObject(s.toPresetJson()))
                }
                val thumb = sampleThumbPng ?: repo.sampleThumbPng(s)
                val entry = withContext(Dispatchers.IO) {
                    PresetStore.save(context, name, methods, thumb)
                }
                statusMsg = "已存入预设库：$name（${methods.size} 档）"
                DebugLog.i("预设库存档 id=${entry.id} name=$name kinds=${methods.keys} " +
                        "thumb=${thumb?.size ?: 0}B")
                refreshLibrary()
            } catch (ex: Exception) {
                DebugLog.e("存入预设库失败", ex)
                error = "存入预设库失败，请重试"
            } finally {
                decoding = false
            }
        }
    }

    // ---- 选择器 ----
    val pickSample = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri ->
        if (uri != null) {
            // 进程死后还能重读这张样片（断点续传的前提）；个别来源可能不支持，
            // 失败只记日志不拦截（本次会话照常，续传时才发现读不了）
            runCatching {
                context.contentResolver.takePersistableUriPermission(
                    uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }.onFailure { DebugLog.e("样片 uri 权限持久化失败（续传可能不可用）", it) }
            scope.launch {
                decoding = true
                error = null
                try {
                    val bmp = withContext(Dispatchers.IO) {
                        repo.decodeCapped(uri, EngineRepository.PROC_MAX)
                    }
                    sampleThumbPng = withContext(Dispatchers.IO) {
                        PresetStore.encodeThumbPng(bmp)
                    }
                    val s = withContext(engineDispatcher) {
                        repo.fitSample(bmp, fitMethod, region = protect == ProtectMode.REGION)
                    }
                    session = s
                    protect = adoptProtect(s, protect)   // 分区档 fit ⇒ 自动选中分区保护
                    sessionSampleUri = uri.toString()
                    sessionPresetJson = null
                    sessionPresetId = null
                    sessionSourceName = null
                    presetThumbBmp = null   // 来源切回参考图，预设缩略随之清掉
                    showSourceSheet = false
                    showEdit = true
                    statusMsg = "参考图解析完成（${methodLabelOf(s.methodId)}）——加入照片开始追色"
                } catch (e: Exception) {
                    DebugLog.e("样片解析失败 $uri", e)
                    error = "参考图解析失败，请换一张试试"
                } finally {
                    decoding = false
                }
            }
        }
    }
    val pickPreset = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        if (uri != null) {
            scope.launch {
                decoding = true
                error = null
                try {
                    val text = withContext(Dispatchers.IO) {
                        context.contentResolver.openInputStream(uri)!!
                            .bufferedReader().readText()
                    }
                    // 统一解析（P2.17）：bundle（预设库导出文件）与单档预设 JSON 都吃。
                    // 校验在 parseImportableBundle 里做——本机不支持的档跳过并回报，
                    // 全部不可用才报错（不静默降级、不静默丢弃）
                    val imported = withContext(engineDispatcher) {
                        repo.parseImportableBundle(text, displayNameOf(uri, context)
                            ?: "导入预设")
                    }
                    withContext(Dispatchers.IO) {
                        PresetStore.save(context, imported.name, imported.methods,
                                         imported.sampleThumb)
                    }
                    statusMsg = "已导入预设库：${imported.name}（${imported.methods.size} 档" +
                            (if (imported.skippedKinds.isEmpty()) ""
                             else "，跳过不支持的档：" +
                                     imported.skippedKinds.joinToString("、") { methodLabelOf(it) }) +
                            "）"
                    DebugLog.i("预设导入入库 $uri → ${imported.name} " +
                            "kinds=${imported.methods.keys} skipped=${imported.skippedKinds}")
                    fitMethod = methodIdOfKind(defaultMethodOf(imported.methods.keys.toList()))
                    // （2026-10-01 修）**导入只入库，不动当前会话**——此前这里写
                    // `sessionSourceName = imported.name`，会把正在编辑的预设会话的
                    // 来源名改成刚导入的那个文件名（来源面板与快照都跟着错标）。
                    refreshLibrary()
                } catch (e: Exception) {
                    DebugLog.e("预设导入失败 $uri", e)
                    error = "预设导入失败，文件可能已损坏"
                } finally {
                    decoding = false
                }
            }
        }
    }
    // ---- 预设导出（P2.17）：库条目 → SAF CreateDocument 存 bundle JSON。
    //     文件 = PresetStore.load 的原样 bundle（全部档 + 样片缩略图），桌面端
    //     将来读 methods 子项即可复用同一份格式。 ----
    var pendingExportEntry by remember { mutableStateOf<PresetStore.Entry?>(null) }
    val exportPreset = rememberLauncherForActivityResult(
        ActivityResultContracts.CreateDocument("application/json")
    ) { uri ->
        val e = pendingExportEntry
        pendingExportEntry = null
        if (uri != null && e != null) {
            scope.launch {
                try {
                    val bundleText = withContext(Dispatchers.IO) {
                        PresetStore.load(context, e.id)?.toString()
                            ?: throw IllegalArgumentException("条目 bundle 已不存在")
                    }
                    withContext(Dispatchers.IO) {
                        context.contentResolver.openOutputStream(uri)!!.use { os ->
                            os.write(bundleText.toByteArray(Charsets.UTF_8))
                            os.flush()
                        }
                    }
                    statusMsg = "已导出预设：${e.name}"
                    DebugLog.i("预设导出 id=${e.id} name=${e.name} kinds=${e.methods} → $uri")
                } catch (ex: Exception) {
                    DebugLog.e("预设导出失败 ${e.id}", ex)
                    error = "预设导出失败，请重试"
                }
            }
        }
    }
    fun exportLibraryEntry(e: PresetStore.Entry) {
        // SAF 建议文件名：条目名不一定合法（/ \ : 等）——清洗后再拼扩展名
        val safe = e.name.replace(Regex("[\\\\/:*?\"<>|]"), "_").trim().ifEmpty { "预设" }
        pendingExportEntry = e
        exportPreset.launch("$safe.colortrace.json")
    }
    val pickPhotos = rememberLauncherForActivityResult(
        ActivityResultContracts.PickMultipleVisualMedia(maxItems = 30)
    ) { uris ->
        if (uris.isNotEmpty()) {
            scope.launch {
                decoding = true
                try {
                    // 去重：照片条的 key 用 uri，重复 uri 会让 LazyRow 抛
                    // "Key was already used" 崩溃（实测）；同一张图加两次也无意义
                    val existing = photos.map { it.uri }.toSet()
                    val fresh = uris.filter { it !in existing }
                    // 每张都持久化读权限：崩溃后重启，续传还要按 uri 重读原图
                    for (u in fresh) runCatching {
                        context.contentResolver.takePersistableUriPermission(
                            u, Intent.FLAG_GRANT_READ_URI_PERMISSION)
                    }.onFailure { DebugLog.e("照片 uri 权限持久化失败 $u", it) }
                    // uri 与缩略图绑成一项；解码失败的整项跳过（不留孤儿索引）
                    val items = withContext(Dispatchers.IO) {
                        fresh.mapNotNull { u ->
                            runCatching {
                                PhotoItem(u, repo.decodeCapped(u, EngineRepository.THUMB_MAX))
                            }.onFailure {
                                DebugLog.e("缩略图解码失败 $u（该项已跳过）", it)
                            }.getOrNull()
                        }
                    }
                    if (items.isEmpty()) {
                        statusMsg = "这张照片已在列表中"
                    } else {
                        photos = photos + items
                        if (activeIndex < 0) activeIndex = 0
                        statusMsg = "已加入 ${items.size} 张照片"
                    }
                    DebugLog.i("加入照片 请求=${uris.size} 去重后=${fresh.size} " +
                            "成功=${items.size}")
                } finally {
                    decoding = false
                }
            }
        }
    }

    // 选中照片变化 → 解码预览并设为 before。
    // 注意：key 用**当前项的 uri**（而非整个列表）——追加照片时不必重解码当前图；
    // 且必须清 afterImg：否则上一张的结果会挂在新图上（用户实测的"残留"）
    LaunchedEffect(activeIndex, photos.getOrNull(activeIndex)?.uri) {
        val item = photos.getOrNull(activeIndex)
        if (item == null) return@LaunchedEffect
        decoding = true
        afterImg = null
        overlayImg = null
        try {
            val bmp = withContext(Dispatchers.IO) {
                repo.decodeCapped(item.uri, EngineRepository.PREVIEW_MAX)
            }
            activeBitmap = bmp
            beforeImg = bmp.asImageBitmap()
        } catch (e: Exception) {
            DebugLog.e("照片打开失败 ${item.uri}", e)
            statusMsg = "无法打开这张照片"
        } finally {
            decoding = false
        }
    }

    // ---- 迁移缓存（P2.7）：key 只含「改变映射本身的量」（样片/照片/保护档）；
    //  strength 与 develop 不进 key——拖微调、拖强度只重跑 render（桌面同纪律） ----
    var migration by remember { mutableStateOf<TiledPipeline.Migration?>(null) }
    var migrationKey by remember { mutableStateOf<Triple<Session, Bitmap, ProtectMode>?>(null) }

    // Effect A：迁移（重建 src/LUT/分割/权重 + 物化全强度 LUT 输出）
    LaunchedEffect(session, activeBitmap, protect) {
        val s = session
        val bmp = activeBitmap
        if (s == null || bmp == null) return@LaunchedEffect
        val key = Triple(s, bmp, protect)
        if (migrationKey == key && migration != null) {
            DebugLog.i("migrate: cache hit")
            return@LaunchedEffect
        }
        delay(200)   // 防抖：连续换图/换档只跑最后一次（可取消）
        applying = true
        try {
            // NonCancellable：Mat 分配中途取消无法回收，迁移一次做完再存
            withContext(engineDispatcher + NonCancellable) {
                val built = TiledPipeline(s.model, repo.selfieNet).migrate(bmp, protect)
                val old = migration
                migration = built
                migrationKey = key
                old?.release()
                // 分区保护是否真的检测到人：wSkin 峰值（无人图整张 ≈0）
                regionNoSkin = built.tiled != null &&
                    TiledPipeline(s.model, repo.selfieNet)
                        .regionSkinPeak(built) < 0.1f
            }
        } catch (ce: kotlinx.coroutines.CancellationException) {
            // 同出图 Effect：取消不是失败，原样上抛（防假"追色失败"挂状态栏）
            throw ce
        } catch (e: Exception) {
            regionNoSkin = false
            DebugLog.e("迁移失败 mode=$protect ${bmp.width}x${bmp.height}", e)
            statusMsg = "追色失败，请重试"
        } finally {
            applying = false
        }
    }

    // Effect B：出图（迁移命中 → 只 render；拖微调/拖强度/拖保护强度走这里）
    LaunchedEffect(migration, strength, developEnabled, devValues, protectStrength,
                   filmValues) {
        val m = migration ?: return@LaunchedEffect
        val s = session ?: return@LaunchedEffect
        // 守卫：切图窗口期内 migration 仍是上一张的 ⇒ 绝不能拿它出图，
        // 否则旧结果会被写回 afterImg 并贴到新图上（用户实测的"残留"）
        if (migrationKey?.second !== activeBitmap) return@LaunchedEffect
        DebugLog.i("render effect: strength=$strength protect=$protectStrength " +
                "dev=$developEnabled devActive=${devActive()}")
        delay(120)   // 微调滑杆防抖
        applying = true
        try {
            val result = withContext(engineDispatcher + NonCancellable) {
                TiledPipeline(s.model, repo.selfieNet)
                    .renderResult(m, strength, if (devActive()) devValues else null,
                                  protectStrength,
                                  if (com.colortrace.engine.FilmLayer.active(filmValues))
                                      filmValues else null)
            }
            afterImg = result.bitmap.asImageBitmap()
            // 预览的 film 失败不再静默（与导出路径同一句话；日志有因）
            if (!result.filmApplied) {
                statusMsg = "胶片效果未应用（处理失败，详见运行日志）"
            }
        } catch (ce: kotlinx.coroutines.CancellationException) {
            // 离开组合/切图的协程取消**不是失败**——原样上抛。否则 statusMsg 会挂上
            // 假"追色失败"（回编辑页还看得见）且错误日志被噪音淹没（10-02 晚实机 8 连）
            throw ce
        } catch (e: Exception) {
            DebugLog.e("出图失败 strength=$strength dev=${devActive()}", e)
            statusMsg = "追色失败，请重试"
        } finally {
            applying = false
        }
    }

    // ---- 保护区域叠加（仅在开关打开时算；mask 与生效保护同源） ----
    // 独立于 afterImg：保存/分享用未叠加的结果
    LaunchedEffect(showMask, migration, protectStrength, protect) {
        val m = migration
        val s = session
        if (!showMask || m == null || s == null || m.mode == ProtectMode.OFF) {
            overlayImg = null
            return@LaunchedEffect
        }
        val mask = withContext(engineDispatcher + NonCancellable) {
            TiledPipeline(s.model, repo.selfieNet).protectionMask(m, protectStrength)
        }
        overlayImg = protectionOverlayBitmap(mask, m.w, m.h).asImageBitmap()
        DebugLog.i("保护区域叠加 ${m.w}x${m.h} protect=$protectStrength mode=$protect")
    }

    /** 预设会话换方法（P2.16）：库条目多档 → 取对应档 JSON 重新 loadPreset。
     *  换的是**完整预设对象**（各档独立 fit 过），无 refit、无样片像素依赖；
     *  会话替换后迁移缓存自动失效。保护档不降级：encoder 天然支持 REGION、
     *  统计档存的是 region 变体（isRegion）⇒ adoptProtect 原样放行。
     *  （声明在 switchMethod 之前——Kotlin 局部函数必须先声明后使用。） */
    fun switchPresetVariant(newMethod: String, json: String) {
        val variants = session?.bundleVariants ?: emptyMap()
        scope.launch {
            applying = true
            exportText = "正在切换到 ${methodLabelOf(newMethod)}…"
            try {
                val s = withContext(engineDispatcher) {
                    repo.loadPreset(json, variants)   // variants 继续携带，还能再换
                }
                protect = adoptProtect(s, protect)
                session = s
                sessionPresetJson = json   // 续传快照跟当前档（其余档随进程结束，见 README 边界）
                fitMethod = newMethod
                statusMsg = "已切换到 ${methodLabelOf(newMethod)}，正在重新追色"
                DebugLog.i("切换预设档 $newMethod（库条目 ${variants.size} 档）")
            } catch (e: Exception) {
                DebugLog.e("切换预设档失败 $newMethod", e)
                statusMsg = "切换方法失败，请重试"
            } finally {
                applying = false
                exportText = ""
            }
        }
    }

    // ---- 编辑页切换方法（P2.13）：设备端 fit 会话 = 同一样片重新 fit（session 更新后
    // 迁移缓存 key（含 session 对象）自动失效，Effect A 重跑）。P2.16 起**预设会话**
    // 也可换：条目带多档时（bundleVariants）从条目取档载入——无样片像素也成立。 ----
    fun switchMethod(newMethod: String) {
        if (newMethod == session?.methodId) return
        if (sessionSampleUri == null) {
            // 预设会话：从库条目的 methods map 取档（无档 = 不可换，UI 已置灰）
            val json = session?.bundleVariants?.get(newMethod) ?: return
            switchPresetVariant(newMethod, json)
            return
        }
        fitMethod = newMethod
        // 2026-09-30 合并：档位不跟着方法走——分区保护下换算法，新会话按**同张样片**
        // 同样带分区数据（encoder 天然支持 REGION；统计类靠 region=true 升档）
        val wantRegion = protect == ProtectMode.REGION
        scope.launch {
            applying = true
            exportText = "正在切换到 ${methodLabelOf(newMethod)}…"
            try {
                val bmp = withContext(Dispatchers.IO) {
                    repo.decodeCapped(Uri.parse(sessionSampleUri!!), EngineRepository.PROC_MAX)
                }
                val s = withContext(engineDispatcher) {
                    repo.fitSample(bmp, newMethod, region = wantRegion)
                }
                session = s          // migrationKey 失效 → Effect A 自动重跑
                statusMsg = "已切换到 ${methodLabelOf(newMethod)}，正在重新追色"
                DebugLog.i("切换方法 $newMethod（样片 $sessionSampleUri）")
            } catch (e: Exception) {
                DebugLog.e("切换方法失败 $newMethod", e)
                statusMsg = "切换方法失败，请重试"
            } finally {
                applying = false
                exportText = ""
            }
        }
    }

    /**
     * 保护模式切换（2026-09-30 合并的关键接缝）：选「分区保护」时，若当前会话**没有**
     * 分区数据、但**样片像素在手**（设备端 fit 会话），按**同张样片**现场升档成分区档
     * ——方法选择器因此只需 3 档算法。分区档内含 plain 的 `global` ⇒ 升档不换调子
     * （OFF/肤色锁定仍走全局映射）。预设会话无样片像素 ⇒ 不升档（该情形 UI 已置灰）。
     */
    fun applyProtectMode(mode: ProtectMode) {
        protect = mode
        if (mode != ProtectMode.REGION) return
        val s = session ?: return
        if (s.isRegion || s.methodId == "encoder") return
        val sampleUri = sessionSampleUri ?: return
        scope.launch {
            applying = true
            exportText = "正在建立分区…"
            try {
                val bmp = withContext(Dispatchers.IO) {
                    repo.decodeCapped(Uri.parse(sampleUri), EngineRepository.PROC_MAX)
                }
                val upgraded = withContext(engineDispatcher) {
                    repo.fitSample(bmp, s.methodId, region = true)
                }
                session = upgraded          // migrationKey 失效 → Effect A 重跑
                statusMsg = "已启用分区保护（${methodLabelOf(s.methodId)}）"
                DebugLog.i("分区升档 ${s.methodId}（样片 $sampleUri）")
            } catch (e: Exception) {
                DebugLog.e("分区升档失败 ${s.methodId}", e)
                statusMsg = "分区保护启用失败"
                protect = ProtectMode.CHROMA   // 回落仍可用的档（不静默假装分区）
            } finally {
                applying = false
                exportText = ""
            }
        }
    }

    // ---- 删除照片（照片条 × 按钮） ----
    fun removePhoto(index: Int) {
        if (index !in photos.indices) return
        val removingActive = index == activeIndex
        val next = photos.toMutableList().also { it.removeAt(index) }
        photos = next
        when {
            next.isEmpty() -> {
                activeIndex = -1
                activeBitmap = null
                beforeImg = null
                afterImg = null
                overlayImg = null
                migration?.release()
                migration = null
                migrationKey = null
            }
            removingActive -> {
                // 删的是当前项 → 落到相邻项（索引与画布状态由上面那个 Effect 重建）
                activeIndex = index.coerceAtMost(next.size - 1)
                activeBitmap = null
                beforeImg = null
                afterImg = null
                overlayImg = null
                migration?.release()
                migration = null
                migrationKey = null
            }
            index < activeIndex -> activeIndex -= 1     // 前面的项被删，索引前移
        }
        statusMsg = "已移除 1 张，还剩 ${next.size} 张"
        DebugLog.i("移除照片 index=$index 剩=${next.size} activeIndex=$activeIndex")
    }

    // 离开组合（Activity 销毁）时释放迁移缓存的原生内存
    DisposableEffect(Unit) {
        onDispose {
            migration?.release()
            migration = null
            migrationKey = null
        }
    }

    // ---- 保存（分辨率：原始尺寸 或 预压缩短边 1920；原图解码 → 分块处理 → 相册） ----
    // 分享入口已按用户反馈取消（2026-09-29）：只保留"保存到相册"
    // P2.21：多一档"预压缩"（[ExportSize.SOCIAL]）——缩放在**解码**这一步做，
    // 与批量导出同一入口 [EngineRepository.decodeForExport]（见 BatchCore 里的等价性说明）
    fun saveResult(size: ExportSize) {
        val item = photos.getOrNull(activeIndex)
        if (item == null || session == null) return
        scope.launch {
            applying = true
            exportText = if (size == ExportSize.ORIGINAL) "正在读取原图…"
                         else "正在按短边 ${size.shortSide} 缩放…"
            statusMsg = null
            try {
                // 原始尺寸：inSample=1 解码原图（预览一直 ≤1024，导出不受此限）
                val full = withContext(Dispatchers.IO) {
                    repo.decodeForExport(item.uri, size.shortSide)
                }
                // 流式导出（P2.12）：接管 full（内部回收），直接产 JPEG 字节
                val ex = withContext(engineDispatcher) {
                    TiledPipeline(session!!.model, repo.selfieNet)
                        .processToJpeg(full, protect, strength,
                                       if (devActive()) devValues else null,
                                       protectStrength,
                                       film = com.colortrace.engine.FilmLayer
                                           .normalize(filmValues)) { d, n ->
                            exportText = "正在导出 ${if (n > 0) d * 100 / n else 0}%"
                        }
                }
                val saved = withContext(Dispatchers.IO) {
                    repo.saveJpegBytes(ex.bytes, "colortrace_${System.currentTimeMillis()}")
                } ?: throw IllegalStateException("无法创建输出文件")
                // 文案里报**实际**像素：预压缩档若原图更小则不会放大，用户能一眼看出来
                var msg = (if (saved.second) "已保存到相册（${ex.w}×${ex.h}）"
                           else "已保存到应用目录（${ex.w}×${ex.h}）")
                // 胶片层失败被整层跳过时不再静默（2026-10-02 全尺寸 OOM 三连：
                // 出图"没感觉"其实是柔光/光晕/颗粒全没跑）——用户必须在保存
                // 提示里看到，而不是自己对比预览才发现
                if (!ex.filmApplied) msg += "；胶片效果未应用（处理失败，详见运行日志）"
                statusMsg = msg
                DebugLog.i("保存完成 ${ex.w}x${ex.h} 导出=${size.label} " +
                        "mode=$protect protect=$protectStrength s=$strength " +
                        "dev=${devActive()} 相册=${saved.second}")
            } catch (e: Exception) {
                if (e is kotlinx.coroutines.CancellationException) throw e
                DebugLog.e("保存失败（${size.label}）", e)
                statusMsg = "保存失败，请检查存储空间"
            } finally {
                applying = false
                exportText = ""
            }
        }
    }

    // ---- 批量导出：P2.18 起执行权在 BatchExportService（前台服务）——
    // 切后台照常跑、通知栏进度；这里只写快照 + 启动服务 + 订阅进度。 ----

    // API 33+ 需要 POST_NOTIFICATIONS 运行时授权才能显示进度通知；
    // 拒绝也不拦截——服务照跑，进度回 App 内看（BatchBus）。
    // （声明在 exportAll 之前——Kotlin 局部声明必须先声明后使用。）
    val notifPermLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (!granted) {
            DebugLog.i("通知权限被拒——批量照跑，进度仅在 App 内显示")
            // 文案规范：拒绝后要给用户一句说明，否则"切后台看不到进度"没有任何解释
            statusMsg = "未开启通知权限：批量导出照常进行，但切到其他应用后看不到进度"
        }
        BatchExportService.start(context)
    }

    fun startBatchService() {
        if (Build.VERSION.SDK_INT >= 33 &&
            context.checkSelfPermission(android.Manifest.permission.POST_NOTIFICATIONS)
                != android.content.pm.PackageManager.PERMISSION_GRANTED
        ) {
            notifPermLauncher.launch(android.Manifest.permission.POST_NOTIFICATIONS)
        } else {
            BatchExportService.start(context)
        }
    }

    /** 用户点「批量导出全部」：把**当前参数 + 会话来源 + 导出尺寸**快照落盘，交给前台
     *  服务跑（服务按快照自举重建会话——等价性已被强杀 E2E 的 MD5 复现证明）。 */
    fun exportAll(size: ExportSize) {
        val s = session
        if (s == null || photos.isEmpty()) return
        // 已有批量在后台跑时禁止再发：快照文件是单槽，覆盖会毁掉在跑批量的续传标记
        if (BatchBus.state.value?.running == true) {
            statusMsg = "已有批量导出正在进行（见通知栏），请等它结束"
            return
        }
        val job = BatchResumeStore.BatchJob(
            sampleUri = sessionSampleUri,
            presetJson = sessionPresetJson,
            mode = protect.name,
            strength = strength,
            protectStrength = protectStrength,
            developEnabled = developEnabled,
            devValues = devValues,
            uris = photos.map { it.uri.toString() },
            method = if (sessionPresetJson == null) fitMethod else "encoder",
            done = emptySet(),
            shortSide = size.shortSide,
            filmValues = filmValues)
        scope.launch {
            withContext(Dispatchers.IO) { BatchResumeStore.save(context, job) }
            statusMsg = "正在开始批量导出…"
            DebugLog.i("批量导出交给前台服务 跑=${job.uris.size} mode=${job.mode} " +
                    "导出=${size.label}")
            startBatchService()
        }
    }

    // ---- 断点续传：启动时查状态文件，有未完成的批量就询问（P2.10） ----
    var resumeJob by remember { mutableStateOf<BatchResumeStore.BatchJob?>(null) }
    LaunchedEffect(Unit) {
        resumeJob = withContext(Dispatchers.IO) { BatchResumeStore.load(context) }
            ?.takeIf { it.pending.isNotEmpty() }
    }

    // 后台批量进度（P2.18）：执行权在前台服务，**App 内镜像旧版体验**——批量进行中
    // busy 覆盖层 + 步级文案照常显示；切到其他应用时服务不停，通知栏接管进度，
    // 回到 App 后 StateFlow 重放当前状态，覆盖层自动恢复。
    LaunchedEffect(Unit) {
        BatchBus.state.collect { st ->
            if (st == null) return@collect
            if (st.running) {
                applying = true
                exportText = st.text
            } else {
                applying = false
                exportText = ""
                statusMsg = st.text
            }
        }
    }

    /** 续传：不再在 Activity 内重建会话跑批（P2.18）——交给前台服务按快照自举，
     *  与普通批量同一条路径；期间可离开 App，体验与普通批量一致。 */
    fun resumeBatch(job: BatchResumeStore.BatchJob) {
        resumeJob = null
        statusMsg = "正在开始批量导出…"
        DebugLog.i("断点续传交给前台服务 待跑=${job.pending.size}/${job.uris.size}")
        startBatchService()
    }

    // ---- 会话快照（P2.20）：覆盖安装 / 被杀 / 系统回收后「继续上次编辑」 ----
    // 写：来源 + 照片条 + 全部参数，防抖落盘（见下方 Effect）。
    // 读：**不在启动路径上重建**——refit 与首次选图同价（1~2s），压在冷启动里
    //     等于每次打开都白付一次；改为欢迎屏一张卡片，用户点一下才重建。
    var resumeSnapshot by remember { mutableStateOf<SessionSnapshotStore.Snapshot?>(null) }
    LaunchedEffect(Unit) {
        resumeSnapshot = withContext(Dispatchers.IO) { SessionSnapshotStore.load(context) }
    }

    /** 库条目的全部档（methodId → 预设 JSON），与 [applyLibraryEntry] 同一构造——
     *  实现挪到 [SessionRestore.variantsOf]（会话恢复与库应用共用一份）。 */
    suspend fun variantsOfEntry(id: String?): Map<String, String> =
        SessionRestore.variantsOf(context, id)

    /** 用户点「继续上次编辑」：会话还在内存（只是退回欢迎屏）就直接回编辑页；
     *  否则按快照自举——参考图重解码 + refit，或预设重载（变体档随条目带回）。
     *  重建主体在 [SessionRestore.rebuild]（可被插桩测试直接驱动）。 */
    fun resumeLastSession() {
        if (session != null) {
            showEdit = true
            return
        }
        val snap = resumeSnapshot ?: return
        scope.launch {
            decoding = true
            exportText = "正在恢复上次编辑…"
            error = null
            try {
                val r = SessionRestore.rebuild(repo, context, snap,
                                               io = Dispatchers.IO,
                                               engine = engineDispatcher)
                // 一次性灌回状态（同一帧内无挂起点 ⇒ 下面的落盘 Effect 只会看到终态）
                session = r.session
                sessionSampleUri = r.sampleUri
                sessionPresetJson = r.presetJson
                sessionPresetId = r.presetId
                sessionSourceName = r.presetName
                sampleThumbPng = r.sampleThumbPng
                presetThumbBmp = withContext(Dispatchers.IO) {
                    // 库条目会话恢复缩略；文件导入无条目（presetId=null）退回星形图标
                    r.presetId?.let { PresetStore.thumbBitmap(context, it)?.asImageBitmap() }
                }
                protect = runCatching { ProtectMode.valueOf(snap.protect) }
                    .getOrDefault(ProtectMode.OFF)
                strength = snap.strength
                protectStrength = snap.protectStrength
                developEnabled = snap.developEnabled
                // 缺键的旧快照按默认值补齐（DevelopSpec.DEFAULTS 是全键表）
                devValues = com.colortrace.engine.DevelopSpec.DEFAULTS + snap.devValues
                filmValues = com.colortrace.engine.FilmLayer.normalize(snap.filmValues)
                fitMethod = r.session.methodId
                photos = r.photos.map { PhotoItem(it.first, it.second) }
                activeIndex = r.activeIndex
                showEdit = true
                // 跳过数要点名：只报恢复张数的话，少了几张用户未必注意（10-03 审计）
                statusMsg = "已恢复上次编辑（${r.photos.size} 张照片" +
                        (if (r.skippedPhotos > 0)
                            "，另 ${r.skippedPhotos} 张原图已不可读被跳过" else "") + "）"
                DebugLog.i("恢复会话 method=${r.session.methodId} protect=$protect " +
                        "照片=${r.photos.size}/${snap.photoUris.size} s=$strength")
            } catch (e: Exception) {
                // 参考图/预设已不可读 = 无从重建：清掉快照，别让"每次启动都失败"的空壳常驻
                DebugLog.e("恢复上次会话失败（清除快照）", e)
                withContext(Dispatchers.IO) { SessionSnapshotStore.clear(context) }
                resumeSnapshot = null
                error = "上次的会话已无法恢复（参考图或预设已不可读）"
            } finally {
                decoding = false
                exportText = ""
            }
        }
    }

    /** 快照落盘 Effect：来源/照片/参数任一变化 → 防抖 0.6s 后原子写。
     *  只在**有会话**时写（没有会话就没有可恢复的现场，别留空壳让欢迎屏多问一句）；
     *  防抖的意义与 Effect A/B 同款：拖强度/精修滑杆期间不写盘，停手才写。 */
    val snapshotPhotoUris = photos.map { it.uri.toString() }
    LaunchedEffect(session, sessionSampleUri, sessionPresetJson, sessionPresetId,
                   sessionSourceName, fitMethod, snapshotPhotoUris, activeIndex,
                   strength, protect, protectStrength, developEnabled, devValues,
                   filmValues) {
        val s = session ?: return@LaunchedEffect
        delay(600)
        val snap = SessionSnapshotStore.Snapshot(
            sampleUri = sessionSampleUri,
            presetId = sessionPresetId,
            presetName = sessionSourceName,
            presetJson = sessionPresetJson,
            method = s.methodId,
            photoUris = snapshotPhotoUris,
            activeIndex = activeIndex,
            strength = strength,
            protect = protect.name,
            protectStrength = protectStrength,
            developEnabled = developEnabled,
            devValues = devValues,
            filmValues = filmValues,
        )
        withContext(Dispatchers.IO) { SessionSnapshotStore.save(context, snap) }
        DebugLog.i("会话快照已更新 method=${snap.method} 照片=${snap.photoUris.size} " +
                "protect=${snap.protect}")
    }

    // 编辑页的"当前方法/是否支持分区保护"以**会话**为准（P2.14：从预设库或文件
    // 载入的会话，其方法档可能与欢迎屏上的选择不同——按 fitMethod 显示会错标）
    val sessionMethodId = session?.methodId ?: fitMethod
    // 方法 pill（P2.16）：设备端 fit 会话可换（refit）；预设会话条目带多档也可换
    //（从条目取档载入）；单档预设条目保持置灰。下拉只列**可用**档。
    val canSwitchMethod = sessionSampleUri != null || (session?.bundleVariants?.size ?: 0) > 1
    val methodOptions: List<String> =
        session?.bundleVariants?.keys?.toList()
            ?.takeIf { it.isNotEmpty() && sessionSampleUri == null }
        ?: listOf("encoder", "reinhard", "ot")
    // 设备端 fit 会话（有样片像素）即使当前是 plain 统计档也可「分区保护」——切过去会
    // 按同张样片现场升档（applyProtectMode）；预设会话则只有本身就带分区数据才放行
    val sessionSupportsRegion = supportsRegion(session) || sessionSampleUri != null
    val sourceIsReference = sessionSampleUri != null
    // 来源面板的细节行：参考图 → 方法档；预设 → 条目名（导入/应用时记下）
    val sourceDetail = if (sourceIsReference)
        "${methodLabelOf(sessionMethodId)} · 参考图已解析"
    else (sessionSourceName ?: "桌面预设")
    // 来源缩略（来源面板显示用）：参考图 = 样片缩略；预设 = 库条目 sampleThumb
    // （2026-10-01 用户点单，与预设库列表同图）
    val sourceThumbImg: ImageBitmap? = remember(sampleThumbPng, presetThumbBmp) {
        sampleThumbPng?.let { b ->
            android.graphics.BitmapFactory.decodeByteArray(b, 0, b.size)?.asImageBitmap()
        } ?: presetThumbBmp
    }
    // 欢迎屏「继续上次编辑」卡片（P2.20）：会话还在内存时按**当前**状态显示
    // （点一下直接回编辑页，不重建）；只剩快照时按快照内容显示，让用户知道
    // 自己要恢复的是什么（方法档 · 保护档 · 照片数）。null = 无可恢复的现场。
    val resumeDetail: String? = when {
        session != null ->
            "${methodLabelOf(sessionMethodId)} · ${protect.label} · ${photos.size} 张照片"
        resumeSnapshot != null -> {
            val snap = resumeSnapshot!!
            "${methodLabelOf(snap.method)} · ${protectModeLabelOf(snap.protect)} · " +
                    "${snap.photoUris.size} 张照片"
        }
        else -> null
    }

    AnimatedContent(
        targetState = showEdit,
        transitionSpec = { pageTransition(targetState) },
        label = "screens",
    ) { editing ->
        if (editing) {
            EditScreen(
                before = beforeImg,
                after = afterImg,
                overlay = overlayImg,
                busy = busy,
                strength = strength,
                protect = protect,
                protectStrength = protectStrength,
                method = sessionMethodId,
                methodOptions = methodOptions,
                canSwitchMethod = canSwitchMethod,
                regionSupported = sessionSupportsRegion,
                showMask = showMask,
                regionNoSkin = regionNoSkin,
                photos = photos.map { it.thumb.asImageBitmap() },
                photoKeys = photos.map { it.uri },
                activeIndex = activeIndex,
                statusMsg = statusMsg,
                busyText = exportText.ifEmpty { "正在追色…" },
                developEnabled = developEnabled,
                devValues = devValues,
                onDevelopToggle = { developEnabled = it },
                onDevelopValue = { key, v ->
                    // 拖滑杆即视为要用精调：总开关没开就自动打开
                    //（旧设计开关关着滑杆只改值不生效——用户拖了没反应会困惑）
                    devValues = devValues + (key to v)
                    if (!developEnabled) developEnabled = true
                },
                filmValues = filmValues,
                onFilmValue = { key, v -> filmValues = filmValues + (key to v) },
                canSave = afterImg != null,
                onBack = { showEdit = false },
                onStrength = { strength = it },
                onMethod = { newMethod -> switchMethod(newMethod) },
                onProtect = { applyProtectMode(it) },
                onProtectStrength = { protectStrength = it },
                onShowMask = { showMask = it },
                onSelectPhoto = { activeIndex = it },
                onRemovePhoto = { removePhoto(it) },
                onAddPhotos = {
                    pickPhotos.launch(PickVisualMediaRequest(
                        ActivityResultContracts.PickVisualMedia.ImageOnly))
                },
                onSave = { size -> saveResult(size) },
                onExportAll = { size -> exportAll(size) },
                sourceIsReference = sourceIsReference,
                sourceDetail = sourceDetail,
                sourceThumb = sourceThumbImg,
                onOpenSource = { showSourceSheet = true },
                showCanvasHint = !canvasHintShown,
                onCanvasHintShown = { canvasHintShown = true },
            )
        } else {
            WelcomeScreen(
                busy = busy,
                busyText = exportText.ifEmpty { "正在解析参考图…" },
                error = error,
                resumeDetail = resumeDetail,
                // 会话还在内存时不提供「放弃」——那会把现场变成够不着的孤儿状态
                canDiscardResume = session == null,
                onResume = { resumeLastSession() },
                onDiscardResume = {
                    scope.launch {
                        withContext(Dispatchers.IO) { SessionSnapshotStore.clear(context) }
                        resumeSnapshot = null
                        DebugLog.i("用户放弃了上次会话快照")
                    }
                },
                onPickSample = {
                    pickSample.launch(PickVisualMediaRequest(
                        ActivityResultContracts.PickVisualMedia.ImageOnly))
                },
                onOpenLibrary = {
                    refreshLibrary()
                    showLibrary = true
                },
            )
        }
    }

    // ---- 断点续传询问：上次批量没跑完（崩溃/被杀/有失败），问要不要接着导 ----
    resumeJob?.let { job ->
        AlertDialog(
            onDismissRequest = {
                scope.launch { withContext(Dispatchers.IO) { BatchResumeStore.clear(context) } }
                resumeJob = null
            },
            title = { Text("继续未完成的批量导出？") },
            text = {
                Text("上次批量导出没有完成：已完成 ${job.done.size}/${job.uris.size}，" +
                        "还剩 ${job.pending.size} 张。" +
                        "将按上次的设置（保护：${protectModeLabelOf(job.mode)} · " +
                        "强度 ${(job.strength * 100).toInt()}% · " +
                        "${ExportSize.labelOfShortSide(job.shortSide)}）继续导出，" +
                        "已完成的不会重复。")
            },
            confirmButton = {
                TextButton(onClick = { resumeBatch(job) }) { Text("继续导出") }
            },
            dismissButton = {
                TextButton(onClick = {
                    scope.launch {
                        withContext(Dispatchers.IO) { BatchResumeStore.clear(context) }
                    }
                    resumeJob = null
                }) { Text("不再继续") }
            })
    }

    // ---- 预设库面板（P2.14） ----
    if (showLibrary) {
        PresetLibrarySheet(
            entries = libraryEntries,
            thumbs = libraryThumbs,
            onApply = { applyLibraryEntry(it) },
            onDelete = { deleteLibraryEntry(it) },
            onExport = { exportLibraryEntry(it) },
            onImport = { pickPreset.launch("*/*") },
            onDismiss = { showLibrary = false },
        )
    }

    // ---- 参考来源面板（P2.14）：编辑页内原地换参考图/预设 ----
    if (showSourceSheet) {
        SourceSheet(
            isReference = sourceIsReference,
            detail = sourceDetail,
            sampleThumb = sourceThumbImg,
            canSavePreset = sessionSampleUri != null,
            onSwitchSample = {
                showSourceSheet = false
                pickSample.launch(PickVisualMediaRequest(
                    ActivityResultContracts.PickVisualMedia.ImageOnly))
            },
            onSwitchPreset = {
                showSourceSheet = false
                refreshLibrary()
                showLibrary = true
            },
            onSavePreset = {
                showSourceSheet = false
                openSavePreset()
            },
            onDismiss = { showSourceSheet = false },
        )
    }

    // ---- 存入预设库：命名 ----
    if (namingPreset) {
        AlertDialog(
            onDismissRequest = { namingPreset = false },
            title = { Text("将当前参考图存入预设") },
            text = {
                OutlinedTextField(
                    value = presetNameInput,
                    onValueChange = { presetNameInput = it },
                    singleLine = true,
                    label = { Text("预设名称（同名会覆盖）") },
                )
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        namingPreset = false
                        val name = presetNameInput.trim()
                        if (name.isNotEmpty()) commitSavePreset(name)
                    },
                    enabled = presetNameInput.isNotBlank(),
                ) { Text("存入") }
            },
            dismissButton = {
                TextButton(onClick = { namingPreset = false }) { Text("取消") }
            })
    }
}
