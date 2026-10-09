package com.colortrace.poc.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import com.colortrace.DebugLog

/**
 * 使用帮助面板（2026-09-30 用户要求）。
 *
 * **动机**：方法下拉里塞长简介会挤（折两行、断在词中间），而保护方式（肤色锁定 /
 * 分区保护）以及操作步骤、小技巧也需要有地方讲清楚。用户拍板：**新建一个帮助入口**
 * ——放在编辑页左上角返回键**旁边**，把「追色方法说明 / 保护方式说明 / 基本操作 /
 * 小技巧」统统收进来，主界面（下拉、调整页）保持简约整洁。
 *
 * 2026-10-01 追加两件：
 *  - **关于**（版本 / 引擎 / 模型 / 许可 / 反馈邮箱）：版本行直接用
 *    [DebugLog.startBanner]，用户能自己念出来，不必再教他跑 adb；
 *  - **查看运行日志**：面板内切到日志视图，读 `files/colortrace.log` 尾部并一键复制
 *    ——反馈问题时把日志贴过来，比"描述现象"有效得多。
 *
 * 内容口径与 `Labels.kt` 的档名、`ProtectMode.label` 一致；方法说明里的技术性质
 * 按项目实测写（Encoder 自训前馈·零样本；Reinhard 解析式最快；OT 高斯闭式解·保真）。
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HelpSheet(onDismiss: () -> Unit) {
    var showLog by remember { mutableStateOf(false) }
    ModalBottomSheet(onDismissRequest = onDismiss) {
        if (showLog) {
            LogView(onBack = { showLog = false })
        } else {
            HelpContent(onOpenLog = { showLog = true }, onDismiss = onDismiss)
        }
    }
}

@Composable
private fun HelpContent(onOpenLog: () -> Unit, onDismiss: () -> Unit) {
    Column(
        Modifier
            .fillMaxWidth()
            .navigationBarsPadding()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp),
    ) {
        Text("使用帮助", style = MaterialTheme.typography.titleLarge)

        Section("追色方法") {
            Entry(
                "AI 追色 · 自训前馈网络（Encoder）",
                "一次前向出图、无需迭代；跨内容零样本——任意参考图都能追，" +
                        "适用面最广，默认档。",
            )
            Entry(
                "经典统计 · Reinhard 统计迁移",
                "把画面 LAB 均值/方差对齐到参考图；纯解析式、速度最快，" +
                        "适合只调整体色调。",
            )
            Entry(
                "影调保真 · 高斯最优传输（OT）",
                "按整体色彩分布搬色调（闭式解、无迭代参数）；保留原图明暗关系，" +
                        "最不容易过火。",
            )
        }

        Section("保护方式") {
            Entry(
                "关闭",
                "不做肤色保护，整幅画面一起追色。",
            )
            Entry(
                "肤色锁定",
                "只锁住肤色的色相与饱和度，亮度跟随全局追色；没有色块边界，" +
                        "适合大多数场景，也是唯一能导出 LUT 的方式。",
            )
            Entry(
                "分区保护",
                "按人物语义分区（皮肤 / 头发 / 衣服 / 背景）分别迁移，皮肤保持原色；" +
                        "参考图里没有人物时会自动回退到全局映射，不会失效。",
            )
        }

        Section("胶片感") {
            Entry(
                "柔光",
                "只给亮部加一层柔光（发光），暗部保持深沉——画面发亮但不发灰，" +
                        "保留立体感；「范围」控制发光从高光渗到阴影的深度。",
            )
            Entry(
                "光晕",
                "把高光提取出来晕开再叠回去；「色调」从白色辉光滑到红橙色，" +
                        "得到老电影镜头的那种散射感。",
            )
            Entry(
                "颗粒",
                "胶片颗粒，只加在明度上（不会出彩噪）；「数量」管多少、" +
                        "「粗糙程度」管质感、「大小」管颗粒粗细。",
            )
            Entry(
                "启用方式",
                "三组各有独立开关，默认全关——打开即落在常用档位；" +
                        "开关关闭时该效果完全不参与出图（滑杆数值保留）。" +
                        "叠加顺序：追色 → 精修 → 胶片感。",
            )
        }

        Section("基本操作") {
            Step("1", "选一张参考图：想要的色调来源——别人的成片、喜欢的电影截图都行。")
            Step("2", "加入自己的照片：自动追色，可以一次加多张。")
            Step("3", "用「追色强度」控制力度；需要时打开保护，让肤色更自然。")
            Step("4", "轻触画布收起面板；拖动中间分割线对比原图；双击画布切换三种对比模式。")
            Step(
                "5",
                "想微调：底部切到「精修」或「胶片感」，拖动滑杆实时生效，" +
                        "双击任一滑杆可复位；胶片感各组有自己的开关，默认不启用。",
            )
            Step("6", "满意后点「保存」；多张照片用「保存 ▾ → 批量导出全部」。")
            Step(
                "7",
                "存成预设：点来源行（参考图 ▾）→「将当前参考图存入预设」，下次一键复用。" +
                        "预设只保存追色本身——精修与胶片感是当前照片的调整，不进预设；" +
                        "批量导出会带上它们。",
            )
        }

        Section("小技巧") {
            Bullet("参考图和照片的光照越接近，追色效果越好——这比调任何参数都管用。")
            Bullet("用「分区保护」时参考图里最好有人物；没有也能用，会自动回退到全局映射。")
            Bullet("打开「显示保护区域」可以把实际生效的范围叠成绿色，用来确认保护是否到位。")
            Bullet("照片多的时候先批量导出，切到别的应用也会继续跑，通知栏能看进度。")
            Bullet("胶片感太重时先降「柔光强度」——它对画面亮度的影响最明显；" +
                    "颗粒的「数量」和「粗糙程度」分开调：先定质感再定浓淡。")
        }

        Section("关于与诊断") {
            Entry("版本", DebugLog.startBanner.ifEmpty { "（启动信息尚未写入）" })
            Entry(
                "引擎 colortrace",
                "核心算法全部自行实现：Reinhard 统计迁移 / 高斯最优传输 / " +
                        "自训零样本编码器 / 语义分区迁移。",
            )
            Entry(
                "随包模型",
                "YuNet 人脸定位（230KB）· selfie_multiclass 语义分区（16.5MB）· " +
                        "自训零样本编码器（6.6MB）。",
            )
            Entry("许可", "Apache-2.0（见仓库根 LICENSE）")
            Entry("反馈与建议", "2398115885@qq.com")
            Spacer(Modifier.height(4.dp))
            TextButton(onClick = onOpenLog) { Text("查看运行日志") }
        }

        Spacer(Modifier.height(8.dp))
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
            TextButton(onClick = onDismiss) { Text("知道了") }
        }
        Spacer(Modifier.height(16.dp))
    }
}

/**
 * 运行日志视图：只读展示 [DebugLog] 的尾部，一键复制。
 *
 * 为什么放进帮助里：这是**没有电脑时唯一能把现场带出来**的入口——日志含阶段耗时
 * （migrate / render / 保存）与带堆栈的报错，用户复制一段就能定位问题。
 */
@Composable
private fun LogView(onBack: () -> Unit) {
    val log = remember { DebugLog.readTail() }
    val clipboard = LocalClipboardManager.current
    var copied by remember { mutableStateOf(false) }
    Column(
        Modifier
            .fillMaxWidth()
            .navigationBarsPadding()
            .padding(horizontal = 20.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("运行日志", style = MaterialTheme.typography.titleLarge)
            Spacer(Modifier.weight(1f))
            TextButton(onClick = {
                clipboard.setText(AnnotatedString(log))
                copied = true
            }) { Text(if (copied) "已复制" else "复制") }
        }
        Text(
            "最近的运行记录（阶段耗时与报错堆栈）。反馈问题时把这里复制过去，比描述现象有用得多。",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Spacer(Modifier.height(10.dp))
        // 初始滚到**最新**几行：排查问题时关心的是刚发生的事，而不是 128KB 的开头。
        val scroll = rememberScrollState()
        LaunchedEffect(scroll.maxValue) {
            if (scroll.maxValue > 0) scroll.scrollTo(scroll.maxValue)
        }
        Text(
            log,
            style = MaterialTheme.typography.bodySmall.copy(fontFamily = FontFamily.Monospace),
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier
                .fillMaxWidth()
                .heightIn(max = 420.dp)
                .verticalScroll(scroll),
        )
        Spacer(Modifier.height(8.dp))
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
            TextButton(onClick = onBack) { Text("返回帮助") }
        }
        Spacer(Modifier.height(16.dp))
    }
}

/** 一个分区：标题 + 下面的条目。 */
@Composable
private fun Section(title: String, content: @Composable () -> Unit) {
    Spacer(Modifier.height(20.dp))
    Text(
        title,
        style = MaterialTheme.typography.titleSmall,
        color = MaterialTheme.colorScheme.primary,
    )
    Spacer(Modifier.height(6.dp))
    content()
}

/** 一条"名称 + 说明"。 */
@Composable
private fun Entry(name: String, desc: String) {
    Column(Modifier.fillMaxWidth().padding(vertical = 5.dp)) {
        Text(name, style = MaterialTheme.typography.bodyMedium)
        Text(
            desc,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(top = 2.dp),
        )
    }
}

/** 编号步骤。 */
@Composable
private fun Step(index: String, text: String) {
    Row(Modifier.fillMaxWidth().padding(vertical = 4.dp)) {
        Text(
            index,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.primary,
            modifier = Modifier.width(22.dp),
        )
        Text(
            text,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

/** 小技巧条目。 */
@Composable
private fun Bullet(text: String) {
    Row(
        Modifier.fillMaxWidth().padding(vertical = 4.dp),
        verticalAlignment = Alignment.Top,
    ) {
        Text(
            "·",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.primary,
            modifier = Modifier.width(14.dp),
        )
        Text(
            text,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}
