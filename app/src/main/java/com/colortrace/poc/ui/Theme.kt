package com.colortrace.poc.ui

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/**
 * 品牌化 M3 主题（android/nowinandroid + Jetsnack 的做法：不改组件语义，
 * 只换 color/shape token）。种子色 = 桌面端同一 amber（#E8A854）——两端品牌一致。
 *
 * 深色为默认：调色工作区纪律（中性深底不干扰色彩判断，与桌面 GUI 同一条）。
 * 动效令牌：Emphasized ≈ M3 emphasized cubic；弹簧用于跟手控件（滑块/手柄）。
 */
object Motion {
    val Emphasized = androidx.compose.animation.core.CubicBezierEasing(0.2f, 0f, 0f, 1f)
    const val PAGE_MS = 450      // M3 long duration：屏幕级过渡
    const val CONTENT_MS = 250   // M3 medium：卡片内容
    const val FAST_MS = 120      // M3 short：小控件
}

// ---- amber ramp（与桌面 gui/theme.py 同源） ----
private val Amber = Color(0xFFE8A854)
private val AmberDim = Color(0xFFC98F3F)
private val AmberContainerDark = Color(0xFF3F2D14)
private val OnAmber = Color(0xFF241708)
private val AmberContainerLight = Color(0xFFFFDDB4)

private val DarkColors = darkColorScheme(
    primary = Amber,
    onPrimary = OnAmber,
    primaryContainer = AmberContainerDark,
    onPrimaryContainer = Color(0xFFFFDDB4),
    secondary = Color(0xFFD3C4A8),
    onSecondary = Color(0xFF382F1D),
    secondaryContainer = Color(0xFF4A4231),
    onSecondaryContainer = Color(0xFFEFE0C4),
    tertiary = Color(0xFFA8CFA0),
    error = Color(0xFFFFB4AB),
    background = Color(0xFF161719),
    onBackground = Color(0xFFE7E9EC),
    surface = Color(0xFF161719),
    onSurface = Color(0xFFE7E9EC),
    surfaceVariant = Color(0xFF26292D),
    onSurfaceVariant = Color(0xFF9BA2AB),
    outline = Color(0xFF3D434A),
    outlineVariant = Color(0xFF2C2F34),
    surfaceContainer = Color(0xFF1E2023),
    surfaceContainerHigh = Color(0xFF26292D),
    surfaceContainerHighest = Color(0xFF31353B),
)

private val LightColors = lightColorScheme(
    primary = Color(0xFF8A5A17),
    onPrimary = Color(0xFFFFFFFF),
    primaryContainer = AmberContainerLight,
    onPrimaryContainer = Color(0xFF2E1D00),
    background = Color(0xFFFCF9F4),
    onBackground = Color(0xFF1D1B16),
    surface = Color(0xFFFCF9F4),
    onSurface = Color(0xFF1D1B16),
    surfaceVariant = Color(0xFFEDE1CE),
    onSurfaceVariant = Color(0xFF4D4639),
)

private val AppShapes = Shapes(
    extraSmall = RoundedCornerShape(6.dp),
    small = RoundedCornerShape(8.dp),
    medium = RoundedCornerShape(12.dp),
    large = RoundedCornerShape(16.dp),
    extraLarge = RoundedCornerShape(20.dp),
)

private val AppTypography = Typography().let {
    it.copy(
        titleLarge = it.titleLarge.copy(fontWeight = FontWeight.SemiBold),
        headlineMedium = it.headlineMedium.copy(fontWeight = FontWeight.Bold,
                                                letterSpacing = 1.sp),
    )
}

@Composable
fun ColortraceTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit,
) {
    MaterialTheme(
        colorScheme = if (darkTheme) DarkColors else LightColors,
        shapes = AppShapes,
        typography = AppTypography,
        content = content,
    )
}
