plugins {
    id("com.android.application") version "8.5.2" apply false
    id("org.jetbrains.kotlin.android") version "2.0.20" apply false
    // Kotlin 2.0 起用 Compose 编译器插件（取代 composeOptions 配置）
    id("org.jetbrains.kotlin.plugin.compose") version "2.0.20" apply false
}
