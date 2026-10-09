plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "com.colortrace.poc"
    compileSdk = 34
    // r27+ 默认满足 16KB page size（Android 15+ 硬要求）；版本写死避免各机器 NDK 漂移
    ndkVersion = "27.3.13750724"

    defaultConfig {
        applicationId = "com.colortrace.poc"
        minSdk = 26
        targetSdk = 34
        versionCode = 2
        versionName = "0.2-ui"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        // native 热循环（src/main/cpp）：LAB 转换的逐位等价移植，见 engine/NativeLab.kt
        externalNativeBuild {
            cmake {
                // 复用 OpenCV AAR 已带的 libc++_shared.so，避免第二份 STL
                arguments += "-DANDROID_STL=c++_shared"
            }
        }
        ndk {
            // 只用到的两个：真机 arm64-v8a、模拟器 x86_64（32 位谁都不用，省一半构建与体积）
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        compose = true
    }

    sourceSets {
        // ONNX 模型（仓库根 .models/，本地不入库）打包进 APK——
        // P0/P1 对拍测试与 P2 主应用共用同一份资产；
        // 金标 JSON 在 src/androidTest/assets/goldens/
        getByName("androidTest") {
            assets.srcDir("../.models")
        }
        getByName("main") {
            assets.srcDir("../.models")
        }
    }
}

dependencies {
    // 与桌面同主版本（桌面 opencv-python 5.0.0）——对拍差异最小的选择
    implementation("org.opencv:opencv:5.0.0")
    // P2 UI：Compose + Material 3（BOM 统一版本）
    implementation(platform("androidx.compose:compose-bom:2024.09.03"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-core")
    implementation("androidx.activity:activity-compose:1.9.2")
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.6")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test:runner:1.6.1")
}
