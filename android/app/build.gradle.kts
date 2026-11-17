plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "dev.multicamdepth"
    compileSdk = 35

    defaultConfig {
        applicationId = "dev.multicamdepth"
        minSdk = 28 // first release with the logical multi-camera API
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
        ndk {
            // OpenCV ships native code for every ABI; real phones only need these two
            abiFilters += listOf("arm64-v8a", "armeabi-v7a")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("org.opencv:opencv:4.10.0")
    testImplementation("junit:junit:4.13.2")
}
