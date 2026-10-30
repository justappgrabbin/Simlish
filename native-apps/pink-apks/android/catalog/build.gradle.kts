plugins { id("com.android.application"); id("org.jetbrains.kotlin.android") }
android { namespace = "org.youniverse.catalog"; compileSdk = 35
  defaultConfig { applicationId = "org.youniverse.catalog"; minSdk = 24; targetSdk = 35; versionCode = 1; versionName = "0.1.0" }
  compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
  kotlinOptions { jvmTarget = "17" } }
dependencies { implementation(project(":engine")) }
