plugins { id("com.android.library"); id("org.jetbrains.kotlin.android") }
android { namespace = "org.youniverse.engine"; compileSdk = 35
  defaultConfig { minSdk = 24; testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner" }
  compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
  kotlinOptions { jvmTarget = "17" } }
dependencies { api("wang.harlon.quickjs:wrapper-android:3.2.3")
  androidTestImplementation("androidx.test:runner:1.6.2"); androidTestImplementation("androidx.test.ext:junit:1.2.1"); androidTestImplementation("junit:junit:4.13.2") }
