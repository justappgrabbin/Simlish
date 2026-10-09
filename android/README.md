# YOU-N-I-VERSE Android debug shell

Java WebView shell; the studio is packaged offline from ../you-n-i-verse. Android 8+ (API 26), Java 17, Gradle 8.9, Android SDK 35. Build with `gradle -p android assembleDebug` from repository root. GitHub Actions builds automatically on main changes and publishes a direct APK under Releases; an Actions ZIP artifact is also retained.

Local assets use WebViewAssetLoader's HTTPS origin for JavaScript modules and IndexedDB. There is no remote app server or INTERNET permission. The file picker supports images, project JSON and an external camera app. Native exports use Android's Create Document picker and report actual completion/cancellation. No broad storage permission is requested. Portraits remain in private WebView/project storage unless exported by the user.

The debug WebView is inspectable with Chrome's chrome://inspect over USB debugging. Android activity recreation and switching away from the app may interrupt pending exports/camera actions; reopen/export if interrupted. Camera, local storage, download and restore still require testing on a physical phone. GitHub cache retains the debug signing key for updates; if evicted, a new signature may require uninstalling the previous debug build, so export projects before uninstalling. This debug wrapper does not add automatic anatomy or wardrobe generation.
