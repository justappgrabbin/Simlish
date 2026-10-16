#!/usr/bin/env bash
# Runs inside the emulator action: install the APK, open the Synthia page with ?selftest=1, wait for the page's own console line.
set -u
adb install -r Synthia-debug.apk
adb logcat -c
adb shell am start -n app.youniverse.studio/.MainActivity --es start "synthia-app.html?selftest=1"
for i in $(seq 1 100); do
  if adb logcat -d | grep -q 'SYNTHIA_SELFTEST'; then break; fi
  sleep 5
done
adb logcat -d | grep 'SYNTHIA_SELFTEST' | tee emulator-selftest.txt || true
adb logcat -d -s AndroidRuntime:E | tail -20
grep -q 'SYNTHIA_SELFTEST PASS' emulator-selftest.txt
