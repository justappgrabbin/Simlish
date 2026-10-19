#!/usr/bin/env bash
# Runs inside the emulator action: install the APK, open each bundled page with ?selftest=1, wait for the page's own console line.
set -u
adb install -r Synthia-debug.apk
: > emulator-selftest.txt
fail=0
for page in "guide.html?selftest=1:GUIDE_SELFTEST" "synthia-app.html?selftest=1:SYNTHIA_SELFTEST" "measure.html?selftest=1:MEASURE_SELFTEST" "play.html?selftest=1:PLAY_SELFTEST"; do
  url="${page%%:*}"; marker="${page##*:}"
  adb logcat -c
  adb shell am force-stop app.youniverse.studio
  adb shell am start -n app.youniverse.studio/.MainActivity --es start "$url"
  for i in $(seq 1 100); do
    if adb logcat -d | grep -q "$marker"; then break; fi
    sleep 5
  done
  adb logcat -d | grep "$marker" | tee -a emulator-selftest.txt || true
  grep -q "$marker PASS" emulator-selftest.txt || fail=1
done
adb logcat -d -s AndroidRuntime:E | tail -20
exit $fail
