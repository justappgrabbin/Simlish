package app.youniverse.studio;
import static org.junit.Assert.*;
import org.junit.Test;
public class NativeAppRulesTest {
 @Test public void packages(){
  assertTrue(NativeAppRules.isPackageName("app.prismkeys"));
  assertTrue(NativeAppRules.isPackageName("org.stellarproximology.trident"));
  assertTrue(NativeAppRules.isPackageName("app.synthia.resonance.preview"));
  assertFalse(NativeAppRules.isPackageName("prismkeys"));
  assertFalse(NativeAppRules.isPackageName("app.prism keys"));
  assertFalse(NativeAppRules.isPackageName("app.;rm"));
  assertFalse(NativeAppRules.isPackageName(null));
 }
 @Test public void sources(){
  assertTrue(NativeAppRules.isApkSource("https://github.com/justappgrabbin/SynthAi/releases/download/prism-keys-build-5/Prism-Keys.apk"));
  assertFalse(NativeAppRules.isApkSource("http://example.com/a.apk"));
  assertFalse(NativeAppRules.isApkSource("file:///sdcard/a.apk"));
  assertFalse(NativeAppRules.isApkSource("javascript:alert(1)"));
 }
 @Test public void files(){
  assertEquals("apk",NativeAppRules.kindOf("Prism-Keys.APK"));
  assertEquals("zip",NativeAppRules.kindOf("paper.zip"));
  assertNull(NativeAppRules.kindOf("x.exe"));
  assertEquals("my-app",NativeAppRules.safeId("my app!.zip"));
  assertTrue(NativeAppRules.isSafeZipEntry("index.html"));
  assertTrue(NativeAppRules.isSafeZipEntry("a/b/c.js"));
  assertFalse(NativeAppRules.isSafeZipEntry("../evil"));
  assertFalse(NativeAppRules.isSafeZipEntry("a/../../evil"));
  assertFalse(NativeAppRules.isSafeZipEntry("/etc/x"));
 }
 @Test public void address(){
  assertTrue(NativeAppRules.hasAddress("18.5.3.4.2"));
  assertFalse(NativeAppRules.hasAddress("18.5"));
  assertFalse(NativeAppRules.hasAddress(null));
 }
}
