package app.youniverse.studio;

/** Pure checks for the native app bridge (no Android imports, so they are unit-testable). */
public final class NativeAppRules {
 private NativeAppRules(){}
 /** Valid Android application id, e.g. app.prismkeys */
 public static boolean isPackageName(String p){
  if(p==null||p.length()>255)return false;
  return p.matches("[A-Za-z][A-Za-z0-9_]*(\\.[A-Za-z][A-Za-z0-9_]*)+");
 }
 /** APK download sources allowed: https only (GitHub releases or any https host the user typed). */
 public static boolean isApkSource(String url){
  if(url==null)return false;
  try{java.net.URI u=new java.net.URI(url);return "https".equals(u.getScheme())&&u.getHost()!=null&&!u.getHost().isEmpty();}catch(Exception e){return false;}
 }
 /** "apk", "zip" or null, from a file name the user picked. */
 public static String kindOf(String name){
  if(name==null)return null;String n=name.toLowerCase(java.util.Locale.ROOT);
  if(n.endsWith(".apk"))return "apk";if(n.endsWith(".zip"))return "zip";return null;
 }
 /** Safe folder id for an imported web app. */
 public static String safeId(String name){
  String s=(name==null?"app":name).replaceAll("\\.(zip|apk)$","").replaceAll("[^A-Za-z0-9_-]+","-").replaceAll("^-+|-+$","");
  return s.isEmpty()?"app":s.length()>60?s.substring(0,60):s;
 }
 /** Zip entry must stay inside the target folder (blocks ../ traversal). */
 public static boolean isSafeZipEntry(String entry){
  if(entry==null||entry.isEmpty()||entry.startsWith("/")||entry.contains("\\"))return false;
  for(String part:entry.split("/"))if(part.equals(".."))return false;
  return true;
 }
 /** Every native launch/install must carry an address the JS gate already admitted. */
 public static boolean hasAddress(String a){return a!=null&&a.matches("\\d+(\\.\\d+){4}");}
}
