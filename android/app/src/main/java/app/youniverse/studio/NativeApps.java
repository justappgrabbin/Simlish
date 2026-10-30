package app.youniverse.studio;

import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.content.pm.ResolveInfo;
import android.net.Uri;
import android.os.Build;
import android.provider.Settings;
import android.webkit.JavascriptInterface;
import android.webkit.WebView;
import androidx.core.content.FileProvider;
import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import org.json.JSONArray;
import org.json.JSONObject;

/** window.NativeApps — lets the Apps tray run real installed Android apps beside web apps.
 *  The JS tray runs the address gate (state-space resolver) first and passes the admitted address;
 *  every call here also refuses without one and writes to a native ledger (files/native-ledger.jsonl). */
public class NativeApps {
 public static final int PICK_PIECE=21;
 private final Activity act; private final WebView web; private final String origin;
 private String pendingAddress;
 public NativeApps(Activity act,WebView web,String origin){this.act=act;this.web=web;this.origin=origin;}

 private boolean trusted(){String u=web.getUrl();return u!=null&&u.startsWith(origin+"/assets/");}
 private void emit(String event,JSONObject detail){String d=detail.toString();act.runOnUiThread(()->web.evaluateJavascript("window.dispatchEvent(new CustomEvent('"+event+"',{detail:"+d+"}));",null));}
 private void log(String action,String target,String address,String result){
  try(FileWriter w=new FileWriter(new File(act.getFilesDir(),"native-ledger.jsonl"),true)){JSONObject o=new JSONObject();o.put("at",System.currentTimeMillis());o.put("action",action);o.put("target",target);o.put("address",address);o.put("result",result);w.write(o+"\n");}catch(Exception ignored){}
 }
 private JSONObject r(boolean ok,String msg){JSONObject o=new JSONObject();try{o.put("ok",ok);o.put("message",msg);}catch(Exception ignored){}return o;}

 @JavascriptInterface public String listInstalled(){
  JSONArray out=new JSONArray();
  try{PackageManager pm=act.getPackageManager();Intent i=new Intent(Intent.ACTION_MAIN);i.addCategory(Intent.CATEGORY_LAUNCHER);
   List<ResolveInfo> l=pm.queryIntentActivities(i,0);
   for(ResolveInfo ri:l){JSONObject o=new JSONObject();o.put("package",ri.activityInfo.packageName);o.put("label",String.valueOf(ri.loadLabel(pm)));out.put(o);}
  }catch(Exception ignored){}
  return out.toString();
 }
 @JavascriptInterface public boolean isInstalled(String pkg){
  if(!NativeAppRules.isPackageName(pkg))return false;
  try{act.getPackageManager().getPackageInfo(pkg,0);return true;}catch(Exception e){return false;}
 }
 /** Launch by package name through its launch intent. Keyboards with no launcher fall back to keyboard settings. */
 @JavascriptInterface public String launch(String pkg,String address){
  if(!trusted())return r(false,"untrusted page").toString();
  if(!NativeAppRules.hasAddress(address)){log("launch",pkg,address,"refused: no address");return r(false,"address gate required").toString();}
  if(!NativeAppRules.isPackageName(pkg))return r(false,"bad package name").toString();
  Intent i=act.getPackageManager().getLaunchIntentForPackage(pkg);
  if(i==null&&isInstalled(pkg)){i=new Intent(Settings.ACTION_INPUT_METHOD_SETTINGS);}
  if(i==null){log("launch",pkg,address,"not installed");return r(false,"not installed").toString();}
  i.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);final Intent go=i;
  act.runOnUiThread(()->{try{act.startActivity(go);}catch(Exception e){emit("native-app-result",r(false,e.getMessage()));}});
  log("launch",pkg,address,"ok");return r(true,"launched").toString();
 }
 @JavascriptInterface public void openKeyboardSettings(){act.runOnUiThread(()->{try{act.startActivity(new Intent(Settings.ACTION_INPUT_METHOD_SETTINGS));}catch(Exception ignored){}});}

 /** Download an APK over https (optional: needs network only for this step), keep it on the device, then install. */
 @JavascriptInterface public void installFromUrl(String url,String id,String address){
  if(!trusted())return;
  if(!NativeAppRules.hasAddress(address)||!NativeAppRules.isApkSource(url)){log("download",url,address,"refused");emit("native-app-result",r(false,"refused: needs https source and an address"));return;}
  new Thread(()->{
   try{File dir=new File(act.getFilesDir(),"apks");dir.mkdirs();File f=new File(dir,NativeAppRules.safeId(id)+".apk");
    HttpURLConnection c=(HttpURLConnection)new URL(url).openConnection();c.setInstanceFollowRedirects(true);c.setConnectTimeout(15000);c.setReadTimeout(60000);
    try(InputStream in=c.getInputStream();OutputStream out=new FileOutputStream(f)){byte[] b=new byte[65536];int n;long t=0;while((n=in.read(b))>0){out.write(b,0,n);t+=n;}}
    log("download",url,address,"ok "+f.length());installFile(f,address);
   }catch(Exception e){log("download",url,address,"failed "+e.getMessage());emit("native-app-result",r(false,"Download failed (offline?). Pick the APK file instead: "+e.getMessage()));}
  }).start();
 }
 /** Install an APK already stored on the device (from installFromUrl or a picked file). */
 @JavascriptInterface public void installStored(String id,String address){
  if(!trusted()||!NativeAppRules.hasAddress(address))return;
  File f=new File(new File(act.getFilesDir(),"apks"),NativeAppRules.safeId(id)+".apk");
  if(!f.exists()){emit("native-app-result",r(false,"no stored APK for "+id));return;}
  installFile(f,address);
 }
 private void installFile(File f,String address){
  act.runOnUiThread(()->{
   try{
    if(Build.VERSION.SDK_INT>=26&&!act.getPackageManager().canRequestPackageInstalls()){
     act.startActivity(new Intent(Settings.ACTION_MANAGE_UNKNOWN_APP_SOURCES,Uri.parse("package:"+act.getPackageName())));
     emit("native-app-result",r(false,"Allow 'Install unknown apps' for this Computer, then tap Install again."));return;}
    Uri u=FileProvider.getUriForFile(act,act.getPackageName()+".files",f);
    Intent i=new Intent(Intent.ACTION_VIEW);i.setDataAndType(u,"application/vnd.android.package-archive");i.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION|Intent.FLAG_ACTIVITY_NEW_TASK);
    act.startActivity(i);log("install",f.getName(),address,"installer opened");
   }catch(Exception e){emit("native-app-result",r(false,"Install failed: "+e.getMessage()));}
  });
 }
 /** Let the user pick any local .apk or .zip (web app folder). Result arrives as 'native-piece-picked'. */
 @JavascriptInterface public void pickPiece(String address){
  if(!NativeAppRules.hasAddress(address))return;pendingAddress=address;
  act.runOnUiThread(()->{Intent i=new Intent(Intent.ACTION_OPEN_DOCUMENT);i.addCategory(Intent.CATEGORY_OPENABLE);i.setType("*/*");i.putExtra(Intent.EXTRA_MIME_TYPES,new String[]{"application/vnd.android.package-archive","application/zip","application/octet-stream"});
   try{act.startActivityForResult(i,PICK_PIECE);}catch(Exception e){emit("native-app-result",r(false,"File picker unavailable"));}});
 }
 public void onPicked(Uri uri){
  final String address=pendingAddress;pendingAddress=null;if(uri==null||address==null)return;
  new Thread(()->{JSONObject o=new JSONObject();
   try{String name=displayName(uri);String kind=NativeAppRules.kindOf(name);if(kind==null)throw new Exception("Pick an .apk or a .zip of a web app folder");
    String id=NativeAppRules.safeId(name);o.put("id",id);o.put("name",name);o.put("kind",kind);o.put("address",address);
    if(kind.equals("apk")){File dir=new File(act.getFilesDir(),"apks");dir.mkdirs();File f=new File(dir,id+".apk");copy(uri,f);
     PackageInfo pi=act.getPackageManager().getPackageArchiveInfo(f.getPath(),0);if(pi==null)throw new Exception("Not a valid APK");
     o.put("package",pi.packageName);o.put("installed",isInstalled(pi.packageName));log("pick-apk",pi.packageName,address,"stored");}
    else{File dir=new File(new File(new File(act.getFilesDir(),"local-root"),"webapps"),id);unzip(uri,dir);
     String entry=new File(dir,"index.html").exists()?"index.html":firstHtml(dir,dir);if(entry==null)throw new Exception("No .html page inside the zip");
     o.put("folder",origin+"/local/webapps/"+id);o.put("entry",entry);log("pick-zip",id,address,"unpacked");}
    o.put("ok",true);
   }catch(Exception e){try{o.put("ok",false);o.put("message",e.getMessage());}catch(Exception ignored){}}
   emit("native-piece-picked",o);
  }).start();
 }
 private String displayName(Uri uri){try(android.database.Cursor c=act.getContentResolver().query(uri,null,null,null,null)){if(c!=null&&c.moveToFirst()){int i=c.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME);if(i>=0)return c.getString(i);}}catch(Exception ignored){}return uri.getLastPathSegment();}
 private void copy(Uri uri,File f)throws IOException{try(InputStream in=act.getContentResolver().openInputStream(uri);OutputStream out=new FileOutputStream(f)){byte[] b=new byte[65536];int n;while((n=in.read(b))>0)out.write(b,0,n);}}
 private void unzip(Uri uri,File dir)throws Exception{dir.mkdirs();try(ZipInputStream z=new ZipInputStream(act.getContentResolver().openInputStream(uri))){ZipEntry e;while((e=z.getNextEntry())!=null){if(!NativeAppRules.isSafeZipEntry(e.getName()))continue;File out=new File(dir,e.getName());if(!out.getCanonicalPath().startsWith(dir.getCanonicalPath()))continue;if(e.isDirectory()){out.mkdirs();continue;}out.getParentFile().mkdirs();try(OutputStream o=new FileOutputStream(out)){byte[] b=new byte[65536];int n;while((n=z.read(b))>0)o.write(b,0,n);}}}}
 private String firstHtml(File d,File base){File[] fs=d.listFiles();if(fs==null)return null;for(File f:fs)if(f.isFile()&&f.getName().endsWith(".html"))return base.toURI().relativize(f.toURI()).getPath();for(File f:fs)if(f.isDirectory()){String s=firstHtml(f,base);if(s!=null)return s;}return null;}
}
