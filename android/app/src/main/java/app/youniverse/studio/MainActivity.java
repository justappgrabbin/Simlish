package app.youniverse.studio;

import android.app.Activity;
import android.os.Bundle;
import android.content.Intent;
import android.content.ClipData;
import android.net.Uri;
import android.provider.MediaStore;
import android.webkit.*;
import android.widget.Toast;
import androidx.core.content.FileProvider;
import androidx.webkit.WebViewAssetLoader;
import java.io.File;
import java.io.OutputStream;

public class MainActivity extends Activity {
 private WebView web;
 private ValueCallback<Uri[]> chooser;
 private Uri cameraUri;
 private byte[] pendingExport;
 private static final String ORIGIN="https://appassets.androidplatform.net";
 private static final int PICK=10, SAVE=11, NATIVE_PHOTO=12;
 private String photoTarget="photo";
 @Override public void onCreate(Bundle state){
  super.onCreate(state);
  web=new WebView(this);setContentView(web);
  WebView.setWebContentsDebuggingEnabled(BuildConfig.DEBUG);
  WebSettings s=web.getSettings();s.setJavaScriptEnabled(true);s.setDomStorageEnabled(true);s.setAllowFileAccess(false);s.setAllowContentAccess(true);s.setMixedContentMode(WebSettings.MIXED_CONTENT_NEVER_ALLOW);s.setMediaPlaybackRequiresUserGesture(true);
  WebViewAssetLoader loader=new WebViewAssetLoader.Builder().addPathHandler("/assets/",new WebViewAssetLoader.AssetsPathHandler(this)).build();
  web.setWebViewClient(new WebViewClient(){
   @Override public WebResourceResponse shouldInterceptRequest(WebView view,WebResourceRequest request){WebResourceResponse response=loader.shouldInterceptRequest(request.getUrl());if(response!=null){String path=request.getUrl().getPath();if(path!=null&&(path.endsWith(".mjs")||path.endsWith(".js")))response.setMimeType("application/javascript");else if(path!=null&&path.endsWith(".wasm"))response.setMimeType("application/wasm");}return response;}
   @Override public boolean shouldOverrideUrlLoading(WebView view,WebResourceRequest request){Uri u=request.getUrl();if(ORIGIN.equals(u.getScheme()+"://"+u.getHost())&&u.getPath()!=null&&u.getPath().startsWith("/assets/"))return false;if(request.isForMainFrame()&&(u.getScheme().equals("https")||u.getScheme().equals("http"))){try{startActivity(new Intent(Intent.ACTION_VIEW,u));}catch(Exception e){toast("No browser available");}}return true;}
  });
  web.setWebChromeClient(new WebChromeClient(){
   @Override public boolean onShowFileChooser(WebView view,ValueCallback<Uri[]> callback,FileChooserParams params){
    if(chooser!=null)chooser.onReceiveValue(null);chooser=callback;cameraUri=null;
    Intent pick=new Intent(Intent.ACTION_GET_CONTENT);pick.addCategory(Intent.CATEGORY_OPENABLE);String[] accepts=params.getAcceptTypes();boolean image=false;for(String t:accepts)if(t.startsWith("image/"))image=true;boolean video=false,model=false;for(String t:accepts){if(t.startsWith("video/"))video=true;if(t.equals(".glb")||t.startsWith("model/"))model=true;}pick.setType(image?"image/*":video?"video/*":model?"*/*":"application/json");pick.putExtra(Intent.EXTRA_ALLOW_MULTIPLE,params.getMode()==FileChooserParams.MODE_OPEN_MULTIPLE);
    Intent select=Intent.createChooser(pick,image?"Choose from Photos":video?"Choose animation video":model?"Open 3D avatar":"Open studio project");
    if(image && params.isCaptureEnabled()){Intent camera=new Intent(MediaStore.ACTION_IMAGE_CAPTURE);if(camera.resolveActivity(getPackageManager())!=null){try{File dir=new File(getCacheDir(),"camera");dir.mkdirs();File f=File.createTempFile("portrait-",".jpg",dir);cameraUri=FileProvider.getUriForFile(MainActivity.this,getPackageName()+".files",f);camera.putExtra(MediaStore.EXTRA_OUTPUT,cameraUri);camera.addFlags(Intent.FLAG_GRANT_WRITE_URI_PERMISSION|Intent.FLAG_GRANT_READ_URI_PERMISSION);camera.setClipData(ClipData.newRawUri("portrait",cameraUri));select.putExtra(Intent.EXTRA_INITIAL_INTENTS,new Intent[]{camera});}catch(Exception e){cameraUri=null;}}}
    try{startActivityForResult(select,PICK);}catch(Exception e){chooser.onReceiveValue(null);chooser=null;toast("File picker unavailable");}return true;
   }
  });
  web.addJavascriptInterface(new NativeExport(),"NativeDownloads");
  web.loadUrl(ORIGIN+"/assets/index.html");
 }
 public class NativeExport {
  @JavascriptInterface public void analyzeFace(String encoded,int requestId){
   runOnUiThread(()->{
    if(web.getUrl()==null||!web.getUrl().startsWith(ORIGIN+"/assets/"))return;
    new Thread(()->{org.json.JSONObject output=new org.json.JSONObject();android.graphics.Bitmap bitmap=null,working=null;
     try{
      output.put("requestId",requestId);
      if(encoded.length()>30_000_000)throw new Exception("Photo is too large for local analysis");
      byte[] bytes=android.util.Base64.decode(encoded,android.util.Base64.DEFAULT);
      android.graphics.BitmapFactory.Options options=new android.graphics.BitmapFactory.Options();options.inJustDecodeBounds=true;android.graphics.BitmapFactory.decodeByteArray(bytes,0,bytes.length,options);int originalWidth=options.outWidth,originalHeight=options.outHeight;if(originalWidth<=0||originalHeight<=0)throw new Exception("Cannot read photo");
      options.inJustDecodeBounds=false;options.inSampleSize=1;while(Math.max(originalWidth,originalHeight)/options.inSampleSize>800)options.inSampleSize*=2;
      bitmap=android.graphics.BitmapFactory.decodeByteArray(bytes,0,bytes.length,options);if(bitmap==null)throw new Exception("Cannot decode photo");
      int width=bitmap.getWidth()-(bitmap.getWidth()%2);if(width<2)throw new Exception("Photo is too small");
      working=android.graphics.Bitmap.createBitmap(width,bitmap.getHeight(),android.graphics.Bitmap.Config.RGB_565);new android.graphics.Canvas(working).drawBitmap(bitmap,0,0,null);
      android.media.FaceDetector.Face[] faces=new android.media.FaceDetector.Face[3];int count=new android.media.FaceDetector(width,working.getHeight(),3).findFaces(working,faces);
      if(count!=1){output.put("status",count>1?"multiple":"none");}
      else{
       android.graphics.PointF middle=new android.graphics.PointF();faces[0].getMidPoint(middle);double scale=(double)originalWidth/bitmap.getWidth(),eyes=faces[0].eyesDistance()*scale;
       output.put("status","detected");output.put("confidence",faces[0].confidence());output.put("rotation",-faces[0].pose(android.media.FaceDetector.Face.EULER_Z));
       org.json.JSONObject bounds=new org.json.JSONObject();bounds.put("x",middle.x*scale-eyes*1.1);bounds.put("y",middle.y*scale-eyes*.8);bounds.put("width",eyes*2.2);bounds.put("height",eyes*2.8);output.put("bounds",bounds);
      }
     }catch(Exception e){try{output.put("status","failed");output.put("message",e.getMessage());}catch(Exception ignored){}}
     finally{if(working!=null)working.recycle();if(bitmap!=null)bitmap.recycle();}
     String message=output.toString();runOnUiThread(()->web.evaluateJavascript("window.dispatchEvent(new CustomEvent('native-face-analysis',{detail:"+message+"}));",null));
    }).start();
   });
  }
  @JavascriptInterface public void chooseImage(String target){runOnUiThread(()->{
   if(web.getUrl()==null||!web.getUrl().startsWith(ORIGIN+"/assets/"))return;
   if(!target.equals("photo")&&!target.equals("body")&&!target.equals("sheet"))return;
   photoTarget=target;
   Intent pick;
   if(android.os.Build.VERSION.SDK_INT>=33){pick=new Intent(MediaStore.ACTION_PICK_IMAGES);pick.setType("image/*");}
   else {pick=new Intent(Intent.ACTION_GET_CONTENT);pick.addCategory(Intent.CATEGORY_OPENABLE);pick.setType("image/*");}
   try{startActivityForResult(pick,NATIVE_PHOTO);}catch(Exception e){try{Intent fallback=new Intent(Intent.ACTION_GET_CONTENT);fallback.setType("image/*");fallback.addCategory(Intent.CATEGORY_OPENABLE);startActivityForResult(Intent.createChooser(fallback,"Choose a photo"),NATIVE_PHOTO);}catch(Exception failure){toast("Photo picker unavailable: "+failure.getMessage());}}
  });}

  @JavascriptInterface public void save(String name,String mime,String encoded){
   runOnUiThread(()->{
    if(web.getUrl()==null||!web.getUrl().startsWith(ORIGIN+"/assets/"))return;
    if(pendingExport!=null){notifySave("Finish or cancel the current save first.");return;}
    try{
     if(encoded.length()>90_000_000)throw new Exception("Export is too large for this debug build");
     if(!mime.equals("image/png")&&!mime.equals("image/jpeg")&&!mime.equals("application/json")&&!mime.equals("model/gltf-binary")&&!mime.equals("video/mp4")&&!mime.equals("application/octet-stream")&&!mime.equals("text/plain"))throw new Exception("Unsupported export type");
     pendingExport=android.util.Base64.decode(encoded,android.util.Base64.DEFAULT);
     Intent out=new Intent(Intent.ACTION_CREATE_DOCUMENT);out.addCategory(Intent.CATEGORY_OPENABLE);out.setType(mime);out.putExtra(Intent.EXTRA_TITLE,name.replaceAll("[^a-zA-Z0-9._-]","_"));startActivityForResult(out,SAVE);
    }catch(Exception e){pendingExport=null;notifySave("Could not open save: "+e.getMessage());}
   });
  }
 }
 @Override protected void onActivityResult(int request,int result,Intent data){
  super.onActivityResult(request,result,data);
  if(request==NATIVE_PHOTO){
   if(result!=RESULT_OK||data==null||data.getData()==null){toast("No photo selected");return;}
   Uri selected=data.getData();String target=photoTarget;
   new Thread(()->{try{
    android.graphics.Bitmap bitmap;
    if(android.os.Build.VERSION.SDK_INT>=28){
     android.graphics.ImageDecoder.Source input=android.graphics.ImageDecoder.createSource(getContentResolver(),selected);
     bitmap=android.graphics.ImageDecoder.decodeBitmap(input,(decoder,info,src)->{int largest=Math.max(info.getSize().getWidth(),info.getSize().getHeight());if(largest>2048)decoder.setTargetSampleSize((largest+2047)/2048);decoder.setAllocator(android.graphics.ImageDecoder.ALLOCATOR_SOFTWARE);});
    }else{
     android.graphics.BitmapFactory.Options opts=new android.graphics.BitmapFactory.Options();opts.inJustDecodeBounds=true;
     try(java.io.InputStream in=getContentResolver().openInputStream(selected)){android.graphics.BitmapFactory.decodeStream(in,null,opts);}
     int sample=1;while(Math.max(opts.outWidth,opts.outHeight)/sample>2048)sample*=2;opts.inJustDecodeBounds=false;opts.inSampleSize=sample;
     try(java.io.InputStream in=getContentResolver().openInputStream(selected)){bitmap=android.graphics.BitmapFactory.decodeStream(in,null,opts);}
    }
    if(bitmap==null)throw new Exception("Image could not be decoded");
    java.io.ByteArrayOutputStream buffer=new java.io.ByteArrayOutputStream();bitmap.compress(android.graphics.Bitmap.CompressFormat.PNG,100,buffer);bitmap.recycle();
    String image="data:image/png;base64,"+android.util.Base64.encodeToString(buffer.toByteArray(),android.util.Base64.NO_WRAP);
    String detail="{target:"+org.json.JSONObject.quote(target)+",image:"+org.json.JSONObject.quote(image)+"}";
    runOnUiThread(()->web.evaluateJavascript("window.dispatchEvent(new CustomEvent('native-image-selected',{detail:"+detail+"}));",null));
   }catch(Exception e){runOnUiThread(()->toast("Could not open photo: "+e.getMessage()));}}).start();
  }

  if(request==PICK&&chooser!=null){Uri[] uris=null;if(result==RESULT_OK){uris=WebChromeClient.FileChooserParams.parseResult(result,data);if((uris==null||uris.length==0)&&cameraUri!=null)uris=new Uri[]{cameraUri};}chooser.onReceiveValue(uris);chooser=null;cameraUri=null;}
  if(request==SAVE){final byte[] bytes=pendingExport;pendingExport=null;if(result!=RESULT_OK||data==null||data.getData()==null){notifySave("Save cancelled. Your project is still in the studio.");return;}Uri uri=data.getData();new Thread(()->{try(OutputStream stream=getContentResolver().openOutputStream(uri)){if(stream==null)throw new Exception("Destination unavailable");stream.write(bytes);stream.flush();notifySave("File saved successfully.");}catch(Exception e){notifySave("Save failed: "+e.getMessage());}}).start();}
 }
 private void notifySave(String message){runOnUiThread(()->{toast(message);web.evaluateJavascript("window.dispatchEvent(new CustomEvent('native-save-result',{detail:"+org.json.JSONObject.quote(message)+"}));",null);});}
 private void toast(String text){Toast.makeText(this,text,Toast.LENGTH_LONG).show();}
 @Override protected void onPause(){super.onPause();web.onPause();}
 @Override protected void onResume(){super.onResume();if(web!=null)web.onResume();}
 @Override public void onBackPressed(){if(web.canGoBack())web.goBack();else super.onBackPressed();}
 @Override protected void onDestroy(){if(chooser!=null)chooser.onReceiveValue(null);web.removeJavascriptInterface("NativeDownloads");web.destroy();super.onDestroy();}
}
