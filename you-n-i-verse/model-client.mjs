export function mountModelClient({getPhoto,addTarget,open3D}){
 const $=id=>document.getElementById(id);let enabled=[],busy=false,connection=null;
 const toDataURL=blob=>new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result);reader.onerror=reject;reader.readAsDataURL(blob);});
 function base(){const url=new URL($('modelHost').value);if(url.protocol!=='https:'&&!(['localhost','127.0.0.1'].includes(url.hostname)&&url.protocol==='http:'))throw Error('Use HTTPS for a remote model host.');return url.href.replace(/\/$/,'');}
 async function api(path,options={},host=connection){if(!host)throw Error('Connect a model host first.');const response=await fetch(host.url+path,{...options,headers:{Authorization:'Bearer '+host.token,...options.headers}});if(!response.ok){const detail=await response.json().catch(()=>({}));throw Error(detail.detail||'Model host returned '+response.status);}return response;}
 $('connectModels').onclick=async()=>{try{const candidate={url:base(),token:$('modelToken').value};const engines=await(await api('/engines',{},candidate)).json();connection=candidate;enabled=Object.keys(engines).filter(id=>engines[id].ready);$('modelEngine').replaceChildren();for(const [id,state] of Object.entries(engines)){const option=new Option(id+(state.ready?'':' · not installed'),id);option.disabled=!state.ready;$('modelEngine').add(option);}$('modelStatus').textContent=enabled.length+' engines available. Unavailable engines need installation on the model host.';if(enabled.length)$('modelEngine').value=enabled[0];$('runModel').disabled=!enabled.length;}catch(error){$('modelStatus').textContent=error.message;}};
 $('runModel').onclick=async()=>{
 if(busy)return;busy=true;$('runModel').disabled=true;const identity=getPhoto(),host=connection;
 try{
  const engine=$('modelEngine').value;if(!enabled.includes(engine))throw Error('Connect to an installed model engine first.');
  const request={engine,params:{prompt:$('modelPrompt').value,seed:Number($('modelSeed').value)}};
  if(engine!=='text-image'){
   if(!identity)throw Error('Choose your identity photo first.');const canvas=document.createElement('canvas'),scale=Math.min(1,1200/Math.max(identity.naturalWidth||identity.width,identity.naturalHeight||identity.height));canvas.width=Math.round((identity.naturalWidth||identity.width)*scale);canvas.height=Math.round((identity.naturalHeight||identity.height)*scale);canvas.getContext('2d').drawImage(identity,0,0,canvas.width,canvas.height);request.photo=canvas.toDataURL('image/png');
  }
  if(engine.startsWith('liveportrait')){const file=$('modelDriving').files[0];if(!file)throw Error('Choose a driving video for animation.');if(file.size>20e6)throw Error('Use a driving clip smaller than 20 MB.');request.driving=await toDataURL(file);}
  if(engine==='facefusion'){const file=$('modelTarget').files[0];if(!file)throw Error('Choose a target picture for FaceFusion.');request.target=await toDataURL(file);}
  $('modelStatus').textContent='Sending this request to your selected model host…';const {id}=await(await api('/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(request)},host)).json();
  let job;do{await new Promise(resolve=>setTimeout(resolve,1500));job=await(await api('/jobs/'+id,{},host)).json();$('modelStatus').textContent=engine+': '+job.status;}while(['queued','running'].includes(job.status));
  $('modelResults').replaceChildren();
  for(const name of job.files){const button=document.createElement('button');button.textContent='Download '+name;button.onclick=async()=>{try{const blob=await(await api('/jobs/'+id+'/files/'+name.split('/').map(encodeURIComponent).join('/'),{},host)).blob();const reader=new FileReader();if(globalThis.NativeDownloads?.save){reader.onload=()=>NativeDownloads.save(name.split('/').pop(),blob.type,String(reader.result).split(',')[1]);reader.readAsDataURL(blob);}else{const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name.split('/').pop();a.click();setTimeout(()=>URL.revokeObjectURL(url),60000);}}catch(error){$('modelStatus').textContent=error.message;}};$('modelResults').append(button);
   if(job.status==='done'&&/\.(png|jpg|jpeg|glb)$/i.test(name)){
    const preview=document.createElement('button');preview.textContent=/\.glb$/i.test(name)?'Open 3D avatar':'Add picture as a morph form';preview.onclick=async()=>{try{if(engine!=='text-image'&&identity!==getPhoto())throw Error('Identity photo changed; retain output as a download or run again.');const blob=await(await api('/jobs/'+id+'/files/'+name.split('/').map(encodeURIComponent).join('/'),{},host)).blob();if(/\.glb$/i.test(name))await open3D(new Uint8Array(await blob.arrayBuffer()));else await addTarget(engine+' · '+name,blob);}catch(error){$('modelStatus').textContent=error.message;}};$('modelResults').append(preview);
   }
  }
  $('modelStatus').textContent=job.status==='failed'?job.error:'Outputs ready. Add generated pictures as morph forms or open exported avatars.';
 }catch(error){$('modelStatus').textContent=error.message;}finally{busy=false;$('runModel').disabled=!enabled.length;}
 };
}
