// Send the ORIGINAL, untouched uploaded file to Supabase Storage. No keys live in the repo: they come from env
// (SUPABASE_URL plus SUPABASE_SERVICE_KEY or SUPABASE_ANON_KEY; bucket SUPABASE_BUCKET). If anything is missing or
// the call fails, the original is queued in an outbox and marked 'pending-supabase'. Nothing here creates a project.
import {sha256} from './sha256.mjs';
export const DEFAULT_BUCKET='simlish-originals'; // INVESTIGATING: bucket name is a placeholder default, not Alexis's choice
export function readSupabaseEnv(env={}){
 const url=env.SUPABASE_URL,key=env.SUPABASE_SERVICE_KEY||env.SUPABASE_ANON_KEY;
 return url&&key?{url:String(url).replace(/\/+$/,''),key,bucket:env.SUPABASE_BUCKET||DEFAULT_BUCKET,usingServiceKey:!!env.SUPABASE_SERVICE_KEY}:null;
}
const b64=u8=>{let s='';for(let i=0;i<u8.length;i+=0x8000)s+=String.fromCharCode(...u8.subarray(i,i+0x8000));return btoa(s)};
const unb64=s=>Uint8Array.from(atob(s),c=>c.charCodeAt(0));
export function memoryOutbox(){const m=new Map();return {put:async i=>{m.set(i.id,i)},list:async()=>[...m.values()],remove:async id=>{m.delete(id)}}}
export function storageOutbox(storage,key='synthia.simlish.supabase.outbox.v1'){ // bytes kept as base64; small files only (browser storage limit)
 const read=()=>{const r=storage.getItem(key);return r?JSON.parse(r):{}};
 return {put:async i=>{const o=read();o[i.id]={...i,bytes:undefined,base64:b64(i.bytes)};storage.setItem(key,JSON.stringify(o))},
  list:async()=>Object.values(read()).map(i=>({...i,bytes:unb64(i.base64),base64:undefined})),
  remove:async id=>{const o=read();delete o[id];storage.setItem(key,JSON.stringify(o))}};
}
export async function fileOutbox(dir){ // Node only (desktop computer); original bytes saved byte-for-byte
 const fs=await import('node:fs/promises'),path=await import('node:path');await fs.mkdir(dir,{recursive:true});
 return {put:async i=>{await fs.writeFile(path.join(dir,i.id+'.bin'),i.bytes);await fs.writeFile(path.join(dir,i.id+'.json'),JSON.stringify({...i,bytes:undefined}))},
  list:async()=>{const out=[];for(const f of await fs.readdir(dir))if(f.endsWith('.json')){const m=JSON.parse(await fs.readFile(path.join(dir,f),'utf8'));out.push({...m,bytes:new Uint8Array(await fs.readFile(path.join(dir,m.id+'.bin')))})}return out},
  remove:async id=>{await fs.rm(path.join(dir,id+'.bin'),{force:true});await fs.rm(path.join(dir,id+'.json'),{force:true})}};
}
const safe=n=>String(n||'upload').replace(/[^\w.\-]+/g,'_').slice(0,120);
export function originalItem({name,bytes,contentType='application/octet-stream',meta={}}){
 const sha=sha256(bytes);return {id:sha.slice(0,24),sha256:sha,name,path:'originals/'+sha+'/'+safe(name),contentType,meta,bytes,size:bytes.length};
}
async function upload(cfg,item,{client,fetchImpl}){
 if(client?.storage?.from){
  const r=await client.storage.from(cfg.bucket).upload(item.path,item.bytes,{contentType:item.contentType,upsert:false});
  if(r.error&&!/exist|duplicate|409/i.test(String(r.error.message||r.error.statusCode||r.error)))throw new Error(r.error.message||String(r.error));
  return {path:item.path,duplicate:!!r.error};
 }
 const res=await fetchImpl(cfg.url+'/storage/v1/object/'+encodeURIComponent(cfg.bucket)+'/'+item.path.split('/').map(encodeURIComponent).join('/'),{method:'POST',headers:{authorization:'Bearer '+cfg.key,apikey:cfg.key,'content-type':item.contentType,'x-upsert':'false'},body:item.bytes});
 if(res.status===409)return {path:item.path,duplicate:true};
 if(!res.ok)throw new Error('Supabase responded '+res.status);
 return {path:item.path,duplicate:false};
}
export async function sendOriginal(file,{env,client=null,fetchImpl=globalThis.fetch,outbox=memoryOutbox()}={}){
 const item=originalItem(file),cfg=client&&!readSupabaseEnv(env)?{bucket:env?.SUPABASE_BUCKET||DEFAULT_BUCKET}:readSupabaseEnv(env);
 const queue=async note=>{await outbox.put({...item,queuedNote:note});return {status:'pending-supabase',queued:true,sha256:item.sha256,path:item.path,note}};
 if(!cfg)return queue('Supabase env not set (SUPABASE_URL and a key); original kept in the outbox');
 try{const r=await upload(cfg,item,{client,fetchImpl});await outbox.remove(item.id);return {status:'uploaded',queued:false,sha256:item.sha256,bucket:cfg.bucket,...r}}
 catch(e){return queue('Upload failed, kept in outbox: '+e.message)}
}
export async function flushOutbox({env,client=null,fetchImpl=globalThis.fetch,outbox}){
 const out=[];for(const i of await outbox.list())out.push({id:i.id,...await sendOriginal({name:i.name,bytes:i.bytes,contentType:i.contentType,meta:i.meta},{env,client,fetchImpl,outbox})});return out;
}
