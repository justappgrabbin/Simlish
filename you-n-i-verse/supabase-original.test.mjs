import test from 'node:test';import assert from 'node:assert/strict';
import {sendOriginal,flushOutbox,memoryOutbox,storageOutbox,readSupabaseEnv,fileOutbox} from './supabase-original.mjs';
import {sha256} from './sha256.mjs';
const bytes=new Uint8Array([9,8,7,0,255]),file={name:'my app.zip',bytes,contentType:'application/zip'};
const mockClient=(result={error:null})=>{const calls=[];return {calls,storage:{from:bucket=>({upload:async(path,body,opts)=>{calls.push({bucket,path,body,opts});return result}})}}};
test('env missing: original queued untouched and marked pending-supabase, never throws',async()=>{
 const outbox=memoryOutbox(),r=await sendOriginal(file,{env:{},outbox});
 assert.equal(r.status,'pending-supabase');const q=await outbox.list();assert.equal(q.length,1);assert.deepEqual(q[0].bytes,bytes);assert.equal(q[0].sha256,sha256(bytes));
 assert.equal(readSupabaseEnv({SUPABASE_URL:'https://x.supabase.co'}),null);
});
test('mocked client uploads the exact original bytes, then clears the outbox',async()=>{
 const client=mockClient(),outbox=memoryOutbox();
 await outbox.put({id:sha256(bytes).slice(0,24),name:'x',bytes});
 const r=await sendOriginal(file,{env:{SUPABASE_URL:'https://x.supabase.co',SUPABASE_ANON_KEY:'k'},client,outbox});
 assert.equal(r.status,'uploaded');assert.equal(client.calls.length,1);assert.deepEqual(client.calls[0].body,bytes);
 assert.equal(client.calls[0].bucket,'simlish-originals');assert.match(client.calls[0].path,/^originals\/[0-9a-f]{64}\/my_app\.zip$/);assert.equal(client.calls[0].opts.upsert,false);
 assert.equal((await outbox.list()).length,0);
});
test('upload failure queues the original instead of failing; flush retries and succeeds',async()=>{
 const outbox=memoryOutbox(),env={SUPABASE_URL:'https://x.supabase.co',SUPABASE_SERVICE_KEY:'k',SUPABASE_BUCKET:'b'};
 const bad=mockClient({error:{message:'network down'}});
 assert.equal((await sendOriginal(file,{env,client:bad,outbox})).status,'pending-supabase');assert.equal((await outbox.list()).length,1);
 const good=mockClient();const res=await flushOutbox({env,client:good,outbox});
 assert.equal(res[0].status,'uploaded');assert.equal(good.calls[0].bucket,'b');assert.equal((await outbox.list()).length,0);
});
test('REST path sends bearer key and original bytes via a mocked fetch; 409 means already stored',async()=>{
 const seen=[];const fetchImpl=async(url,o)=>{seen.push({url,o});return {ok:seen.length>1?false:true,status:seen.length>1?409:200}};
 const env={SUPABASE_URL:'https://x.supabase.co/',SUPABASE_ANON_KEY:'anon'};
 const r=await sendOriginal(file,{env,fetchImpl,outbox:memoryOutbox()});assert.equal(r.status,'uploaded');
 assert.match(seen[0].url,/^https:\/\/x\.supabase\.co\/storage\/v1\/object\/simlish-originals\/originals\//);assert.equal(seen[0].o.headers.authorization,'Bearer anon');assert.deepEqual(seen[0].o.body,bytes);
 const dup=await sendOriginal(file,{env,fetchImpl,outbox:memoryOutbox()});assert.equal(dup.status,'uploaded');assert.equal(dup.duplicate,true);
});
test('storage and file outboxes keep original bytes intact',async()=>{
 const m=new Map(),s={getItem:k=>m.get(k)??null,setItem:(k,v)=>m.set(k,v)},o=storageOutbox(s);
 await sendOriginal(file,{env:{},outbox:o});assert.deepEqual((await o.list())[0].bytes,bytes);
 const {mkdtemp}=await import('node:fs/promises'),{tmpdir}=await import('node:os'),{join}=await import('node:path');
 const f=await fileOutbox(await mkdtemp(join(tmpdir(),'ob-')));await sendOriginal(file,{env:{},outbox:f});const l=await f.list();assert.deepEqual(l[0].bytes,bytes);await f.remove(l[0].id);assert.equal((await f.list()).length,0);
});
