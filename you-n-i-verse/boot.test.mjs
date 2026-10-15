import test from 'node:test';import assert from 'node:assert/strict';
import './vendor/kernel.js';
import {bootSimlish,currentAddressContext,CAPABILITIES} from './boot.mjs';
import {personalize} from './personalize.mjs';
import {readSimlishRegistrations,pendingRegistrations} from './synthia-registration.mjs';
import {memoryOutbox,sendOriginal} from './supabase-original.mjs';
const mem=()=>({d:{},getItem(k){return this.d[k]??null},setItem(k,v){this.d[k]=v}});
const address={planetary:'Sun',dimension:'Being',zodiac:'Virgo',gate:6,line:4,color:4,tone:4,base:2,degree:25,minute:59,second:31,arcSecond:92,house:5};
test('startup registers Simlish as investigating with no address, then registered once the uploader gives the full address',async()=>{
 const s=mem();let r=await bootSimlish({storage:s});
 assert.equal(r.registration.status,'investigating');assert.equal(r.sync.status,'skipped');assert.equal(currentAddressContext(s),null);
 assert.deepEqual(readSimlishRegistrations(s).simlish.capabilities,[...CAPABILITIES]);assert.ok(CAPABILITIES.includes('app-finisher'));
 personalize({name:'Alexis',worldTheme:'forest',address},s);
 r=await bootSimlish({storage:s});assert.equal(r.registration.status,'registered');assert.equal(currentAddressContext(s).leadingDimension,'Being');
});
test('startup calls syncRegistrations when an endpoint is configured and clears the outbox on success',async()=>{
 const s=mem();const calls=[];
 const r=await bootSimlish({storage:s,env:{SIMLISH_REGISTRY_ENDPOINT:'https://example.test/register'},fetchImpl:async(u,o)=>{calls.push(JSON.parse(o.body));return {ok:true,status:200}}});
 assert.equal(calls.length,1);assert.equal(calls[0].registration.id,'simlish');assert.deepEqual(pendingRegistrations(s),{});assert.equal(r.sync.status,'attempted');
});
test('startup survives a failing endpoint (stays pending) and flushes queued Supabase originals when configured',async()=>{
 const s=mem(),outbox=memoryOutbox(),bytes=new Uint8Array([1,2,3]);
 await sendOriginal({name:'a.zip',bytes},{env:{},outbox});
 const uploads=[];const client={storage:{from:()=>({upload:async(p,b)=>{uploads.push(b);return {error:null}}})}};
 const r=await bootSimlish({storage:s,env:{SIMLISH_REGISTRY_ENDPOINT:'https://example.test/r',SUPABASE_URL:'https://x.supabase.co',SUPABASE_ANON_KEY:'k'},fetchImpl:async()=>({ok:false,status:503}),client,outbox});
 assert.ok(pendingRegistrations(s).simlish);assert.equal(r.originals.results[0].status,'uploaded');assert.deepEqual(uploads[0],bytes);
});
