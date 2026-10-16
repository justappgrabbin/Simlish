// Thin adapter: same-thread execution providers for Alexis's BrowserExecutionSurface (automata/browser-execution-surface.mjs, her patch).
// Her surface has register(provider) with canExecute()/execute(); nothing in her copied files is changed.
// Rules (hers): pure JavaScript, no external dependencies, no backend, no web worker. Only built-ins are used.
// What the providers do follows her own providers: JSON.parse and text decode (patch), WebAssembly.compile/instantiate (patch, left to her surface),
// self-contained JavaScript (her bridge's runJavaScriptText: AsyncFunction + console capture; her patch's iframe path needs `document`, which a same-thread host does not have).
import {BrowserExecutionSurface} from './automata/browser-execution-surface.mjs';
import {normalizeArtifact} from './automata/artifact-normalizer.mjs';
import {readZip} from '../zip-lite.mjs';
import {nuclear,hammingNeighbors} from '../state-space.mjs';
import {Population} from './population.mjs';

export const PROVIDER_INVESTIGATING=Object.freeze([
 'same-thread JavaScript cannot be sandboxed or interrupted without a web worker or vm (both ruled out). The guard is a deny-list of host/escape names plus a static check for obvious endless loops. It is a best-effort filter, NOT a security boundary: obfuscated code can get around it. Pass executeCode:false to analyse without running.',
 'JavaScript with import/export (module syntax) is not run: her surface records it as unresolved (module-loader). No module linker was written.',
 'language adapters other than JavaScript are placeholders: python, typescript/jsx and the other source languages run generic byte/primitive execution and record "adapter:<lang> (placeholder)" as unresolved.',
 'HTML scripts are not run without a DOM: structure is read, scripts are counted and recorded as unresolved (the iframe provider in her patch is for browsers).',
 'generic byte/primitive execution (hamming-1 steps between successive six-bit states, nuclear fold) is an engineering choice built on spec §2 (Hamming distance 1) and §36 (nuclear), not a rule Alexis stated for arbitrary files.']);

const AsyncFunction=Object.getPrototypeOf(async function(){}).constructor;
const extOf=n=>{const m=String(n||'').toLowerCase().match(/\.([a-z0-9_-]+)$/);return m?m[1]:''};
export const PLACEHOLDER_LANGUAGES=Object.freeze({py:'python',pyw:'python',ts:'typescript',tsx:'typescript',jsx:'jsx',rb:'ruby',go:'go',rs:'rust',java:'java',kt:'kotlin',c:'c',h:'c',cpp:'c++',cs:'c#',php:'php',sh:'shell',swift:'swift',lua:'lua'});
const jsonSafe=v=>{if(v===undefined)return null;try{const s=JSON.stringify(v);return s===undefined?String(v):JSON.parse(s)}catch{return String(v)}};
const flags={backendUsed:false,workerUsed:false};

// ---- JavaScript, same thread -------------------------------------------------
const DENY=/\b(process|require|globalThis|global|window|self|document|fetch|XMLHttpRequest|WebSocket|Worker|SharedWorker|importScripts|eval|Function|constructor|__proto__|Reflect|Proxy|Atomics|SharedArrayBuffer|WebAssembly|indexedDB|localStorage|sessionStorage|navigator|Deno|Bun|child_process|setImmediate|queueMicrotask|AsyncFunction|GeneratorFunction)\b|\bimport\s*\(/;
const ENDLESS=/\bwhile\s*\(\s*(?:true|1)\s*\)|\bfor\s*\(\s*;\s*;\s*\)/;
const SHADOW=['process','require','globalThis','global','window','self','document','fetch','XMLHttpRequest','WebSocket','Worker','navigator','Function','Reflect','Proxy'];
export const hasModuleSyntax=t=>/(^|\n)\s*(?:import|export)\b/.test(String(t??''));
export function guardJavaScript(source){
 const m=DENY.exec(source);if(m)return {ok:false,reason:'blocked by same-thread guard: "'+m[0]+'"',capability:'capability:'+m[0].replace(/\s*\($/,'')};
 if(ENDLESS.test(source)&&!/\b(?:break|return|throw)\b/.test(source))return {ok:false,reason:'blocked by same-thread guard: endless loop with no exit',capability:'capability:interruptible-execution'};
 return {ok:true};
}
export async function runJavaScript(source,{timeoutMs=2500,globals={}}={}){
 const lines=[];const push=level=>(...a)=>{if(lines.length<128)lines.push({level,args:a.map(v=>typeof v==='string'?v:jsonSafe(v))})};
 const cons=Object.freeze({log:push('log'),info:push('info'),warn:push('warn'),error:push('error')});
 const names=['console','synthiaGlobals',...SHADOW],values=[cons,Object.freeze({...globals}),...SHADOW.map(()=>undefined)];
 let timer;
 try{
  const fn=new AsyncFunction(...names,'"use strict";\n'+source+'\n');
  const value=await Promise.race([fn(...values),new Promise((_,rej)=>{timer=setTimeout(()=>rej(new Error('execution exceeded '+timeoutMs+'ms')),timeoutMs)})]);
  return {ok:true,result:{engine:'same-thread-async-function',stdout:lines,returnValue:jsonSafe(value)}};
 }catch(e){return {ok:false,error:String(e&&e.message||e),result:{engine:'same-thread-async-function',stdout:lines}}}
 finally{clearTimeout(timer)}
}
export const javascriptProvider=({executeCode=true}={})=>({id:'javascript-same-thread',
 canExecute:n=>n.kind==='javascript'&&!PLACEHOLDER_LANGUAGES[extOf(n.name)],
 async execute(n,ctx={}){
  const base={kind:'javascript',path:'same-thread-js',...flags};
  if(hasModuleSyntax(n.text))return {...base,ok:false,error:'module graph requires dependency linking before execution',unresolvedDependencies:['module-loader']};
  if(ctx.executeCode===false||!executeCode)return {...base,ok:true,path:'same-thread-js-analysis-only',result:{engine:'not-run (executeCode:false)',stdout:[],returnValue:null}};
  const g=guardJavaScript(n.text);
  if(!g.ok)return {...base,ok:false,error:g.reason,unresolvedDependencies:[g.capability]};
  const r=await runJavaScript(n.text,{timeoutMs:ctx.timeoutMs??2500,globals:ctx.globals??{}});
  return {...base,...r};
 }});

// ---- CSS: structural parse (no DOM, nothing rendered) ----------------------------
export function parseCss(text){
 const s=String(text).replace(/\/\*[\s\S]*?\*\//g,''),stack=[];let rules=0,atRules=0,declarations=0,selectors=0,buf='';const custom=new Set(),errors=[];
 for(let i=0;i<s.length;i++){const c=s[i];
  if(c==='{'){const head=buf.trim();buf='';if(head.startsWith('@')){atRules++;stack.push('at')}else{rules++;selectors+=head.split(',').filter(x=>x.trim()).length;stack.push('rule')}}
  else if(c==='}'){if(!stack.length){errors.push('unmatched } at '+i);continue}
   if(stack.pop()==='rule'&&buf.trim())declarations++;buf=''}
  else if(c===';'){if(stack[stack.length-1]==='rule'){declarations++;const d=buf.trim();if(d.startsWith('--'))custom.add(d.split(':')[0].trim())}buf=''}
  else buf+=c}
 if(stack.length)errors.push(stack.length+' unclosed block(s)');
 return {rules,atRules,selectors,declarations,customProperties:[...custom].sort(),errors};
}
export const cssProvider=()=>({id:'css-structural',canExecute:n=>extOf(n.name)==='css',
 async execute(n){const r=parseCss(n.text);return {kind:'css',path:'same-thread-css-parse',...flags,ok:r.errors.length===0,...(r.errors.length?{error:r.errors.join('; ')}:{}),result:{engine:'css-structural-parse',stdout:[],returnValue:r}}}});

// ---- HTML: structure only -----------------------------------------------------------
export function parseHtml(text){
 const tags=[...String(text).matchAll(/<([a-zA-Z][a-zA-Z0-9-]*)\b[^>]*>/g)].map(m=>m[1].toLowerCase()),count={};for(const t of tags)count[t]=(count[t]||0)+1;
 const scripts=[...String(text).matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/gi)],external=scripts.filter(m=>/\bsrc\s*=/.test(m[1])).length;
 return {tagCount:tags.length,tags:Object.fromEntries(Object.entries(count).sort()),scripts:scripts.length,externalScripts:external,inlineScripts:scripts.length-external,title:(/<title[^>]*>([\s\S]*?)<\/title>/i.exec(text)||[])[1]?.trim()||null};
}
export const htmlProvider=()=>({id:'html-structural',canExecute:n=>['html','htm'].includes(extOf(n.name)),
 async execute(n){const r=parseHtml(n.text),un=[];if(r.scripts)un.push('provider:html-dom (scripts not run in a same-thread host without a DOM)');if(r.externalScripts)un.push('network:external-script-src (no network used)');
  return {kind:'html',path:'same-thread-html-parse',...flags,ok:true,unresolvedDependencies:un,result:{engine:'html-structural-parse',stdout:[],returnValue:r}}}});

// ---- text: summary instead of echoing the whole text ---------------------------------
export const textProvider=()=>({id:'text-summary',canExecute:n=>n.kind==='text'&&!PLACEHOLDER_LANGUAGES[extOf(n.name)],
 async execute(n){const t=n.text,lines=t.split('\n'),csv=extOf(n.name)==='csv'?lines.filter(Boolean).map(l=>l.split(',').length):null;
  return {kind:'text',path:'same-thread-text',...flags,ok:true,result:{engine:'TextDecoder',stdout:[],returnValue:{chars:t.length,lines:lines.length,words:(t.match(/\S+/g)||[]).length,...(csv?{rows:csv.length,columns:Math.max(0,...csv)}:{})}}}}});

// ---- generic byte/primitive execution: any file, no language needed -------------------------
export function genericExecute(bytes){
 const states=[];let buf=0,nb=0;for(const b of bytes){buf=(buf<<8)|b;nb+=8;while(nb>=6){states.push((buf>>>(nb-6))&63);nb-=6;buf&=nb?(1<<nb)-1:0}}
 if(nb)states.push((buf<<(6-nb))&63);
 const hist=new Array(64).fill(0);let h1=0,xor=0,nuc=0;
 const near=new Set();for(let a=0;a<64;a++)for(const h of hammingNeighbors(a,1))near.add(a*64+h.code);
 for(let i=0;i<states.length;i++){hist[states[i]]++;xor^=states[i];nuc=(nuc*31+nuclear(states[i]))>>>0;if(i&&near.has(states[i-1]*64+states[i]))h1++}
 const top=hist.map((c,code)=>({code,count:c})).filter(x=>x.count).sort((a,b)=>b.count-a.count||a.code-b.code).slice(0,5);
 return {sixBitStates:states.length,distinctStates:hist.filter(Boolean).length,hamming1Steps:h1,hamming1Ratio:states.length>1?+(h1/(states.length-1)).toFixed(4):0,stateXor:xor,nuclearFold:nuc,topStates:top};
}
export const placeholderProvider=()=>({id:'language-placeholder',canExecute:n=>!!PLACEHOLDER_LANGUAGES[extOf(n.name)]&&n.kind!=='wasm'&&n.kind!=='zip',
 async execute(n){const lang=PLACEHOLDER_LANGUAGES[extOf(n.name)];
  return {kind:n.kind,path:'generic-primitive-execution',...flags,ok:true,unresolvedDependencies:['adapter:'+lang+' (placeholder)'],result:{engine:'generic-primitive-execution',stdout:[],returnValue:{language:lang,placeholder:true,...genericExecute(n.bytes)}}}}});
export const genericProvider=()=>({id:'generic-primitive',canExecute:n=>n.kind!=='json'&&n.kind!=='wasm'&&n.kind!=='zip'&&n.kind!=='javascript',
 async execute(n){return {kind:n.kind,path:'generic-primitive-execution',...flags,ok:true,result:{engine:'generic-primitive-execution',stdout:[],returnValue:genericExecute(n.bytes)}}}});

// ---- zip: each member is executed by the same surface (one level; nested archives are recorded, not opened) --------------------
export const zipProvider=(surface,{maxEntries=32,maxEntryBytes=1<<20,maxTotalBytes=4<<20}={})=>({id:'zip-members',canExecute:n=>n.kind==='zip',
 async execute(n,ctx={}){
  const base={kind:'zip',path:'same-thread-zip-members',...flags};
  let entries;try{entries=await readZip(n.bytes)}catch(e){return {...base,ok:false,error:'zip not readable: '+e.message}}
  const members=[],skipped=[];let total=0;
  for(const e of entries){
   if(members.length>=maxEntries||e.data.length>maxEntryBytes||total+e.data.length>maxTotalBytes){skipped.push({path:e.path,reason:'limit'});continue}
   if(ctx.depth>=1&&/\.(zip|jar|apk)$/i.test(e.path)){skipped.push({path:e.path,reason:'nested archive recorded, not opened'});continue}
   total+=e.data.length;
   const nm=await normalizeArtifact({name:e.path,bytes:e.data});
   const r=await surface.execute(nm,{...ctx,depth:(ctx.depth||0)+1});
   members.push({path:e.path,kind:nm.kind,ok:!!r.ok,engine:r.result?.engine??r.path??null,error:r.ok?null:r.error??null,unresolved:r.unresolvedDependencies??[]});
  }
  const bad=members.filter(m=>!m.ok);
  return {...base,ok:bad.length===0,...(bad.length?{error:bad.length+' member(s) failed: '+bad.map(m=>m.path).join(', ')}:{}),
   unresolvedDependencies:[...new Set(members.flatMap(m=>m.unresolved))],result:{engine:'zip-member-execution',stdout:[],returnValue:{members,skipped,entries:entries.length}}};
 }});

// ---- primitive organizations (.prim): resolved and executed by the shared population (population.mjs), not by a file-specific engine ----
export const primitiveProvider=()=>({id:'primitive-population',canExecute:n=>extOf(n.name)==='prim',
 async execute(n){const r=new Population().resolve(n.bytes,{name:n.name}),top=r.program.top,out=[];const inst=r.instance();for(let t=0;t<8;t++)out.push(inst.tick(Array(top.nIn).fill(0)));
  return {kind:'primitive-organization',path:'population-resolution',...flags,ok:true,result:{engine:'population-tick',stdout:[],returnValue:{wellFormed:r.program.wellFormed,definitions:r.program.defs.length,nIn:top.nIn,nOut:top.nOut,actsAs:top.composite,address:r.address.key,firstTicksWithZeroInput:out}}}}});

// Build her surface with the same-thread providers registered (order matters: first match wins; json/wasm fall through to her built-ins).
export function createSurface({executeCode=true}={}){
 const surface=new BrowserExecutionSurface();
 for(const p of [primitiveProvider(),javascriptProvider({executeCode}),cssProvider(),htmlProvider(),zipProvider(surface),placeholderProvider(),textProvider(),genericProvider()])surface.register(p);
 return surface;
}
