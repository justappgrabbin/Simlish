// node computer-tray/native-bridge.test.mjs — tray ↔ NativeApps bridge contract, with a fake Android bridge.
import assert from 'node:assert/strict';
const ls={}; globalThis.localStorage=new Proxy(ls,{get:(t,k)=>t[k],set:(t,k,v)=>(t[k]=String(v),true)});
const el=()=>({textContent:'',innerHTML:'',checked:false,value:'',hidden:true});
globalThis.document={querySelector:()=>el(),addEventListener(){}}; globalThis.addEventListener=()=>{};
globalThis.alert=m=>alerts.push(m); globalThis.confirm=()=>true; let answer='18.5.3.4.2'; globalThis.prompt=()=>answer;
globalThis.fetch=async()=>({json:async()=>[]}); const alerts=[];
const { openNative, p2pEligible } = await import('./apps-tray.mjs');
const { ledger } = await import('./address-gate.mjs');
const calls=[]; const installed=new Set(['app.prismkeys']);
const bridge={isInstalled:p=>installed.has(p),launch:(p,a)=>(calls.push(['launch',p,a]),JSON.stringify({ok:true})),openKeyboardSettings:()=>calls.push(['kbd']),
  installFromUrl:(u,id,a)=>calls.push(['dl',u,id,a]),installStored:(id,a)=>calls.push(['stored',id,a])};
assert.equal(openNative({id:'prism-keys',name:'Prism Keys',package:'app.prismkeys',opens:'keyboard-setup'},bridge),true);
assert.deepEqual(calls[0],['launch','app.prismkeys','18.5.3.4.2']);
assert.ok(ledger().some(x=>x.id==='prism-keys'),'launch logged in address ledger');
assert.equal(openNative({id:'res',name:'Res',package:'app.synthia.resonance.preview',apkSource:'https://github.com/x.apk'},bridge),true);
assert.equal(calls.at(-1)[0],'dl');
answer='99.9.9.9.9'; const before=calls.length;
assert.equal(openNative({id:'bad',name:'Bad',package:'a.b'},bridge),false); assert.equal(calls.length,before,'gate refusal blocks bridge');
answer='18.5.3.4.2';
assert.equal(openNative({id:'x',name:'X',package:'a.b'},null),false,'no bridge = graceful message');
assert.equal((await p2pEligible({}, {p2p:false})).ok,false);
assert.equal((await p2pEligible({}, {p2p:true,p2pByHD:false})).ok,true);
assert.equal((await p2pEligible({}, {p2p:true,p2pByHD:true},{requireDefinedCenters:[]})).ok,true);
assert.equal((await p2pEligible({definedCenters:['A']}, {p2p:true,p2pByHD:true},{requireDefinedCenters:['B']})).ok,false);
console.log('native bridge contract: all checks passed');
