import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync,existsSync} from 'node:fs';
import {Trident,tokenize,detokenize,loadStateDict,CONFIG} from './trident-js.mjs';
const PT=new URL('../models/trident_syntia.pt',import.meta.url),FX=JSON.parse(readFileSync(new URL('./reference/hf-synthia/trident-fixture.json',import.meta.url),'utf8'));
const have=existsSync(PT);
test('tokenize follows train_syntia.py (ord % 4096 on lowercased text, 64 chars)',()=>{assert.deepEqual(tokenize('Gate 57'),[103,97,116,101,32,53,55]);assert.equal(tokenize('x'.repeat(100)).length,64);assert.equal(detokenize(tokenize('abc')),'abc')});
test('her trident_syntia.pt loads: 125 tensors, parameter count matches the architecture',{skip:!have},async()=>{
 const sd=await loadStateDict(new Uint8Array(readFileSync(PT)));assert.equal(sd.tensors.size,125);
 assert.deepEqual(sd.tensors.get('embed.weight').shape,[4096,128]);assert.deepEqual(sd.tensors.get('heads.code.lm_head.weight').shape,[4096,128]);
 let n=0;for(const [k,t] of sd.tensors)if(k!=='pos_enc.pe')n+=t.data.length;assert.equal(n,3594755,'parameter count')});
test('forward matches real PyTorch (fixture produced by running model.py on her .pt): router, per-head top ids and values, greedy continuation',{skip:!have},async()=>{
 const m=await Trident.fromBytes(new Uint8Array(readFileSync(PT)));
 for(const r of FX){
  const {logits,router}=m.forward(r.ids);
  r.router.forEach((v,i)=>assert.ok(Math.abs(v-router[i])<2e-4,'router '+r.text));
  const top=[...logits.keys()].sort((a,b)=>logits[b]-logits[a]).slice(0,5);assert.deepEqual(top,r.last_top,'top5 '+r.text);
  r.last_top_vals.forEach((v,i)=>assert.ok(Math.abs(v-logits[r.last_top[i]])<2e-3,'value '+r.text));
  for(const h of CONFIG.headNames){const o=m.forward(r.ids,{head:h}).logits,t=[...o.keys()].sort((a,b)=>o[b]-o[a]).slice(0,3);assert.deepEqual(t,r['head_'+h].top,h+' '+r.text)}
  const g=m.generate(r.ids,{maxNew:12,temp:0});assert.deepEqual(g.slice(-12),r.greedy12,'greedy '+r.text)}});
test('sampling is repeatable with an injected random source and stays inside the vocabulary',{skip:!have},async()=>{
 const m=await Trident.fromBytes(new Uint8Array(readFileSync(PT)));let s=1;const rand=()=>(s=(s*16807)%2147483647)/2147483647;
 const a=m.generate(tokenize('gate'),{maxNew:6,rand});s=1;const b=m.generate(tokenize('gate'),{maxNew:6,rand});assert.deepEqual(a,b);assert.ok(a.every(i=>i>=0&&i<4096))});
