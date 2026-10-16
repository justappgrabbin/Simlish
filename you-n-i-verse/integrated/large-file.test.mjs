import test from 'node:test';import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';import {createHash} from 'node:crypto';import {fileURLToPath} from 'node:url';import {dirname,join} from 'node:path';
import '../vendor/kernel.js';
import {executeDAG} from './spine/dag-runtime.mjs';
import {ExecutionPipeline,MemoryExecutionPipelineStore} from './spine/execution-pipeline.mjs';
import {ExecutionAddressResolver} from './spine/execution-address-resolver.mjs';
const here=dirname(fileURLToPath(import.meta.url)),sha=f=>createHash('sha256').update(readFileSync(join(here,f))).digest('hex');
test('her original dag-runtime is kept byte-for-byte beside the repaired one (original sha of the zip copies: 7ecf09727e...)',()=>{assert.ok(sha('spine/dag-runtime.original.mjs').startsWith('7ecf09727e'));assert.notEqual(sha('spine/dag-runtime.mjs'),sha('spine/dag-runtime.original.mjs'))});
test('repaired DAG: graphs of <=64 nodes still checkpoint after every node; large graphs are bounded',async()=>{
 const g=n=>Array.from({length:n},(_,i)=>({id:'n'+i,dependsOn:i?['n'+(i-1)]:[]}));
 assert.equal((await executeDAG(g(10),async()=>1)).checkpoints.length,11);
 const big=await executeDAG(g(500),async()=>1);assert.ok(big.ok&&big.checkpoints.length<=70&&Object.values(big.status).every(s=>s==='done'));
});
test('her pipeline runs a file over 3072 bytes (3073 bytes = 4098 states) and a 20000-byte file; the 4096-state cap of her descend is reported, not hidden',async()=>{
 const res=new ExecutionAddressResolver();
 for(const N of [3073,20000]){
  const bytes=Uint8Array.from({length:N},(_,i)=>(i*131+7+(i>>5))&255);
  const p=new ExecutionPipeline({executeArtifactCore:async(a,c)=>{const r=res.resolve({artifact:a,execution:{ok:true,path:'x',kind:'binary',result:{engine:'stub'}},decomposition:c.executionDecomposition});return {ok:true,kind:'binary',path:'x',result:{engine:'stub'},canonicalAddress:r.address,canonicalAddressKey:r.key,addressBasis:r.derivation.basis,executionDerivedAddress:r.address,executionDerivedAddressKey:r.key}},addressResolver:res,store:new MemoryExecutionPipelineStore()});
  const r=await p.execute({name:'h.bin',bytes},{executeCode:false});
  assert.equal(r.pipeline.primitiveGraph.nodeCount,4096,'cap inherited from her descend (execution-address-resolver.mjs line 87)');
  assert.ok(Math.ceil(N*8/6)>4096);assert.ok(r.pipeline.fold.fields.every(f=>f.primitiveCount>700),'the 4096 primitives are split over her five fields');
 }
});
