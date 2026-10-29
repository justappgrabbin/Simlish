import {BASE_CHAINS,resolveAnchor,CHAIN_SOURCE} from './causal-chains.mjs';
import {evaluateConditions} from './causal-conditions.mjs';
export function traverseCausal(input,{chains=BASE_CHAINS,verification=null,explore=false}={}){
 const required=['causal-chain-placement','gate-anchor-placement','line-trajectory','resonance-weight'];
 if(!explore&&!required.every(id=>verification?.canApply(id,'empirical')))return {status:'awaiting-verification',emitted:false,admitted:false,required};
 const conditions=evaluateConditions(input,{chains});if(conditions.status!=='conditions-valid')return {...conditions,emitted:false};
 const chain=chains[input.base],entry=resolveAnchor(chain,conditions.anchor.anchor),nodes=entry.layer==='macro'?chain.nodes:chain.microNodes;
 // Line 1/2 continue along supplied adjacent edges. Line 6's terminal leap is
 // explicitly authorized by the user specification, not a fabricated chain edge.
 const path=input.line===6?(entry.index===nodes.length-1?[nodes.at(-1)]:[nodes[entry.index],nodes.at(-1)]):nodes.slice(entry.index);
 const steps=path.slice(1).map((to,i)=>({from:path[i],to,rule:input.line===6?'line-6-terminal-leap':'source-sequential-edge',source:input.line===6?conditions.source:CHAIN_SOURCE,weight:conditions.compatibility.weight}));
 return {...conditions,status:explore?'hypothesis-traversed':'traversed',emitted:!explore,hypothesis:explore,entry,path,steps,embodiment:chain.embodiment,dimension:chain.dimension,identityId:input.identityId??null,source:CHAIN_SOURCE};
}
