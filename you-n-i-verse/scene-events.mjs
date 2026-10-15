import {resolveAddress} from './address-resolution.mjs';
import {autoLingAutomaton,autoNovelAutomaton} from './vendor/klein/index.mjs';
const worlds=new Set(['cosmic','forest','ocean','desert']);
export async function interpretScene(text,{kernel=globalThis.AgentomagotchiKernel,addressContext=null}={}){
 const normalized=text.trim().toLowerCase().replace(/[.!]$/,'');if(!/^you enters (cosmic|forest|ocean|desert)$/.test(normalized))return {ok:false,reason:'Use a supported scene: you enters cosmic, forest, ocean or desert. Other words remain unresolved.'};
 const ling=autoLingAutomaton();await ling.call({operation:'learn',rule:{id:'scene-enter.v1',relation:'ENTERS',pattern:['$actor','enters','$world'],template:'$actor enters $world'}});
 const parsed=await ling.call({operation:'recognize',text:normalized});const match=parsed.matches[0];const structure=match?[match.bindings['$actor'],'enters',match.bindings['$world']]:null;if(!structure||!worlds.has(structure[2]))return {ok:false,reason:'No compatible world'};
 if(!kernel)throw Error('Address kernel unavailable');
 const resolved=resolveAddress({kernel,...(addressContext||{})});
 if(!resolved.ok)return {ok:false,reason:resolved.reason,needs:resolved.needs,structure,world:structure[2]};
 const projections=resolved.projections;
 const novel=autoNovelAutomaton();await novel.call({operation:'register',domain:{id:'world-entry',primitives:[{id:'actor'},{id:'world'}],combinators:[{id:'entry.sequence.v1',type:'sequence',inputs:['actor','world'],output:'scene',apply:(actor,world)=>({actor,world})}]}});
 const story=await novel.call({operation:'generate',spec:{domain:'world-entry',seeds:[{type:'actor',value:structure[0]},{type:'world',value:structure[2]}]}});
 return {ok:true,event:{id:globalThis.crypto?.randomUUID?.()??`entry-${Date.now()}`,grammarRule:'scene-enter.v1',structure,world:structure[2],lineage:story.lineage,projections,addressSource:resolved.evidence.source,leadingDimension:resolved.leadingDimension,effect:'world-surface-treatment',recordedAt:new Date().toISOString()}};
}
