// Transcribed from supplied screenshots: printed pages 5 and 115.
const macro=[['Movement','Energy','Creation','Seeing','Landscape','Environment'],['Evolution','Gravity','Memory','Taste','Love','Light'],['Being','Matter','Touch','Sex','Survival'],['Design','Structure','Progress','Smell','Life','Art'],['Space','Form','Illusion','Hearing','Music','Freedom']];
const micro=[['Activity','Reaction','Limitation','Perspective','Relation'],['Character','Separation','Nature','Integration','Spirit'],['Biology','Chemistry','Objectivity','Geometry','Trajectory'],['Homo Sapiens','Growth','Decay','Continuity','Manifestation'],['Type','Fantasy','Subjectivity','Rhythm','Timing']];
const embodiment=['Individuality','The Mind','The Body','The Ego','Personality'];
export const CHAIN_SOURCE='user-screenshots/printed-pages-5-and-115';
export const BASE_CHAINS=Object.freeze(Object.fromEntries(macro.map((nodes,i)=>[i+1,Object.freeze({base:i+1,dimension:nodes[0],nodes:Object.freeze(nodes),microNodes:Object.freeze(micro[i]),embodiment:embodiment[i],edges:Object.freeze(nodes.slice(1).map((to,j)=>Object.freeze({from:nodes[j],to,source:CHAIN_SOURCE}))),microEdges:Object.freeze(micro[i].slice(1).map((to,j)=>Object.freeze({from:micro[i][j],to,source:CHAIN_SOURCE}))),source:CHAIN_SOURCE,notes:i===2?'Printed Being chain contains an incomplete “Matter is.” clause; no extra node invented.':i===3?'Other supplied pages print Smelt/Smelling. This page-5 chain retains Smell.':null})])));
export function resolveAnchor(chain,label){
 if(!chain)return null;const parts=label.split(' / ');
 for(const [layer,nodes] of [['macro',chain.nodes],['micro',chain.microNodes]]){const indices=parts.map(p=>nodes.indexOf(p));if(indices.every(i=>i>=0)&&indices.every((v,i)=>i===0||v===indices[i-1]+1))return {layer,index:indices[0],nodes:parts};}
 return null;
}
