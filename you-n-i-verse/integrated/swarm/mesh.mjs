// The emergent mesh of the one swarm area.
// Spec (her swarmdoc.md sections 4-7, used as written): swarm units are primitive particles; a unit has no permanent role (its function comes from address, position, dimension, relations, state, context);
//   units cluster when they discover structural relationships (shared address components, transformation paths, dimensional qualities, position, history, state);
//   a mesh edge records source -> relation -> target with address + dimension + transformation + context + provenance, and edges strengthen, branch and combine as experience accumulates.
// EXISTING, used as-is:
//   vendor/mesh/state-space/mesh-state-space.js (her StateSpace: 5 dimension layers x 64 gate nodes, each layer in its own sequence order) for node identity, trigrams and layer position;
//   vendor/mesh/dimension-qualities.mjs (her per-dimension macro chain and micro nature, Black Book p126 statements) for each node's dimensional qualities;
//   state-space.mjs: hammingNeighbors/bitsOf distance, complement, reverseLines, exchangeTrigrams, nuclear, gateToMacro, macroToGate; chart-engine.mjs CHANNELS (her 36 channels);
//   measure.mjs: the six-bit packing (descend without its slice) and her pipeline field rule (state+index)%5, so a unit's dimension is decided exactly as in her pipeline.
// NOT hand-wired: there is no edge table. The only way an edge exists is that two units INTERACT (they sit within `window` places of each other in the content) and a relation predicate over their states holds.
//   The predicates are her functions evaluated over the 64x64 state pairs once (REL table), so nothing is typed in.
// NEW, derived — please confirm: the interaction window (units i and i+1..i+W interact), weight = number of interactions, decay/prune adaptation, weighted label-propagation clusters.
import {StateSpace as MeshSpace} from '../../vendor/mesh/state-space/mesh-state-space.js';
import {DIMENSION_QUALITIES} from '../../vendor/mesh/dimension-qualities.mjs';
import {DIMENSIONS,complement,reverseLines,exchangeTrigrams,nuclear,gateToMacro,macroToGate,bitsOf} from '../../state-space.mjs';
import {CHANNELS} from '../chart-engine.mjs';
import {sixBitStatesAll,fieldIndexForPrimitive} from '../measure.mjs';
import {sha256} from '../../sha256.mjs';
export const RELATIONS=Object.freeze(['identity','hamming-1','complement','reverse','exchange-trigrams','nuclear','channel']);
export const SYMMETRIC=Object.freeze({identity:true,'hamming-1':true,complement:true,reverse:true,'exchange-trigrams':true,nuclear:false,channel:true});
export const RULE_SOURCE=Object.freeze({identity:'the same six-bit state held in two dimension layers (her layers differ in order, and Movement/Space seat different states under one slot label: see layerSlot)','hamming-1':'one line flipped: state-space.mjs hammingNeighbors radius 1 (her field_core channelEmerges Hamming-1 emergence)',complement:'state-space.mjs complement',reverse:'state-space.mjs reverseLines','exchange-trigrams':'state-space.mjs exchangeTrigrams',nuclear:'state-space.mjs nuclear (directed)',channel:'chart-engine.mjs CHANNELS (her 36 channels)'});
const popc=x=>{let n=0;while(x){n+=x&1;x>>=1}return n};
// relation table over all 64x64 state pairs, computed from her functions (bit i of REL[a*64+b] = RELATIONS[i] holds from a to b)
export const REL=(()=>{const t=new Uint8Array(4096),ch=new Set();for(const [x,y] of CHANNELS){const a=gateToMacro(x),b=gateToMacro(y);ch.add(a*64+b);ch.add(b*64+a)}
 for(let a=0;a<64;a++)for(let b=0;b<64;b++){let m=0;if(a===b)m|=1;if(popc(a^b)===1)m|=2;if(b===complement(a))m|=4;if(a!==b&&b===reverseLines(a))m|=8;if(a!==b&&b===exchangeTrigrams(a))m|=16;if(a!==b&&b===nuclear(a))m|=32;if(ch.has(a*64+b))m|=64;t[a*64+b]=m}return t})();
// analytic potential: how many edges the full state space can ever hold (nodes are (dimension, state))
export function potentialEdges(){const out={},nD=DIMENSIONS.length,dimPairs=nD*(nD-1)/2;let total=0;
 RELATIONS.forEach((r,i)=>{let ordered=0,orderedNoSelf=0;for(let a=0;a<64;a++)for(let b=0;b<64;b++)if(REL[a*64+b]>>i&1){ordered++;if(a!==b)orderedNoSelf++}
  const within=r==='identity'?0:SYMMETRIC[r]?nD*orderedNoSelf/2:nD*orderedNoSelf; // within one dimension layer
  const cross=r==='identity'?dimPairs*ordered:SYMMETRIC[r]?dimPairs*ordered:nD*(nD-1)*ordered; // between two different dimension layers
  out[r]={withinDimension:within,crossDimension:cross,total:within+cross};total+=within+cross});return {byRelation:out,total}}
const nodeIdx=(d,s)=>d*64+s;
export class EmergentMesh{
 constructor({primitives=null,window=2}={}){this.space=new MeshSpace();this.window=window;this.prim=primitives?new Map(primitives.primitives.map(p=>[p.id,p])):null;
  this.nodes=new Map();this.edges=new Map();this.contents=new Map();this.epoch=0;this.interactions=0;this.units=0;
  // her layers hold hexagrams by binary: look a unit's state up in the layer by that binary (her `binary` arrays are line 1 first, bitsOf is top line first, so the key is reversed), not by gate label (Movement and Space hold transformed states under the same slot label)
  this.byBinary=Object.fromEntries(DIMENSIONS.map(d=>[d,new Map(this.space.dimensions[d].nodes.map((n,i)=>[n.binary.slice().reverse().join(''),{node:n,index:i}]))]))}
 reset(){this.nodes=new Map();this.edges=new Map();this.contents=new Map();this.epoch=0;this.interactions=0;this.units=0}
 // ---- adaptive growth ----
 weave(bytes,{name='content',window=this.window,maxUnits=1500000}={}){
  const id=sha256(bytes).slice(0,12),states=sixBitStatesAll(bytes),n=Math.min(states.length,maxUnits);let newE=0,newN=0,touched=0;const ci=this.contents.get(id)||{id,name,units:0,weaves:0};ci.units+=n;ci.weaves++;this.contents.set(id,ci);
  const dims=new Uint8Array(n);for(let i=0;i<n;i++)dims[i]=fieldIndexForPrimitive(states[i],i);
  for(let i=0;i<n;i++){const ni=nodeIdx(dims[i],states[i]);let nd=this.nodes.get(ni);if(!nd){nd={visits:0,byContent:{}};this.nodes.set(ni,nd);newN++}nd.visits++;nd.byContent[id]=(nd.byContent[id]||0)+1;
   for(let k=1;k<=window&&i+k<n;k++){const a=states[i],b=states[i+k],m=REL[a*64+b];if(!m)continue;this.interactions++;const na=ni,nb=nodeIdx(dims[i+k],b);
    for(let r=0;r<7;r++){if(!(m>>r&1))continue;if(r===0&&na===nb)continue;if(r!==0&&na===nb)continue;
     let s=na,t=nb;if(SYMMETRIC[RELATIONS[r]]&&s>t){s=nb;t=na}
     const key=(r*320+s)*320+t;let e=this.edges.get(key);if(!e){e={r,s,t,w:0,first:this.units+i,last:0,byContent:{}};this.edges.set(key,e);newE++}
     e.w++;e.last=this.units+i;e.byContent[id]=(e.byContent[id]||0)+1;touched++;
     if(r!==0&&!this.nodes.has(nb)){this.nodes.set(nb,{visits:0,byContent:{}})}}}}
  this.units+=n;this.epoch++;return {content:id,units:n,truncated:states.length>n,newNodes:newN,newEdges:newE,edgeStrengthenings:touched}}
 // forget a little of everything and drop what falls below `floor`: edges not re-made by new experience fade
 adapt({decay=0.1,floor=1}={}){let dropped=0;for(const [k,e] of this.edges){e.w*=1-decay;if(e.w<floor){this.edges.delete(k);dropped++}}return {dropped,edges:this.edges.size}}
 // ---- reading ----
 nodeId(idx){const d=idx>>6,s=idx&63;return DIMENSIONS[d]+':'+macroToGate(s)}
 node(idx){const d=idx>>6,s=idx&63,dim=DIMENSIONS[d],gate=macroToGate(s),hit=this.byBinary[dim].get(bitsOf(s)),ln=hit.node,layer=this.space.dimensions[dim],q=DIMENSION_QUALITIES[dim],obs=this.nodes.get(idx),p=this.prim?.get('gate.'+gate)||null;
  return {id:dim+':'+gate,dimension:dim,gate,state:s,binary:bitsOf(s),trigrams:ln.trigrams,layerRole:layer.layerRole,sequence:layer.sequenceName,layerSlot:ln.gate,layerPosition:hit.index,
   macroChain:q?.macroChain.map(x=>x.value)??null,micro:q?{name:q.micro.value.name,nature:q.micro.value.nature,keynote:q.micro.value.keynote}:null,gateForm:p?.form??null,visits:obs?.visits??0}}
 edge(e){const A=this.node(e.s),B=this.node(e.t),r=RELATIONS[e.r],lines=r==='hamming-1'?[1,2,3,4,5,6].filter(l=>((A.state^B.state)>>(l-1))&1):null;
  return {source:A.id,relation:r,target:B.id,symmetric:SYMMETRIC[r],weight:+e.w.toFixed(3),
   address:{gates:[A.gate,B.gate],states:[A.binary,B.binary],layerSlots:[A.layerSlot,B.layerSlot]},
   dimension:[A.dimension,B.dimension],transformation:{relation:r,flippedLines:lines},
   context:{contents:Object.entries(e.byContent).map(([id,n])=>({id,name:this.contents.get(id)?.name,interactions:n})),firstUnit:e.first,lastUnit:e.last},
   provenance:{rule:RULE_SOURCE[r],layers:'vendor/mesh/state-space/mesh-state-space.js',emergedFrom:'unit interaction, not a table'}}}
 neighbors(id,{relation=null,limit=25}={}){const out=[];for(const e of this.edges.values()){if(relation&&RELATIONS[e.r]!==relation)continue;if(this.nodeId(e.s)===id||this.nodeId(e.t)===id)out.push(e)}return out.sort((a,b)=>b.w-a.w).slice(0,limit).map(e=>this.edge(e))}
 degree(){const d=new Map();for(const e of this.edges.values()){d.set(e.s,(d.get(e.s)||0)+1);d.set(e.t,(d.get(e.t)||0)+1)}return d}
 // deterministic weighted label propagation: units cluster by the relations they found, edges keep their reasons
 clusters({iterations=12,minWeight=0}={}){const adj=new Map();for(const e of this.edges.values()){if(e.w<minWeight||e.r===0&&false)continue;(adj.get(e.s)||adj.set(e.s,[]).get(e.s)).push([e.t,e.w]);(adj.get(e.t)||adj.set(e.t,[]).get(e.t)).push([e.s,e.w])}
  const ids=[...adj.keys()].sort((a,b)=>a-b),label=new Map(ids.map(i=>[i,i]));
  for(let it=0;it<iterations;it++){let changed=0;for(const i of ids){const sc=new Map();for(const [j,w] of adj.get(i))sc.set(label.get(j),(sc.get(label.get(j))||0)+w);let best=label.get(i),bw=sc.get(best)||0;for(const [l,w] of sc)if(w>bw||w===bw&&l<best){best=l;bw=w}if(best!==label.get(i)){label.set(i,best);changed++}}if(!changed)break}
  const groups=new Map();for(const i of ids){const l=label.get(i);(groups.get(l)||groups.set(l,[]).get(l)).push(i)}
  return [...groups.values()].sort((a,b)=>b.length-a.length||a[0]-b[0]).map((m,ci)=>{const dimCount=Object.fromEntries(DIMENSIONS.map(d=>[d,0]));for(const i of m)dimCount[DIMENSIONS[i>>6]]++;const rel={};for(const e of this.edges.values())if(label.get(e.s)===label.get(m[0])&&label.get(e.t)===label.get(m[0]))rel[RELATIONS[e.r]]=(rel[RELATIONS[e.r]]||0)+1;
   return {cluster:ci,size:m.length,dimensions:dimCount,mixedDimension:Object.values(dimCount).filter(x=>x>0).length,relations:rel,sampleNodes:m.slice(0,6).map(i=>this.nodeId(i))}})}
 summary({clusterLimit=8}={}){const pot=potentialEdges(),byRel={},cross={within:0,cross:0},byDim=Object.fromEntries(DIMENSIONS.map(d=>[d,{nodes:0,visits:0}]));
  for(const e of this.edges.values()){const r=RELATIONS[e.r];const b=byRel[r]||(byRel[r]={edges:0,weight:0,potential:pot.byRelation[r].total});b.edges++;b.weight+=e.w;if((e.s>>6)===(e.t>>6))cross.within++;else cross.cross++}
  for(const [i,n] of this.nodes){const d=DIMENSIONS[i>>6];byDim[d].nodes++;byDim[d].visits+=n.visits}
  for(const r of Object.values(byRel))r.weight=+r.weight.toFixed(1);
  const deg=this.degree(),dv=[...deg.values()].sort((a,b)=>a-b),ws=[...this.edges.values()].map(e=>e.w).sort((a,b)=>a-b),med=ws.length?ws[ws.length>>1]:0,cl=this.clusters({minWeight:med});
  return {epoch:this.epoch,units:this.units,interactions:this.interactions,contents:[...this.contents.values()],nodes:this.nodes.size,nodesPossible:320,edges:this.edges.size,edgesPossible:pot.total,saturation:+(this.edges.size/pot.total).toFixed(4),
   byRelation:byRel,withinDimensionEdges:cross.within,crossDimensionEdges:cross.cross,byDimension:byDim,degree:dv.length?{min:dv[0],median:dv[dv.length>>1],max:dv[dv.length-1]}:null,
   clusters:{basis:'edges with weight >= the median edge weight ('+med.toFixed(2)+'); all-edge clustering collapses into one component on dense content',count:cl.length,mixedDimension:cl.filter(c=>c.mixedDimension>1).length,top:cl.slice(0,clusterLimit)},
   strongest:[...this.edges.values()].sort((a,b)=>b.w-a.w).slice(0,5).map(e=>this.edge(e)),labels:{derived:['interaction window','weight = interactions','decay/prune','label-propagation clusters'],existing:['mesh-state-space layers','DIMENSION_QUALITIES','state-space.mjs relations','CHANNELS','six-bit packing and (state+index)%5 field rule']}}}
}
export function verifyEdges(mesh){ // independent re-check with plain bit operations: every edge in the mesh is a true instance of its relation
 const bad=[];const rev=x=>{let r=0;for(let i=0;i<6;i++)if(x>>i&1)r|=1<<(5-i);return r},pc=x=>x.toString(2).split('1').length-1;
 const chan=new Set(CHANNELS.map(([a,b])=>gateToMacro(a)+','+gateToMacro(b)));
 for(const e of mesh.edges.values()){const a=e.s&63,b=e.t&63,r=RELATIONS[e.r];let ok;
  if(r==='identity')ok=a===b&&(e.s>>6)!==(e.t>>6);else if(r==='hamming-1')ok=pc(a^b)===1;else if(r==='complement')ok=(a^b)===63;else if(r==='reverse')ok=a!==b&&rev(a)===b;else if(r==='exchange-trigrams')ok=a!==b&&(((a&7)<<3)|(a>>3))===b;
  else if(r==='nuclear'){const bit=l=>a>>(l-1)&1;ok=a!==b&&[2,3,4,3,4,5].reduce((x,l,i)=>x|bit(l)<<i,0)===b}else if(r==='channel')ok=chan.has(a+','+b)||chan.has(b+','+a);
  if(!ok)bad.push({r,s:e.s,t:e.t})}return {checked:mesh.edges.size,bad}}
