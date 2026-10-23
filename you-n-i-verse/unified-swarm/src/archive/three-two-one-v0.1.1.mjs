/**
 * Structural 3–2–1 projection of the SAME six-line state.
 * 3 bigram views (2 bits each); 2 trigram views (3 bits each);
 * 1 hexagram view (6 bits).  This is NOT six independently assigned
 * semantic identities, and no 3->2 correspondence law is assumed.
 * A human/owner can supply source-supported semantic relationships later.
 * Bit index 0 = BOTTOM line, consistent with the preserved ZhouyiState.
 */
import {assertSix,lines,changeLine} from '../vendor/pass05/zhouyi-state.mjs';
import {sixForGate,gateForSix} from '../vendor/pass05/king-wen-fuxi-map.mjs';
const deepFreeze=x=>{if(x&&typeof x==='object'&&!Object.isFrozen(x)){Object.values(x).forEach(deepFreeze);Object.freeze(x);}return x;};

export function projectThreeTwoOne({gate=null,six=null,provenance}={}){
  if(typeof provenance!=='string'||!provenance.trim())throw new TypeError('3-2-1 requires explicit provenance');
  if(gate===null&&six===null)throw new TypeError('3-2-1 requires an observed gate or explicit six-line value');
  if(gate!==null&&(!Number.isInteger(gate)||gate<1||gate>64))throw new RangeError('gate must be 1..64');
  if(six!==null)assertSix(six);
  const resolvedSix=gate===null?six:sixForGate(gate);
  if(six!==null&&six!==resolvedSix)throw new RangeError('gate conflicts with supplied six-line state');
  const resolvedGate=gate??gateForSix(resolvedSix),all=lines(resolvedSix);
  const bigint2=[0,2,4].map((start,index)=>({role:'level3',index:index+1,kind:'bigram',lineNumbers:[start+1,start+2],bits:all.slice(start,start+2),value:all[start]+2*all[start+1]}));
  const trigram=[0,3].map((start,index)=>({role:'level2',index:index+1,kind:'trigram',lineNumbers:[start+1,start+2,start+3],bits:all.slice(start,start+3),value:all[start]+2*all[start+1]+4*all[start+2]}));
  const whole={role:'level1',index:1,kind:'hexagram',lineNumbers:[1,2,3,4,5,6],bits:[...all],value:resolvedSix};
  return deepFreeze({schema:'synthia.321.structural.v1',gate:resolvedGate,six:resolvedSix,bottomFirst:true,provenance,
    roleCounts:{level3:3,level2:2,level1:1},
    sixLines:all,levels:{level3:bigint2,level2:trigram,level1:[whole]},
    invariant:'three bigram/two trigram/one hexagram views share the SAME six source lines',
    semanticRelationPolicy:'no unsourced equivalence, synthesis, meaning, or external genotype inferred'});
}

/** Mount all six projection roles into the sourced ConsciousnessSwarm as actual
 * resident state-space units. Coordinates only affect the drawn phenotype.
 * No semantic edges are synthesized by mounting. */
export function mountThreeTwoOne(unified,{gate,six,provenance,prefix='hex'}={}){
  const shape=projectThreeTwoOne({gate,six,provenance});
  if(typeof prefix!=='string'||!/^[A-Za-z][\w-]{0,36}$/.test(prefix))throw new TypeError('invalid 3-2-1 namespace');
  const positions={level3:[[-.68,.46],[0,.46],[.68,.46]],level2:[[-.35,-.05],[.35,-.05]],level1:[[0,-.60]]};
  const projected=[];
  for(const level of ['level3','level2','level1']){
    for(const p of shape.levels[level]){
      const id=`${prefix}-${shape.gate}-${level}-${p.index}`;
      if(unified.swarm.nodes.has(id))throw new Error('3-2-1 formation already mounted: '+id);
      projected.push({id,label:`${p.kind} ${p.index} · ${p.bits.join('')}`,
        x:positions[level][p.index-1][0],y:positions[level][p.index-1][1],
        address:null,provenance});
    }
  }
  for(const item of projected)unified.add(item);
  return {shape,nodeIds:projected.map(p=>p.id),mountStatus:'six-resident-roles-mounted',
    semanticLinksAdded:0};
}

/** Recompute all 3/2/1 views after a source-executable moving-line change.
 * This changes a real six-line state; it is not a visual-only tween. */
export function transitionThreeTwoOne(input,{line,provenance}={}){
  const before=projectThreeTwoOne(input);
  const after=projectThreeTwoOne({six:changeLine(before.six,line),provenance:provenance??before.provenance});
  return deepFreeze({schema:'synthia.321.transition.v1',action:'change-line',line,
    before,after,changed:{level3:Math.floor((line-1)/2)+1,level2:Math.floor((line-1)/3)+1,level1:1},
    semanticClaim:'structural six-line change only; meaning requires positional authority'});
}
