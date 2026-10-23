/**
 * User-corrected 3–2–1: THREE binary positions at the lowest level,
 * TWO binary positions at the middle, ONE binary position at the highest.
 * EACH position is one binary (0/1), never a 2-bit/3-bit/6-bit grouping.
 * Separate source-backed Zhouyi six-line utilities live in zhouyi-six-line.mjs.
 *
 * A Human Design address contains nested coordinates. Bundling them into a
 * line/group does NOT generate an asserted 0/1 state; callers supply each bit
 * with evidence. No hash, gate parity, line parity, or threshold fallback.
 */
const copy=value=>structuredClone(value);
const freeze=value=>{
  if(value&&typeof value==='object'&&!Object.isFrozen(value)){
    for(const v of Object.values(value))freeze(v);
    Object.freeze(value);
  }
  return value;
};
const SHAPE=Object.freeze({level3:3,level2:2,level1:1});
export const BINARY_321_COUNTS=SHAPE;
/** Lowest-layer Human Design grouping as stated by the system's author.
 * Axis/axon/house wording remains explicitly provisional until clarified;
 * arcAxis is not automatically interchangeable with another address schema.
 */
export const HUMAN_LOWEST_LINES=freeze([
  {id:'color-tone-base',title:'Color / Tone / Base',keys:['color','tone','base'],status:'author-stated-grouping'},
  {id:'line',title:'Line',keys:['line'],status:'author-stated-grouping'},
  {id:'axis-house',title:'Axis / Axon and House',keys:['arcAxis','axis','axon','house'],status:'third-group-wording-needs-precision'},
]);
const bit=value=>value===0||value===1;
const validProvenance=v=>typeof v==='string'&&v.trim().length>0;

export function humanLowestLines(address={}){
  if(!address||typeof address!=='object'||Array.isArray(address))throw new TypeError('human address must be a record');
  return freeze(HUMAN_LOWEST_LINES.map(def=>({
    id:def.id,title:def.title,status:def.status,
    components:Object.fromEntries(def.keys.filter(k=>Object.hasOwn(address,k)).map(k=>[k,copy(address[k])])),
    note:'address components are grouped, but do not determine the binary bit by themselves',
  })));
}

/** levels: {level3:[{bit,provenance?} x3],level2:[...] x2,level1:[...] x1}.
 * Optional name/semantic label can only come from the caller; positions do not
 * inherit fake meaning from their cardinality or screen coordinates.
 */
export function projectThreeTwoOne({levels,provenance,human=null}={}){
  if(!validProvenance(provenance))throw new TypeError('3-2-1 requires explicit provenance');
  if(!levels||typeof levels!=='object'||Array.isArray(levels))
    throw new TypeError('3-2-1 requires explicit binary levels, not just a gate or six-line hexagram');
  const resolved={};
  for(const [level,count] of Object.entries(SHAPE)){
    const supplied=levels[level];
    if(!Array.isArray(supplied)||supplied.length!==count)
      throw new RangeError(level+' requires '+count+' separate binary positions');
    resolved[level]=supplied.map((source,index)=>{
      if(!source||typeof source!=='object'||!bit(source.bit))
        throw new TypeError(level+' '+(index+1)+' requires one explicitly supplied binary 0 or 1');
      if(source.provenance!=null&&!validProvenance(source.provenance))
        throw new TypeError('binary evidence must be an explicit provenance string');
      if(source.name!=null&&(typeof source.name!=='string'||!source.name.trim()))
        throw new TypeError('binary name must be an explicitly supplied nonempty string');
      const group=level==='level3'?HUMAN_LOWEST_LINES[index]:null;
      return {role:level,index:index+1,kind:'binary',bit:source.bit,
        name:source.name??null,sourceGroup:group?.id??null,
        provenance:source.provenance??provenance};
    });
  }
  return freeze({schema:'synthia.321.binary.v2',provenance,bottomToTop:['level3','level2','level1'],
    roleCounts:copy(SHAPE),levels:resolved,humanGroups:human==null?null:humanLowestLines(human),
    representation:'3 binary positions below 2 binary positions below 1 binary position; each value is one bit',
    semanticPolicy:'source meaning and binary assignments must be supplied, never inferred from a hexagram or address ordinal'});
}

/** Adds precisely six binary resident positions, preserving existing swarm edges. */
export function mountThreeTwoOne(unified,{levels,provenance,human=null,prefix='binary'}={}){
  const shape=projectThreeTwoOne({levels,provenance,human});
  if(typeof prefix!=='string'||!/^[A-Za-z][\w-]{0,36}$/.test(prefix))
    throw new TypeError('invalid binary namespace');
  const positions={level3:[[-.68,.46],[0,.46],[.68,.46]],level2:[[-.35,-.05],[.35,-.05]],level1:[[0,-.60]]};
  const projected=[];
  for(const level of shape.bottomToTop){
    for(const p of shape.levels[level]){
      const id=`${prefix}-${level}-${p.index}`;
      if(unified.swarm.nodes.has(id))throw new Error('3-2-1 formation already mounted: '+id);
      projected.push({id,label:`${p.name??(p.sourceGroup??level)} · ${p.bit}`,
        x:positions[level][p.index-1][0],y:positions[level][p.index-1][1],
        address:null,provenance:p.provenance});
    }
  }
  for(const item of projected)unified.add(item);
  return {shape,nodeIds:projected.map(x=>x.id),mountStatus:'six-single-bit-roles-mounted',semanticLinksAdded:0};
}

/** Change one 0/1 bit at a named level; never flip an arbitrary Fu Xi line. */
export function transitionThreeTwoOne(input,{level,index,toBit,provenance}={}){
  const before=projectThreeTwoOne(input);
  if(!Object.hasOwn(SHAPE,level))throw new RangeError('unknown 3-2-1 binary level');
  if(!Number.isInteger(index)||index<1||index>SHAPE[level])throw new RangeError('index outside binary level');
  if(!bit(toBit))throw new TypeError('transition requires binary toBit 0 or 1');
  const next=copy(before.levels);
  const fromBit=next[level][index-1].bit;
  next[level][index-1].bit=toBit;
  const after=projectThreeTwoOne({levels:next,provenance:provenance??before.provenance,
    human:input?.human??null});
  return freeze({schema:'synthia.321.binary-transition.v2',before,after,
    changed:fromBit!==toBit,at:{level,index},fromBit,toBit,
    policy:'one binary position changed; relationships and higher-level states are not silently recomputed'});
}
