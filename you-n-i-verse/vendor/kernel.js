(function(global){
'use strict';
const VERSION='0.7.0';
const PRIMITIVES=Object.freeze({
  BEING:Object.freeze({id:'BE',name:'Being',qualities:Object.freeze(['presence','state'])}),
  DESIGN:Object.freeze({id:'DE',name:'Design',qualities:Object.freeze(['relation','structure'])}),
  MOVEMENT:Object.freeze({id:'MO',name:'Movement',qualities:Object.freeze(['transition','transfer'])}),
  EVOLUTION:Object.freeze({id:'EV',name:'Evolution',qualities:Object.freeze(['retention','adaptation'])})
});
const DIMENSIONS=Object.freeze(['Movement','Evolution','Being','Design','Space']);
const PLANETARIES=Object.freeze(['Sun','Earth','Moon','North Node','South Node','Mercury','Venus','Mars','Jupiter','Saturn','Uranus','Neptune','Pluto']);
const ZODIACS=Object.freeze(['Aries','Taurus','Gemini','Cancer','Leo','Virgo','Libra','Scorpio','Sagittarius','Capricorn','Aquarius','Pisces']);
const LINE_NAMES=Object.freeze(['Foundation','Natural','Experimentation','Friendship','Projection','Transition']);
const COLOR_NAMES=Object.freeze(['Fear','Hope','Desire','Need','Guilt','Innocence']);
const TONE_NAMES=Object.freeze(['Security','Uncertainty','Action','Meditation','Judgement','Acceptance']);
const TONE_SENSES=Object.freeze(['Smell','Taste','Outer Vision','Inner Vision','Feeling','Touch']);
const BASE_NAMES=Object.freeze(['Base 1','Base 2','Base 3','Base 4','Base 5']);
const GATE_NAMES=Object.freeze([
'Creative / Self-Expression','Receptive / Direction of Self','Difficulty at Beginning / Ordering','Formulization','Fixed Rhythms','Friction','Role of Self in Interaction','Holding Together / Contribution','Focus','Behavior of Self','Ideas','Caution','Listener','Power Skills','Extremes','Skills','Opinions','Correction','Wanting','Now','Control','Openness','Assimilation','Rationalization','Innocence','Egoist','Caring','Game Player','Perseverance','Feelings','Leadership','Continuity','Privacy','Power','Change','Crisis','Friendship','Fighter','Provocation','Aloneness','Contraction','Growth','Insight','Alertness','Gatherer','Determination of Self','Realization','Depth','Principles','Values','Shock','Stillness','Beginnings','Ambition','Abundance','Stimulation','Intuition','Joy of Life','Sexuality','Limitation','Mystery','Detail','Doubt','Confusion']);
const TRIGRAMS=Object.freeze({
 Qian:Object.freeze({symbol:'☰',meaning:'Heaven',lines:Object.freeze([1,1,1])}), Kun:Object.freeze({symbol:'☷',meaning:'Earth',lines:Object.freeze([0,0,0])}),
 Zhen:Object.freeze({symbol:'☳',meaning:'Thunder',lines:Object.freeze([1,0,0])}), Xun:Object.freeze({symbol:'☴',meaning:'Wind',lines:Object.freeze([0,1,1])}),
 Kan:Object.freeze({symbol:'☵',meaning:'Water',lines:Object.freeze([0,1,0])}), Li:Object.freeze({symbol:'☲',meaning:'Fire',lines:Object.freeze([1,0,1])}),
 Gen:Object.freeze({symbol:'☶',meaning:'Mountain',lines:Object.freeze([0,0,1])}), Dui:Object.freeze({symbol:'☱',meaning:'Lake',lines:Object.freeze([1,1,0])})
});
const KW=Object.freeze([
[1,'Qian','Qian','Qian','The Creative'],[2,'Kun','Kun','Kun','The Receptive'],[3,'Zhen','Kan','Zhun','Difficulty at the Beginning'],[4,'Kan','Gen','Meng','Youthful Folly'],[5,'Qian','Kan','Xu','Waiting'],[6,'Kan','Qian','Song','Conflict'],[7,'Kun','Kan','Shi','The Army'],[8,'Kan','Kun','Bi','Holding Together'],[9,'Qian','Xun','Xiao Xu','Small Taming'],[10,'Dui','Qian','Lu','Treading'],[11,'Qian','Kun','Tai','Peace'],[12,'Kun','Qian','Pi','Standstill'],[13,'Li','Qian','Tong Ren','Fellowship'],[14,'Qian','Li','Da You','Great Possession'],[15,'Gen','Kun','Qian','Modesty'],[16,'Kun','Zhen','Yu','Enthusiasm'],[17,'Zhen','Dui','Sui','Following'],[18,'Xun','Gen','Gu','Work on the Decayed'],[19,'Dui','Kun','Lin','Approach'],[20,'Kun','Xun','Guan','Contemplation'],[21,'Zhen','Li','Shi He','Biting Through'],[22,'Li','Gen','Bi','Grace'],[23,'Kun','Gen','Bo','Splitting Apart'],[24,'Zhen','Kun','Fu','Return'],[25,'Zhen','Qian','Wu Wang','Innocence'],[26,'Qian','Gen','Da Xu','Great Taming'],[27,'Zhen','Gen','Yi','Nourishment'],[28,'Xun','Dui','Da Guo','Great Exceeding'],[29,'Kan','Kan','Kan','The Abysmal'],[30,'Li','Li','Li','The Clinging'],[31,'Gen','Dui','Xian','Influence'],[32,'Xun','Zhen','Heng','Duration'],[33,'Gen','Qian','Dun','Retreat'],[34,'Qian','Zhen','Da Zhuang','Great Power'],[35,'Kun','Li','Jin','Progress'],[36,'Li','Kun','Ming Yi','Darkening of the Light'],[37,'Li','Xun','Jia Ren','The Family'],[38,'Dui','Li','Kui','Opposition'],[39,'Gen','Kan','Jian','Obstruction'],[40,'Kan','Zhen','Jie','Deliverance'],[41,'Dui','Gen','Sun','Decrease'],[42,'Zhen','Xun','Yi','Increase'],[43,'Qian','Dui','Guai','Breakthrough'],[44,'Xun','Qian','Gou','Coming to Meet'],[45,'Kun','Dui','Cui','Gathering'],[46,'Xun','Kun','Sheng','Pushing Upward'],[47,'Kan','Dui','Kun','Oppression'],[48,'Xun','Kan','Jing','The Well'],[49,'Li','Dui','Ge','Revolution'],[50,'Xun','Li','Ding','The Cauldron'],[51,'Zhen','Zhen','Zhen','The Arousing'],[52,'Gen','Gen','Gen','Keeping Still'],[53,'Gen','Xun','Jian','Gradual Progress'],[54,'Dui','Zhen','Gui Mei','The Marrying Maiden'],[55,'Li','Zhen','Feng','Abundance'],[56,'Gen','Li','Lu','The Wanderer'],[57,'Xun','Xun','Xun','The Gentle'],[58,'Dui','Dui','Dui','The Joyous'],[59,'Kan','Xun','Huan','Dispersion'],[60,'Dui','Kan','Jie','Limitation'],[61,'Dui','Xun','Zhong Fu','Inner Truth'],[62,'Gen','Zhen','Xiao Guo','Small Exceeding'],[63,'Li','Kan','Ji Ji','After Completion'],[64,'Kan','Li','Wei Ji','Before Completion']]);
function buildKW(){const t=new Array(65); for(const [gate,ln,un,romanized,keyword] of KW){const lower=TRIGRAMS[ln],upper=TRIGRAMS[un],lines=[...lower.lines,...upper.lines],fuxi=lines.reduce((s,b,i)=>s+b*(1<<i),0);t[gate]=Object.freeze({gate,userName:GATE_NAMES[gate-1],romanized,keyword,lower:ln,upper:un,lines:Object.freeze(lines),bitStringBottomToTop:lines.join(''),fuxi,complementFuxi:fuxi^63});}return Object.freeze(t)}
const KING_WEN=buildKW();
const CHANNELS=Object.freeze([
['1-8',[1,8],'Inspiration'],['2-14',[2,14],'The Beat'],['3-60',[3,60],'Mutation'],['4-63',[4,63],'Logic'],['5-15',[5,15],'Rhythm'],['6-59',[6,59],'Mating'],['7-31',[7,31],'Alpha'],['9-52',[9,52],'Concentration'],['10-20',[10,20],'Awakening'],['10-34',[10,34],'Exploration'],['10-57',[10,57],'Perfected Form'],['11-56',[11,56],'Curiosity'],['12-22',[12,22],'Openness'],['13-33',[13,33],'Witness'],['16-48',[16,48],'Wavelength'],['17-62',[17,62],'Acceptance'],['18-58',[18,58],'Judgment'],['19-49',[19,49],'Synthesis'],['20-34',[20,34],'Charisma'],['20-57',[20,57],'Brainwave'],['21-45',[21,45],'Money Line'],['23-43',[23,43],'Structuring'],['24-61',[24,61],'Awareness'],['25-51',[25,51],'Initiation'],['26-44',[26,44],'Surrender'],['27-50',[27,50],'Preservation'],['28-38',[28,38],'Struggle'],['29-46',[29,46],'Discovery'],['30-41',[30,41],'Recognition'],['32-54',[32,54],'Transformation'],['34-57',[34,57],'Power'],['35-36',[35,36],'Transitoriness'],['37-40',[37,40],'Community'],['39-55',[39,55],'Emoting'],['42-53',[42,53],'Maturation'],['47-64',[47,64],'Abstraction']].map(x=>Object.freeze({id:x[0],gates:Object.freeze(x[1]),name:x[2]})));
const ADDRESS_FIELDS=Object.freeze(['planetary','dimension','gate','line','color','tone','base','degree','minute','second','arc','zodiac','house']);
function ri(name,v,min,max){v=Number(v);if(!Number.isInteger(v)||v<min||v>max)throw new RangeError(name+' must be '+min+'..'+max+'; got '+v);return v}
function rn(name,v,min,maxx){v=Number(v);if(!Number.isFinite(v)||v<min||v>=maxx)throw new RangeError(name+' must be >= '+min+' and < '+maxx+'; got '+v);return v}
function ne(name,v,vals){if(Number.isInteger(v)){ri(name,v,1,vals.length);return {index:v,name:vals[v-1]}}const i=vals.findIndex(x=>x.toLowerCase()===String(v).toLowerCase());if(i<0)throw new RangeError(name+' invalid');return {index:i+1,name:vals[i]}}
class FullAddress{
 constructor(raw){
  if(!raw||typeof raw!=='object')throw new TypeError('FullAddress requires object');
  const missing=ADDRESS_FIELDS.filter(k=>raw[k]===undefined||raw[k]===null||raw[k]==='');
  if(missing.length)throw new Error('INCOMPLETE ADDRESS: missing '+missing.join(', '));
  const planetary=ne('planetary',raw.planetary,PLANETARIES),dimension=ne('dimension',raw.dimension,DIMENSIONS),zodiac=ne('zodiac',raw.zodiac,ZODIACS);
  this.planetary=planetary.name;this.planetaryIndex=planetary.index;
  this.dimension=dimension.name;this.dimensionIndex=dimension.index;
  this.gate=ri('gate',raw.gate,1,64);this.line=ri('line',raw.line,1,6);
  this.color=ri('color',raw.color,1,6);this.tone=ri('tone',raw.tone,1,6);this.base=ri('base',raw.base,1,5);
  this.degree=ri('degree',raw.degree,0,29);this.minute=ri('minute',raw.minute,0,59);this.second=rn('second',raw.second,0,60);
  this.arc=ri('arc',raw.arc,0,99);
  this.zodiac=zodiac.name;this.zodiacIndex=zodiac.index;this.house=ri('house',raw.house,1,12);
  this.behavior=this.line;this.motivation=this.color;this.sense=this.tone;this.environment=this.base;
  Object.freeze(this)
 }
 toObject(){return {planetary:this.planetary,planetaryIndex:this.planetaryIndex,dimension:this.dimension,dimensionIndex:this.dimensionIndex,gate:this.gate,line:this.line,behavior:this.behavior,color:this.color,motivation:this.motivation,tone:this.tone,sense:this.sense,base:this.base,environment:this.environment,degree:this.degree,minute:this.minute,second:this.second,arc:this.arc,zodiac:this.zodiac,zodiacIndex:this.zodiacIndex,house:this.house}}
 toString(){const s=Number.isInteger(this.second)?String(this.second):this.second.toFixed(3).replace(/0+$/,'').replace(/\.$/,'');return `${this.planetary}>${this.dimension}>G${this.gate}.L${this.line}.C${this.color}.T${this.tone}.B${this.base}>${this.degree}°${this.minute}′${s}″>Arc${this.arc}>${this.zodiac}>H${this.house}`}
}
const ANGULAR=Object.freeze({HEXAGRAM_SPAN_SECONDS:20250,LINE_SPAN_SECONDS:3375,COLOR_SPAN_SECONDS:562.5,TONE_SPAN_SECONDS:93.75,BASE_SPAN_SECONDS:18.75});
function dmsWithinZodiacSeconds(d,m,s){return ri('degree',d,0,29)*3600+ri('minute',m,0,59)*60+rn('second',s,0,60)}
function fnv1a(str){let h=0x811c9dc5;for(let i=0;i<str.length;i++){h^=str.charCodeAt(i);h=Math.imul(h,0x01000193)>>>0}return h>>>0}
function granularSpecialty(a){a=a instanceof FullAddress?a:new FullAddress(a);const zs=dmsWithinZodiacSeconds(a.degree,a.minute,a.second),phase=(zs+a.arc/100)/(30*3600),code=fnv1a(`${a.degree}|${a.minute}|${a.second}|${a.arc}|${a.zodiacIndex}|${a.house}`);return Object.freeze({phase,code,microBias:code/0xffffffff})}
class XorShift32{constructor(seed){this.state=(seed>>>0)||0x6d2b79f5}nextUint(){let x=this.state>>>0;x^=x<<13;x^=x>>>17;x^=x<<5;this.state=x>>>0;return this.state}next(){return this.nextUint()/0x100000000}}
function clone(v){return JSON.parse(JSON.stringify(v))}
const FIELD_SCHEMA=Object.freeze({
  planetary:Object.freeze({count:13,values:PLANETARIES}),
  dimension:Object.freeze({count:5,values:DIMENSIONS}),
  gate:Object.freeze({count:64,min:1,max:64}),
  line:Object.freeze({count:6,min:1,max:6}),
  color:Object.freeze({count:6,min:1,max:6}),
  tone:Object.freeze({count:6,min:1,max:6}),
  base:Object.freeze({count:5,min:1,max:5}),
  degree:Object.freeze({count:30,min:0,max:29}),
  minute:Object.freeze({count:60,min:0,max:59}),
  second:Object.freeze({count:60,min:0,max:59}),
  arc:Object.freeze({count:100,min:0,max:99}),
  zodiac:Object.freeze({count:12,values:ZODIACS}),
  house:Object.freeze({count:12,min:1,max:12})
});
const EXACT_ADDRESS_CARDINALITY=5n*64n*13n*6n*6n*6n*5n*30n*60n*60n*100n*12n*12n;
const RESOLUTION_LEVELS=Object.freeze(['bit','binary','trinary','line','gate','filter','channel','dimension','full-address']);
function addressKey(a){a=a instanceof FullAddress?a:new FullAddress(a);return a.toString()}
function projectAddress(a,level){a=a instanceof FullAddress?a:new FullAddress(a);switch(level){
  case'bit':return `${a.planetary}|${a.dimension}|G${a.gate}|L${a.line}|C${a.color}|T${a.tone}|B${a.base}|${a.degree}:${a.minute}:${a.second}|A${a.arc}|${a.zodiac}|H${a.house}`;
  case'binary':return `${a.planetary}|${a.dimension}|G${a.gate}|L${a.line}|binary:${KING_WEN[a.gate].lines[a.line-1]}|C${a.color}|T${a.tone}|B${a.base}`;
  case'trinary':return `${a.planetary}|${a.dimension}|G${a.gate}|tri:${Math.ceil(a.line/2)}|C${a.color}|T${a.tone}|B${a.base}`;
  case'line':return `${a.planetary}|${a.dimension}|G${a.gate}|L${a.line}`;
  case'gate':return `${a.planetary}|${a.dimension}|G${a.gate}`;
  case'filter':return `${a.planetary}|${a.dimension}|G${a.gate}`;
  case'channel':return `${a.planetary}|${a.dimension}|G${a.gate}`;
  case'dimension':return `${a.dimension}`;
  case'full-address':return addressKey(a);
  default:throw new RangeError('bad scale '+level)
}}
class AddressedBitState{
 constructor(address){this.address=address instanceof FullAddress?address:new FullAddress(address);this.initialValue=KING_WEN[this.address.gate].lines[this.address.line-1];this.value=this.initialValue;this.charge=0;this.tension=0;this.metastable=false;this.phase=0;this.pressures=0;this.lastTick=0}
 snapshot(){return {fullAddress:this.address.toObject(),fullAddressString:this.address.toString(),initialValue:this.initialValue,value:this.value,charge:this.charge,tension:this.tension,metastable:this.metastable,phase:this.phase,pressures:this.pressures,lastTick:this.lastTick}}
}
class PrimitiveKernel{
 being(bit){return Object.freeze({primitive:'Being',exists:true,value:bit.value})}
 design(bit,rels){return Object.freeze({primitive:'Design',fullAddress:bit.address.toObject(),relationCount:(rels||[]).length})}
 movement(bit,p){p=Math.max(0,Math.min(1,p));return Object.freeze({primitive:'Movement',mayPrime:p>0,mayChange:bit.metastable||(bit.charge+p>=.75),target:bit.value^1})}
 evolution(a,b){return Object.freeze({primitive:'Evolution',retained:a.value!==b.value,from:a.value,to:b.value,address:b.fullAddress})}
}
class EventLedger{constructor(){this.events=[];this.head='GENESIS'}append(event){const body=clone(Object.assign({},event,{previousHash:this.head}));body.hash=fnv1a(JSON.stringify(body)).toString(16).padStart(8,'0');this.events.push(Object.freeze(body));this.head=body.hash;return body}snapshot(){return this.events.map(clone)}}
class AgentomagotchiField{
 constructor(opts){opts=opts||{};this.seed=(opts.seed===undefined?0xA60E2026:opts.seed)>>>0;this.rng=new XorShift32(this.seed);this.kernel=new PrimitiveKernel();this.ledger=new EventLedger();this.tick=0;this.points=new Map();this.relations=[];this.relationIndex=new Map();this.assemblies=new Map()}
 get potentialNodeViews(){return 5*64*13}
 get potentialExactAddresses(){return EXACT_ADDRESS_CARDINALITY}
 get potentialExactAddressesString(){return EXACT_ADDRESS_CARDINALITY.toString()}
 get materializedNodes(){return this.points.size}
 get addressedPoints(){return this.points.size}
 get dimensions(){return {size:DIMENSIONS.length}}
 getPoint(aLike){const a=aLike instanceof FullAddress?aLike:new FullAddress(aLike),k=addressKey(a);let p=this.points.get(k);if(!p){p=new AddressedBitState(a);this.points.set(k,p)}return p}
 peekPoint(aLike){const a=aLike instanceof FullAddress?aLike:new FullAddress(aLike);return this.points.get(addressKey(a))||null}
 resolveAddress(aLike){const a=aLike instanceof FullAddress?aLike:new FullAddress(aLike),bit=this.getPoint(a),gate=KING_WEN[a.gate];return Object.freeze({address:a,point:bit,node:bit,bit,gate,specialty:granularSpecialty(a),qualities:Object.freeze({behavior:{index:a.line,name:LINE_NAMES[a.line-1]},motivation:{index:a.color,name:COLOR_NAMES[a.color-1]},sense:{index:a.tone,tone:TONE_NAMES[a.tone-1],name:TONE_SENSES[a.tone-1]},base:{index:a.base,name:BASE_NAMES[a.base-1]}})})}
 getNode(d,g){d=ne('dimension',d,DIMENSIONS).name;g=ri('gate',g,1,64);const matches=[];for(const p of this.points.values())if(p.address.dimension===d&&p.address.gate===g)matches.push(p);return {dimension:d,gate:g,hexagram:KING_WEN[g],bits:matches,bitString:()=>KING_WEN[g].bitStringBottomToTop,snapshot:()=>({dimension:d,gate:g,canonicalBits:KING_WEN[g].lines.slice(),addressedPoints:matches.map(x=>x.snapshot())})}}
 relationKey(a,b,scale,type='relates'){const ap=projectAddress(a,scale),bp=projectAddress(b,scale);return `${scale}|${type}|${ap}→${bp}`}
 relationsFor(aLike,scale){const a=aLike instanceof FullAddress?aLike:new FullAddress(aLike);const full=addressKey(a);return this.relations.filter(r=>{if(scale&&r.scale!==scale)return false;return r.evidence.some(e=>e.aFull===full||e.bFull===full)||r.aFull===full||r.bFull===full})}
 dimensionActivity(d){d=ne('dimension',d,DIMENSIONS).name;let charge=0,tension=0,count=0;for(const p of this.points.values())if(p.address.dimension===d){charge+=p.charge;tension+=p.tension;count++}return {dimension:d,charge,tension,count,mean:count?Math.min(1,(charge+tension*.5)/count):0}}
 legalCandidates(bit,p,s){const mv=this.kernel.movement(bit,p),c=[{id:'remain',weight:Math.max(.05,1-bit.charge-p*.35)}];if(mv.mayPrime)c.push({id:'prime',weight:.25+p+s.microBias*.15});if(!bit.metastable&&mv.mayChange)c.push({id:'enter-changing',weight:.25+bit.charge+p});if(bit.metastable&&bit.tension+p>=.9)c.push({id:'flip',weight:.2+bit.tension+p});return c}
 weighted(cands){const total=cands.reduce((s,c)=>s+c.weight,0),roll=this.rng.next();let cur=roll*total,chosen=cands[cands.length-1];for(const c of cands){cur-=c.weight;if(cur<=0){chosen=c;break}}return {chosen,roll,total}}
 relationSignal(rel){return Math.max(0,Math.min(1,rel.charge*.55+rel.coherence*.3+Math.min(1,rel.support/8)*.15))}
 reinforceRelation(rel,intensity=.2,evidence=null){
  const p=Math.max(0,Math.min(1,Number(intensity)||0));
  rel.support+=1;rel.charge=Math.min(1,rel.charge+p*(.42+rel.coherence*.18));rel.coherence=Math.min(1,rel.coherence+p*.22);
  rel.active=rel.charge>=rel.threshold;rel.lastTick=this.tick;
  if(evidence)rel.evidence.push(clone(evidence));
  if(rel.evidence.length>24)rel.evidence.shift();
  if(rel.active&&!rel.activatedAt)rel.activatedAt=this.tick;
  return rel
 }
 stimulateRelations(aLike,intensity=.15,reason='pressure'){
  const a=aLike instanceof FullAddress?aLike:new FullAddress(aLike),changed=[];
  for(const rel of this.relations){
    const p=projectAddress(a,rel.scale);
    if(rel.aProjection===p||rel.bProjection===p){
      this.reinforceRelation(rel,intensity,{tick:this.tick,reason,fullAddress:a.toObject(),fullAddressString:a.toString()});
      const otherFull=rel.aProjection===p?rel.bFull:rel.aFull;
      const other=this.points.get(otherFull);
      if(other){const signal=this.relationSignal(rel)*intensity;other.charge=Math.min(1,other.charge+signal*.18);other.tension=Math.min(1,other.tension+signal*.07)}
      changed.push(rel)
    }
  }
  return changed
 }
 applyPressure(addressLike,intensity,source){
  const r=this.resolveAddress(addressLike),a=r.address,bit=r.bit,s=r.specialty,p=Math.max(0,Math.min(1,Number(intensity===undefined?.25:intensity))),bb=bit.snapshot(),
    pr={being:this.kernel.being(bit),design:this.kernel.design(bit,this.relationsFor(a)),movement:this.kernel.movement(bit,p)},
    cands=this.legalCandidates(bit,p,s),res=this.weighted(cands);
  switch(res.chosen.id){case'prime':bit.charge=Math.min(1,bit.charge+p*(.45+s.microBias*.2));bit.tension=Math.min(1,bit.tension+p*.25);break;case'enter-changing':bit.metastable=true;bit.tension=Math.min(1,bit.tension+.35+p*.35);bit.charge=Math.min(1,bit.charge+p*.2);break;case'flip':bit.value^=1;bit.metastable=false;bit.tension=0;bit.charge*=.35;bit.phase=(bit.phase+.5)%1;break;default:bit.charge=Math.max(0,bit.charge-.015);bit.tension=Math.max(0,bit.tension-.01)}
  bit.pressures++;bit.lastTick=++this.tick;
  const propagated=this.stimulateRelations(a,p*.65,'propagated:'+String(source||'world'));
  const ba=bit.snapshot(),evo=this.kernel.evolution(bb,ba);
  return this.ledger.append({tick:this.tick,law:'pre-probabilistic -> probabilistic resolution -> post-deterministic history',source:source||'world',fullAddress:a.toObject(),fullAddressString:a.toString(),addressPointKey:addressKey(a),primitiveRead:pr,specialty:s,candidates:clone(cands),probabilityRoll:res.roll,selected:res.chosen.id,pointBefore:bb,pointAfter:ba,evolution:evo,propagatedRelations:propagated.map(x=>x.id)})
 }
 connect(aLike,bLike,scale='bit',metadata={}){
  if(!RESOLUTION_LEVELS.includes(scale))throw new RangeError('bad scale');
  const a=aLike instanceof FullAddress?aLike:new FullAddress(aLike),b=bLike instanceof FullAddress?bLike:new FullAddress(bLike);
  if(scale==='channel'){const key=[a.gate,b.gate].sort((x,y)=>x-y).join('-');if(!CHANNELS.some(c=>c.id===key))throw new Error('No canonical channel '+key)}
  const type=String(metadata.type||'relates'),key=this.relationKey(a,b,scale,type);
  let rel=this.relationIndex.get(key);
  const ev={tick:this.tick,aFull:addressKey(a),bFull:addressKey(b),a:a.toObject(),b:b.toObject(),metadata:clone(metadata)};
  if(rel){this.reinforceRelation(rel,metadata.intensity===undefined?.22:metadata.intensity,ev)}
  else{
    rel={id:'R'+(this.relations.length+1),scale,type,a:a.toObject(),b:b.toObject(),aFull:addressKey(a),bFull:addressKey(b),aProjection:projectAddress(a,scale),bProjection:projectAddress(b,scale),charge:0,coherence:.05,support:0,counter:0,threshold:Number(metadata.threshold||.66),active:false,activatedAt:null,lastTick:this.tick,evidence:[]};
    this.relations.push(rel);this.relationIndex.set(key,rel);this.reinforceRelation(rel,metadata.intensity===undefined?.24:metadata.intensity,ev)
  }
  this.tick++;
  this.ledger.append({tick:this.tick,type:'connection-observation',scale,a:a.toString(),b:b.toString(),aProjection:rel.aProjection,bProjection:rel.bProjection,relationId:rel.id,charge:rel.charge,coherence:rel.coherence,support:rel.support,active:rel.active});
  return rel
 }
 counterobserve(relId,reason='counterevidence'){
  const rel=this.relations.find(r=>r.id===relId);if(!rel)return false;
  rel.counter++;rel.charge=Math.max(0,rel.charge-.14);rel.coherence=Math.max(0,rel.coherence-.09);rel.active=rel.charge>=rel.threshold;rel.lastTick=++this.tick;
  rel.evidence.push({tick:this.tick,kind:'counterevidence',reason});
  this.ledger.append({tick:this.tick,type:'connection-counterevidence',relationId:rel.id,reason,charge:rel.charge,coherence:rel.coherence,active:rel.active});
  return true
 }
 relationSnapshot(){return this.relations.map(r=>clone(r))}
 activeRelations(){return this.relations.filter(r=>r.active)}
 relationSummary(){const active=this.activeRelations(),top=[...this.relations].sort((a,b)=>this.relationSignal(b)-this.relationSignal(a)).slice(0,8);return {total:this.relations.length,active:active.length,top:top.map(r=>({id:r.id,scale:r.scale,charge:r.charge,coherence:r.coherence,support:r.support,active:r.active,aProjection:r.aProjection,bProjection:r.bProjection,signal:this.relationSignal(r)}))}}
 snapshot(){return {version:'0.7.0',seed:this.seed,tick:this.tick,schema:FIELD_SCHEMA,potentialExactAddresses:this.potentialExactAddressesString,addressedPoints:[...this.points.values()].map(p=>p.snapshot()),relations:this.relationSnapshot(),ledger:this.ledger.snapshot()}}
}
const AgentomagotchiMesh=AgentomagotchiField;
function makeFullAddress(o){return new FullAddress(Object.assign({planetary:'Sun',dimension:'Movement',gate:1,line:1,color:1,tone:1,base:1,degree:0,minute:0,second:0,arc:0,zodiac:'Aries',house:1},o||{}))}
global.AgentomagotchiKernel=Object.freeze({VERSION:'0.6.0',PRIMITIVES,DIMENSIONS,PLANETARIES,ZODIACS,LINE_NAMES,COLOR_NAMES,TONE_NAMES,TONE_SENSES,BASE_NAMES,GATE_NAMES,KING_WEN,CHANNELS,ADDRESS_FIELDS,RESOLUTION_LEVELS,ANGULAR,FIELD_SCHEMA,EXACT_ADDRESS_CARDINALITY,FullAddress,AddressedBitState,AgentomagotchiField,AgentomagotchiMesh,EventLedger,makeFullAddress,granularSpecialty,dmsWithinZodiacSeconds,addressKey,projectAddress,XorShift32});
})(typeof window!=='undefined'?window:globalThis);
