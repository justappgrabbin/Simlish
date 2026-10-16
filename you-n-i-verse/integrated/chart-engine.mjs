// Birth chart engine for the Human Design + Astrology guide. Pure JavaScript, no backend, no network.
// EXISTING (hers, vendored unchanged under vendor/isohuman): calculateGeonatalChart + humanDesignResolution (SynthAi donors/isohuman) and the MIT astronomy-engine it uses.
// ported from her field_core.js: levelBits / hammingDistance / channelEmerges (gate/line and color/tone/base levels; the arcsecond level needs an arc-within-gate value and is not used here).
// STANDARD (not from her sources, labelled in every output): the 36-channel table, gate->center map used for definition, Type/Authority/Profile/Definition rules, aspect angles+orbs, relationship terms.
import {calculateGeonatalChart,humanDesignResolution} from '../vendor/isohuman/GeonatalCalculator.browser.mjs';
import {SiderealTime} from '../vendor/isohuman/astronomy-engine.mjs';
export const STANDARD='standard Human Design/astrology convention, not from her sources — confirm';
export const SIGNS=['Aries','Taurus','Gemini','Cancer','Leo','Virgo','Libra','Scorpio','Sagittarius','Capricorn','Aquarius','Pisces'];
export const BODIES=['Sun','Earth','North Node','South Node','Moon','Mercury','Venus','Mars','Jupiter','Saturn','Uranus','Neptune','Pluto']; // the donor's 13, in its order
export const CHANNELS=Object.freeze([[1,8],[2,14],[3,60],[4,63],[5,15],[6,59],[7,31],[9,52],[10,20],[10,34],[10,57],[11,56],[12,22],[13,33],[16,48],[17,62],[18,58],[19,49],[20,34],[20,57],[21,45],[23,43],[24,61],[25,51],[26,44],[27,50],[28,38],[29,46],[30,41],[32,54],[34,57],[35,36],[37,40],[39,55],[42,53],[47,64]].map(c=>Object.freeze(c)));
export const CENTER_OF_GATE=Object.freeze((()=>{const m={};const put=(c,g)=>g.forEach(x=>m[x]=c);
 put('Head',[64,61,63]);put('Ajna',[47,24,4,17,43,11]);put('Throat',[62,23,56,35,12,45,33,8,31,20,16]);put('G',[1,13,25,46,2,15,10,7]);put('Heart',[21,40,26,51]);
 put('Spleen',[48,57,44,50,32,28,18]);put('Solar Plexus',[6,37,22,36,30,55,49]);put('Sacral',[5,14,29,59,9,3,42,27,34]);put('Root',[53,60,52,19,39,41,58,38,54]);return m})());
export const CENTERS=['Head','Ajna','Throat','G','Heart','Spleen','Solar Plexus','Sacral','Root'];
const MOTORS=['Sacral','Heart','Solar Plexus','Root'];
const norm=x=>((x%360)+360)%360;
// ---------- time ----------
export function tzOffsetMinutes(timeZone,utcMs){const p=Object.fromEntries(new Intl.DateTimeFormat('en-US',{timeZone,hourCycle:'h23',year:'numeric',month:'numeric',day:'numeric',hour:'numeric',minute:'numeric',second:'numeric'}).formatToParts(new Date(utcMs)).map(x=>[x.type,x.value]));
 return (Date.UTC(+p.year,+p.month-1,+p.day,+p.hour,+p.minute,+p.second)-Math.floor(utcMs/1000)*1000)/60000}
export function birthToUtc({date,time='12:00',timeZone=null,utcOffsetMinutes=null}){
 const d=/^(\d{4})-(\d{1,2})-(\d{1,2})$/.exec(String(date)),t=/^(\d{1,2}):(\d{2})(?::(\d{2}))?$/.exec(String(time));
 if(!d)throw new RangeError('date must be YYYY-MM-DD');if(!t)throw new RangeError('time must be HH:MM');
 const local=Date.UTC(+d[1],+d[2]-1,+d[3],+t[1],+t[2],+(t[3]||0));if(+t[1]>23||+t[2]>59)throw new RangeError('time out of range');
 if(utcOffsetMinutes!==null)return new Date(local-utcOffsetMinutes*60000);
 if(!timeZone)throw new RangeError('timeZone (IANA name) or utcOffsetMinutes is required');
 let off=tzOffsetMinutes(timeZone,local);off=tzOffsetMinutes(timeZone,local-off*60000);return new Date(local-off*60000)}
// ---------- positions ----------
export function dms(lon){const l=norm(lon),sign=Math.floor(l/30),w=l-sign*30,deg=Math.floor(w),m=(w-deg)*60,min=Math.floor(m),sec=Math.floor((m-min)*60);return {sign:SIGNS[sign],signIndex:sign+1,deg,min,sec,text:`${deg}°${String(min).padStart(2,'0')}'${String(sec).padStart(2,'0')}" ${SIGNS[sign]}`}}
// DEFECT in her donor (vendor/isohuman, unexported equalHouseAscendant): its atan2(-cos θ, sin ε·tan φ + cos ε·sin θ) returns the DESCENDANT (180° off the true ascendant),
// so the donor's own house numbers are shifted by 6. The original is kept unchanged; here the same formula is used with +180° and houses are recomputed from it.
// Verified against Free Human Design's ascendant in the tests.
export function ascendant(date,{latitude,longitude}){const lst=norm(SiderealTime(date)*15+longitude),r=Math.PI/180,th=lst*r,la=latitude*r,ob=23.4392911*r;return norm(Math.atan2(-Math.cos(th),Math.sin(ob)*Math.tan(la)+Math.cos(ob)*Math.sin(th))/r+180)}
export const houseOf=(lon,asc)=>Math.floor(norm(lon-asc)/30)+1;
export const substructure=lon=>{const r=humanDesignResolution(lon);return {...r,ids:{gate:'gate.'+r.gate,line:`gate.${r.gate}.line.${r.line}`,color:'color.'+r.color,tone:'tone.'+r.tone,base:'base.'+r.base}}};
// ---------- standard HD derivation (flagged) ----------
export function definedChannels(gates){const s=new Set(gates);return CHANNELS.filter(([a,b])=>s.has(a)&&s.has(b))}
export function derive(gates){
 const ch=definedChannels(gates),def=new Set();for(const [a,b] of ch){def.add(CENTER_OF_GATE[a]);def.add(CENTER_OF_GATE[b])}
 const adj=Object.fromEntries(CENTERS.map(c=>[c,new Set()]));for(const [a,b] of ch){adj[CENTER_OF_GATE[a]].add(CENTER_OF_GATE[b]);adj[CENTER_OF_GATE[b]].add(CENTER_OF_GATE[a])}
 const comp=start=>{const seen=new Set([start]),q=[start];while(q.length){const c=q.pop();for(const n of adj[c])if(!seen.has(n)){seen.add(n);q.push(n)}}return seen};
 const comps=[];const left=new Set(def);while(left.size){const c=comp([...left][0]);comps.push([...c]);for(const x of c)left.delete(x)}
 const throat=def.has('Throat')?comp('Throat'):new Set(),motorToThroat=MOTORS.some(m=>throat.has(m))/* any motor (Sacral, Heart, Solar Plexus, Root) connected to the Throat */,sacral=def.has('Sacral');
 const type=!def.size?'Reflector':sacral?(motorToThroat?'Manifesting Generator':'Generator'):motorToThroat?'Manifestor':'Projector';
 const heartThroat=throat.has('Heart'),gThroat=throat.has('G');
 const authority=!def.size?'Lunar':def.has('Solar Plexus')?'Emotional':sacral?'Sacral':def.has('Spleen')?'Splenic':def.has('Heart')?(heartThroat?'Ego Manifested':'Ego Projected'):def.has('G')?'Self-Projected':'Mental (none inside)';
 const definition=!def.size?'none':({1:'single',2:'split',3:'triple split',4:'quadruple split'}[comps.length]||comps.length+' parts');
 return {type,authority,definition,channels:ch,definedCenters:CENTERS.filter(c=>def.has(c)),openCenters:CENTERS.filter(c=>!def.has(c)),components:comps,rules:STANDARD}}
// ---------- aspects (conventional angles/orbs, flagged) ----------
export const ASPECTS=Object.freeze([{name:'conjunction',angle:0,orb:8},{name:'sextile',angle:60,orb:6},{name:'square',angle:90,orb:8},{name:'trine',angle:120,orb:8},{name:'opposition',angle:180,orb:8}]);
export function aspects(placements){const out=[],list=placements.filter(p=>!['Earth','South Node'].includes(p.body));
 for(let i=0;i<list.length;i++)for(let j=i+1;j<list.length;j++){let d=Math.abs(list[i].longitude-list[j].longitude);d=d>180?360-d:d;for(const a of ASPECTS){const o=Math.abs(d-a.angle);if(o<=a.orb)out.push({a:list[i].body,b:list[j].body,aspect:a.name,orb:+o.toFixed(3)})}}
 return out.sort((x,y)=>x.orb-y.orb)}
// ---------- her field_core.js emergence rule (gate/line and color/tone/base levels) ----------
const levelBits=(g,l,c,t,b)=>({gateLine:((g-1)&63)<<3|((l-1)&7),colorToneBase:((c-1)&7)<<6|((t-1)&7)<<3|((b-1)&7)});
export const hammingDistance=(a,b)=>{let x=a^b,n=0;while(x){n+=x&1;x>>=1}return n};
export function channelEmerges(A,B){const a=levelBits(A.gate,A.line,A.color,A.tone,A.base),b=levelBits(B.gate,B.line,B.color,B.tone,B.base),gateLine=hammingDistance(a.gateLine,b.gateLine)===1,colorToneBase=hammingDistance(a.colorToneBase,b.colorToneBase)===1;return {any:gateLine||colorToneBase,gateLine,colorToneBase}}
// ---------- the chart ----------
export function calculateBirthChart({name='',date,time='12:00',timeZone=null,utcOffsetMinutes=null,latitude,longitude}){
 const lat=Number(latitude),lon=Number(longitude);if(!Number.isFinite(lat)||!Number.isFinite(lon))throw new RangeError('latitude and longitude are required (decimal degrees)');
 const utc=birthToUtc({date,time,timeZone,utcOffsetMinutes}),loc={latitude:lat,longitude:lon},c=calculateGeonatalChart(utc,loc);
 const trop=c.placements.filter(p=>p.frame==='tropical');
 const asc0=ascendant(utc,loc);const row=p=>({orientation:p.orientation,body:p.planetary,longitude:p.longitude,...dms(p.longitude),house:houseOf(p.longitude,asc0),gate:p.gate,line:p.line,color:p.color,tone:p.tone,base:p.base,ids:{gate:'gate.'+p.gate,line:`gate.${p.gate}.line.${p.line}`,color:'color.'+p.color,tone:'tone.'+p.tone,base:'base.'+p.base,planet:'planet.'+({'North Node':'NNode','South Node':'SNode'}[p.planetary]||p.planetary)}});
 const activations=trop.map(row),personality=activations.filter(a=>a.orientation==='personality'),design=activations.filter(a=>a.orientation==='design');
 const gates=[...new Set(activations.map(a=>a.gate))].sort((x,y)=>x-y),sun=o=>activations.find(a=>a.orientation===o&&a.body==='Sun');
 const hd={...derive(gates),profile:`${sun('personality').line}/${sun('design').line}`,personalityGates:[...new Set(personality.map(a=>a.gate))].sort((x,y)=>x-y),designGates:[...new Set(design.map(a=>a.gate))].sort((x,y)=>x-y),gates,
  both:[...new Set(personality.map(a=>a.gate))].filter(g=>design.some(d=>d.gate===g)).sort((x,y)=>x-y)};
 const asc=asc0,frames=f=>c.placements.filter(p=>p.frame===f&&p.orientation==='personality').map(p=>({body:p.planetary,longitude:p.longitude,...dms(p.longitude)}));
 const natal=personality.map(a=>({body:a.body,longitude:a.longitude,sign:a.sign,deg:a.deg,min:a.min,sec:a.sec,text:a.text,house:a.house}));
 return {name:String(name||''),input:{date,time,timeZone,utcOffsetMinutes,latitude:lat,longitude:lon},utc:utc.toISOString(),designUtc:c.designTimestamp,
  hd,activations,
  astrology:{ascendant:{longitude:asc,...dms(asc)},houseMethod:'equal houses from the ascendant (the donor\'s method, with its 180° ascendant defect corrected); Placidus is not implemented',natal,aspects:aspects(natal),sidereal:frames('sidereal_fagan_bradley'),draconic:frames('draconic_tropical_true'),trueNodeLongitude:c.trueNodeLongitude,faganBradleyAyanamsa:c.faganBradleyAyanamsa},
  methods:{...c.methods,wheel:'gate 41 starts at 302° (2°00\' Aquarius): the donor\'s wheel',derivation:STANDARD}}}
// ---------- two people ----------
export function compareCharts(A,B){
 const ga=new Set(A.hd.gates),gb=new Set(B.hd.gates),fa=new Set(A.hd.channels.map(c=>c.join('-'))),fb=new Set(B.hd.channels.map(c=>c.join('-')));
 const rel=[];for(const [x,y] of CHANNELS){const k=x+'-'+y,aF=fa.has(k),bF=fb.has(k),aHas=ga.has(x)||ga.has(y),bHas=gb.has(x)||gb.has(y);
  if(aF&&bF)rel.push({channel:k,kind:'companionship'});
  else if(aF&&!bF&&(gb.has(x)||gb.has(y)))rel.push({channel:k,kind:'compromise',holder:'A'});
  else if(bF&&!aF&&(ga.has(x)||ga.has(y)))rel.push({channel:k,kind:'compromise',holder:'B'});
  else if(aF&&!bHas)rel.push({channel:k,kind:'dominance',holder:'A'});
  else if(bF&&!aHas)rel.push({channel:k,kind:'dominance',holder:'B'});
  else if(!aF&&!bF&&((ga.has(x)&&gb.has(y))||(ga.has(y)&&gb.has(x))))rel.push({channel:k,kind:'electromagnetic'})}
 const dA=new Set(A.hd.definedCenters),dB=new Set(B.hd.definedCenters);
 const centers=CENTERS.map(c=>({center:c,A:dA.has(c)?'defined':'open',B:dB.has(c)?'defined':'open'}));
 const emerge=[];for(const a of A.activations.filter(x=>x.orientation==='personality'))for(const b of B.activations.filter(x=>x.orientation==='personality')){const e=channelEmerges(a,b);if(e.any)emerge.push({A:`${a.body} ${a.gate}.${a.line}.${a.color}.${a.tone}.${a.base}`,B:`${b.body} ${b.gate}.${b.line}.${b.color}.${b.tone}.${b.base}`,gateLine:e.gateLine,colorToneBase:e.colorToneBase})}
 const cross=[];for(const a of A.astrology.natal)for(const b of B.astrology.natal){if(['Earth','South Node'].includes(a.body)||['Earth','South Node'].includes(b.body))continue;let d=Math.abs(a.longitude-b.longitude);d=d>180?360-d:d;for(const s of ASPECTS){const o=Math.abs(d-s.angle);if(o<=s.orb)cross.push({A:a.body,B:b.body,aspect:s.name,orb:+o.toFixed(3)})}}
 return {relationship:rel,centers,sharedGates:[...ga].filter(g=>gb.has(g)).sort((x,y)=>x-y),emergence:{rule:'her field_core.js Hamming-1 rule at the gate/line level and the color/tone/base level (personality placements of A against B)',pairs:emerge},synastry:cross.sort((x,y)=>x.orb-y.orb),
  labels:{relationship:STANDARD+' (companionship, electromagnetic, dominance, compromise — definitions vary between teachers)',synastry:'conventional angles and orbs, '+STANDARD,emergence:'her rule'}}}
