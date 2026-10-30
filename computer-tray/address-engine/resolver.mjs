// Address -> filled state + all 8 BHANTUGH encodings. Pure JS, no backend.
// Address order is the doc's Section M traversal: Gate -> Line -> Color -> Tone -> Base.
import {gateToMacro,macroToGate} from './vendor/state-space.mjs';
import {ORDER,facesForHexagram} from './vendor/axis-order.mjs';
import {natoOf,translitOf} from './vendor/letter-layers.mjs';
const _u=new URL('./state-space.data.json',import.meta.url);
export const S=typeof process!=='undefined'&&process.versions?.node&&_u.protocol==='file:'?JSON.parse((await import('node:fs')).readFileSync(_u)):await (await fetch(_u)).json(); // load-only change; rules untouched
const SIGNS=S.meta.zodiacOrder; // as listed in the doc's Zodiac table (Aries..Pisces)
export const SIZES={gate:5+37/60+30/3600,line:56/60+15/3600,color:(9+22.5/60)/60,tone:(1+33.75/60)/60,base:18.75/3600}; // Section M
export const COUNTS={gate:64,line:6,color:6,tone:6,base:5};
// ---- mandala start degrees, read from every position claim in the doc ----
const signIdx=s=>{if(!s)return -1;const k=s.trim().slice(0,3).toLowerCase();return SIGNS.findIndex(z=>z.slice(0,3).toLowerCase()===k)};
const dms=(d,m,s)=>+d+ +m/60+ +(s||0)/3600;
function startOf(text,signCol){
  const re=/(\d{1,2})°(\d{2})'(?:(\d{2}(?:\.\d+)?)")?\s*([A-Za-z]+)?/g;const p=[...text.matchAll(re)];if(!p.length)return null;
  const a=p[0],b=p[1];let si=signIdx(a[4]);
  if(si<0&&b){si=signIdx(b[4]);if(si>=0&&dms(b[1],b[2],b[3])<dms(a[1],a[2],a[3]))si=(si+11)%12}
  if(si<0)si=signIdx((signCol||'').split(/[\/ ]/)[0]);
  return si<0?null:+(si*30+dms(a[1],a[2],a[3])).toFixed(6);
}
export const POSITIONS={};
for(const g of Object.values(S.gates)){const c=[];
  const unsigned=[];
  for(const x of g.claims){const k=Object.keys(x).find(k=>/Position/.test(k)),pos=k&&x[k];if(pos){const v=startOf(pos,x['Sign']||x['Astrological Sign']);if(v!=null)c.push({id:x.id,start:v,text:pos});else{const m=pos.match(/(\d{1,2})°(\d{2})'(\d{2})?/);if(m)unsigned.push({id:x.id,inSign:+dms(m[1],m[2],m[3]).toFixed(6),text:pos})}}}
  for(const n of g.notes){const m=(n.text||'').match(/Location on Mandala:\**\s*(.*)/);if(m){const v=startOf(m[1]);if(v!=null)c.push({id:n.id,start:v,text:m[1]})}}
  const uniq=[...new Set(c.map(x=>x.start))];const clash=uniq.length===1&&unsigned.some(u=>Math.abs(u.inSign-uniq[0]%30)>1e-6);
  POSITIONS[g.gate]={claims:c,unsignedClaims:unsigned,start:uniq.length===1&&!clash?uniq[0]:null,conflict:uniq.length>1||clash,candidates:uniq};}
// ---- address ----
export function parse(a){
  if(typeof a==='string'){const p=a.split('.').map(Number);if(p.length!==5)throw new RangeError('address is gate.line.color.tone.base');a={gate:p[0],line:p[1],color:p[2],tone:p[3],base:p[4]}}
  if(a&&a.degree!=null)return fromDegree(a.degree);
  for(const k of Object.keys(COUNTS))if(!Number.isInteger(a[k])||a[k]<1||a[k]>COUNTS[k])throw new RangeError(`${k} must be 1..${COUNTS[k]}`);
  return {gate:a.gate,line:a.line,color:a.color,tone:a.tone,base:a.base};
}
export const code=a=>`${a.gate}.${a.line}.${a.color}.${a.tone}.${a.base}`;
export const indexOf=a=>((((a.gate-1)*6+a.line-1)*6+a.color-1)*6+a.tone-1)*5+a.base-1; // 0..69119
export function fromIndex(i){const base=i%5+1;i=(i-base+1)/5;const tone=i%6+1;i=(i-tone+1)/6;const color=i%6+1;i=(i-color+1)/6;const line=i%6+1;return {gate:(i-line+1)/6+1,line,color,tone,base}}
// Degree (0..360 from 0° of the doc's first sign) -> address, only through gates whose doc positions agree.
export function fromDegree(deg){
  deg=+(((deg%360)+360)%360).toFixed(9);
  const hits=Object.entries(POSITIONS).filter(([,p])=>p.start!=null&&((deg-p.start+360)%360)<SIZES.gate-1e-12);
  if(hits.length!==1)return {unresolved:true,degree:deg,candidates:hits.map(h=>+h[0]),reason:hits.length?'more than one gate claims this arc':'no agreed gate position covers this degree (see open questions)'};
  const [g,p]=hits[0];let r=(deg-p.start+360)%360;const out={gate:+g};
  for(const k of ['line','color','tone','base']){const i=Math.min(COUNTS[k]-1,Math.floor(r/SIZES[k]+1e-9));out[k]=i+1;r-=i*SIZES[k]}
  return out;
}
// ---- state ----
const first=(arr,key)=>arr.map(x=>x[key]).find(Boolean);
export function state(addr){
  const a=parse(addr);if(a.unresolved)return a;
  const g=S.gates[a.gate],anchor=g.claims.filter(c=>c.kind==='gateAnchor');
  const toneRow=S.tones[a.tone].claims[0],baseRow=S.bases[a.base].claims[0];
  const btCell=baseRow[Object.keys(baseRow).find(k=>k.startsWith(`Tone ${a.tone} `))];
  const inList=(s)=>new RegExp(`Color ${a.color}\\b`).test(s||'');
  const tc=inList(toneRow['Compatible Colors'])?'compatible':inList(toneRow['Incompatible Combinations'])?'incompatible':'not stated in doc';
  return {address:code(a),index:indexOf(a),...a,
    gateState:{names:[...new Set(g.claims.map(c=>c['I Ching Name']||c['Name']).filter(Boolean))],archetype:first(anchor,'Primary Archetype'),baseAssociation:first(anchor,'Base Association'),entryAnchorNode:first(anchor,'Entry Anchor Node'),zodiac:[...new Set(g.claims.map(c=>c['Zodiac Sign']||c['Sign']||c['Astrological Sign']).filter(Boolean))],house:first(anchor,'House'),centers:[...new Set(g.claims.map(c=>c['Biological Center']||c['Center']).filter(Boolean))],harmonic:[...new Set(g.claims.map(c=>c['Harmonic']||c['Harmonic Gate']).filter(Boolean))],mandalaStart:POSITIONS[a.gate].start,positionConflict:POSITIONS[a.gate].conflict,notes:g.notes.length,docIds:g.claims.map(c=>c.id)},
    lineState:{orbital:S.lines[a.line].claims.map(c=>({line:c['Line #'],orbital:c['Orbital Name'],angle:c['Traversal Angle'],effect:c['Mechanics & Graph Effect']})),gateLine:g.lines[a.line].map(x=>x.text||x.Expression||x.cells)},
    colorState:{color:a.color,names:S.colors[a.color]?.names??[]},
    toneState:{tone:a.tone,name:S.tones[a.tone].name,binaryFraming:toneRow['Binary Operational Framing'],toneColor:tc},
    baseState:{base:a.base,name:S.bases[a.base].name,nonPrimitive:S.bases[a.base].nonPrimitive,baseToneCell:btCell},
    axisRows:S.axes.filter(r=>new RegExp(`Base ${a.base}\\b|D${a.base}\\b`).test(r['Primary Associated Bases'])).map(r=>({id:r.id,axis:r['Axis Type'],vector:r['Directional Vector']})),
    ruleO:{baseTone:btCell&&/Valid|Resonant/.test(btCell)?'valid':'halt',toneColor:tc,resonance:/Resonant Anchor/.test(btCell||'')?'amplify 2x':'standard emission'},
    encodings:encode(a)};
}
// ---- BHANTUGH ----
const label=a=>{const st=S;return [st.gates[a.gate].claims.find(c=>c['I Ching Name'])?.['I Ching Name'],st.lines[a.line].claims[0]['Orbital Name'],st.colors[a.color]?.names[0],st.tones[a.tone].name,st.bases[a.base].name].join(' ')};
export function encode(addr){
  const a=parse(addr),c=code(a),chars=[...c],words=label(a),wchars=[...words];
  return {text:c,words,
    B:{binary:chars.map(ch=>ch.codePointAt(0).toString(2).padStart(8,'0')).join(' '),components:Object.fromEntries(Object.keys(COUNTS).map(k=>[k,(a[k]-1).toString(2).padStart(k==='gate'?6:3,'0')])),index:indexOf(a).toString(2).padStart(17,'0')},
    H:chars.map(ch=>`&#x${ch.codePointAt(0).toString(16).toUpperCase()};`).join(''),
    A:chars.map(ch=>ch.codePointAt(0)),
    N:wchars.map(ch=>natoOf(ch)?.word??(ch===' '?'(space)':null)),
    T:wchars.map(ch=>{const t=translitOf(ch);return {ch,russian:t?.russian?.map(x=>x.cyr)??null,greek:t?.greek??null}}),
    U:chars.map(ch=>'U+'+ch.codePointAt(0).toString(16).toUpperCase().padStart(4,'0')),
    G:{hexagramCode:gateToMacro(a.gate),dodecaFaces:facesForHexagram(gateToMacro(a.gate)),lineAxis:ORDER[a.line-1]},
    Hx:{index:indexOf(a).toString(16).toUpperCase().padStart(5,'0'),bytes:chars.map(ch=>ch.codePointAt(0).toString(16).toUpperCase().padStart(2,'0')).join(' ')}};
}
// decoders for round-trip
export const decode={
  B:s=>s.binary.split(' ').map(b=>String.fromCodePoint(parseInt(b,2))).join(''),
  H:s=>s.replace(/&#x([0-9A-F]+);/g,(_,h)=>String.fromCodePoint(parseInt(h,16))),
  A:s=>String.fromCodePoint(...s),
  U:s=>s.map(u=>String.fromCodePoint(parseInt(u.slice(2),16))).join(''),
  Hx:s=>code(fromIndex(parseInt(s.index,16))),
  G:(s,rest)=>{const order=ORDER;let codeN=0;order.forEach((o,i)=>{if(s.dodecaFaces[i]===o.yangFace)codeN|=1<<i});return code({...rest,gate:macroToGate(codeN)})},
};
