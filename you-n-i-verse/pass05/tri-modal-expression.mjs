/** Deterministic computational shape/color/sound projection of one formation state.
 * Explicitly a design mapping, not a claim of physical synesthesia.
 * No browser APIs, network, dependencies, workers or model calls.
 */
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const hash=s=>{let h=2166136261;for(const c of s)h=Math.imul(h^c.charCodeAt(0),16777619)>>>0;return h>>>0;};
const serialize=v=>JSON.stringify(v)??'undefined';
export function expressPrimitive({address,value=0,action='form',generation=0}){
  if(!address||!Number.isInteger(address.hexagram6)||address.hexagram6<0||address.hexagram6>63)throw Error('six-bit address required');
  if(!Number.isInteger(generation)||generation<0)throw Error('invalid generation');
  const bits=Array.from({length:6},(_,i)=>(address.hexagram6>>i)&1);
  const seed=hash(serialize({ontology:address.ontology??{},value,action,generation}));
  const magnitude=typeof value==='number'&&Number.isFinite(value)?Math.abs(value):serialize(value).length;
  const energy=clamp(Math.log1p(magnitude)/10,0,1);
  const symmetry=bits.filter((b,i)=>i<3&&b===bits[5-i]).length/3;
  const vertices=3+(bits[0]+2*bits[1]+4*bits[2])%6;
  const rotation=(seed%360)*Math.PI/180;
  const points=Array.from({length:vertices},(_,i)=>{
    const theta=rotation+i*2*Math.PI/vertices;
    const r=0.65+0.3*(bits[i%6]?1:0)+0.05*energy;
    return [Number((r*Math.cos(theta)).toFixed(5)),Number((r*Math.sin(theta)).toFixed(5))];
  });
  const hue=(address.hexagram6*360/64+seed%29)%360;
  const saturation=Math.round(45+40*energy);
  const lightness=Math.round(40+25*symmetry);
  const frequency=110*Math.pow(2,(address.hexagram6%36)/12);
  const overtones=[1,2,3].map((harmonic,i)=>({harmonic,amplitude:Number(((i===0?0.75:i===1?0.18:0.07)*(bits[(i+3)%6]?1:0.5)).toFixed(4))}));
  return {schema:'synthia-trimodal-expression-v1',source:{address:structuredClone(address),action,generation,value:structuredClone(value)},shape:{vertices,points,symmetry},color:{hue,saturation,lightness,css:`hsl(${hue.toFixed(2)} ${saturation}% ${lightness}%)`},sound:{frequencyHz:Number(frequency.toFixed(4)),overtones,durationSeconds:0.25+0.05*(generation%6)},energy};
}
export function toSVG(expression,{size=256}={}){
  if(!Number.isFinite(size)||size<=0||size>4096)throw Error('invalid size');
  const polygon=expression.shape.points.map(([x,y])=>`${(size/2+x*size*0.39).toFixed(2)},${(size/2+y*size*0.39).toFixed(2)}`).join(' ');
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}"><polygon points="${polygon}" fill="${expression.color.css}" stroke="currentColor" stroke-width="2"/></svg>`;
}
export function toPCM(expression,{sampleRate=16000}={}){
  if(!Number.isInteger(sampleRate)||sampleRate<8000||sampleRate>192000)throw Error('invalid sample rate');
  const n=Math.round(expression.sound.durationSeconds*sampleRate);
  const out=new Int16Array(n);
  for(let i=0;i<n;i++){
    const t=i/sampleRate, fade=Math.min(1,t*60,(n-i)/sampleRate*60);
    let s=0;for(const overtone of expression.sound.overtones)s+=overtone.amplitude*Math.sin(2*Math.PI*expression.sound.frequencyHz*overtone.harmonic*t);
    out[i]=Math.round(clamp(s*fade*0.75,-1,1)*32767);
  }
  return out;
}
export function expressExecution(execution,{action='execute',generation=0}={}){
  return expressPrimitive({address:execution.address,value:execution.value,action,generation});
}
