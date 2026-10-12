// Numeric search only. The seed is not a semantic/state-space address.
export async function fitParticles({bounds,initial,score,particles=24,iterations=40,seed=719,cancelled=()=>false}){
 if(!bounds.length||bounds.length!==initial.length||bounds.some(([a,b],i)=>!Number.isFinite(a)||!Number.isFinite(b)||a>=b||!Number.isFinite(initial[i])||initial[i]<a||initial[i]>b)||particles<2||particles>64||iterations<1||iterations>100)throw Error('Invalid bounded swarm fit.');
 const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296};
 let best=initial.slice(),bestScore=score(best);if(!Number.isFinite(bestScore))throw Error('Fit objective must be finite.');const initialScore=bestScore;
 const swarm=Array.from({length:particles},(_,i)=>{const x=i===0?initial.slice():bounds.map(([a,b])=>a+random()*(b-a));const value=score(x);if(value<bestScore){best=x.slice();bestScore=value}return {x,v:x.map(()=>0),best:x.slice(),score:value}});
 for(let k=0;k<iterations;k++){
  if(cancelled())throw Error('Swarm fit cancelled.');
  for(const p of swarm){for(let j=0;j<bounds.length;j++){const [a,b]=bounds[j],limit=(b-a)*.2;p.v[j]=Math.max(-limit,Math.min(limit,.65*p.v[j]+1.4*random()*(p.best[j]-p.x[j])+1.4*random()*(best[j]-p.x[j])));p.x[j]=Math.max(a,Math.min(b,p.x[j]+p.v[j]));}const value=score(p.x);if(!Number.isFinite(value))throw Error('Fit objective must be finite.');if(value<p.score){p.best=p.x.slice();p.score=value}if(value<bestScore){best=p.x.slice();bestScore=value}}
  if(k%5===4)await new Promise(resolve=>setTimeout(resolve,0));
 }
 return {values:best,initialScore,score:bestScore,particles,iterations,backend:'javascript-cpu',workers:false,gpu:false};
}
export function lightingLoss(samples,values){let error=0;for(const p of samples)for(let c=0;c<3;c++){const delta=p.source[c]+values[c]+values[c+3]*p.x+values[c+6]*p.y-p.target[c];error+=delta*delta}return error/Math.max(1,samples.length*3)}
export async function fitLighting(samples,baseline){
 if(samples.length<12)return {values:[...baseline,0,0,0,0,0,0],accepted:false,reason:'insufficient-skin-samples'};
 const train=samples.filter((_,i)=>i%3!==0),held=samples.filter((_,i)=>i%3===0),initial=[...baseline.map(v=>Math.max(-60,Math.min(60,v))),0,0,0,0,0,0];
 const fit=await fitParticles({bounds:Array.from({length:9},(_,i)=>i<3?[-60,60]:[-40,40]),initial,score:v=>lightingLoss(train,v)+v.reduce((a,x,i)=>a+x*x*(i<3?.003:.02),0)});
 const before=lightingLoss(held,initial),after=lightingLoss(held,fit.values),accepted=after<=before;
 return {...fit,values:accepted?fit.values:initial,accepted,heldOutBefore:before,heldOutAfter:accepted?after:before,samples:samples.length,scope:'low-frequency-face-lighting',semanticRulesChanged:false};
}
