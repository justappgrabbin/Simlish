// ABC (artificial bee colony) as a primitive of the swarm. It EXISTS in her files; it is connected here without altering any original.
// Where it is found (searched: all 319 unpacked zips + 74 zip/docx archives opened in memory, swarm-os-v14 incl. its zip, pipe/, her docs, this repo's branches/history, oct7src = justappgrabbin/SynthAi):
//   V9  swarm-os-v14/x/v9/swarm_os_full.py lines 3, 15 ("PSO+GA+ABC Hybrid"), 89 (roles employed 0.4 / onlooker 0.4 / scout 0.2), 132-155 abc_phase()   (also swarm-os-v14/x/v10/swarm_os_bantu.py lines 13, 114, 178)
//   S   oct7src/computer/runtime/morph-search.mjs lines 36-60 (employed/onlooker/scout, one axis per bee), documented in computer/MORPH-OPTIMIZATION.md (commit af811ab, author justappgrabbin)
// Both are ported rule for rule, as two DIFFERENT ABC primitives side by side (neither replaces the other). Pure JavaScript, no dependencies.
// INVESTIGATING:
export const ABC_INVESTIGATING=Object.freeze([
 'V9 abc_phase is written for 2-D agents whose fitness is 1/(1+distance to a moving target) + words*0.05 + warmth*0.3 + trail*0.001. Here the objective of a genotype (number of wrong output bits) takes the place of the distance term only; the words/warmth/trail terms are not used. That substitution is an engineering reading, not hers.',
 'a six-bit atom is a hexagram, not a number: moving a position by 0.03 means moving atoms numerically (the existing PSO in swarm-fit.mjs does the same). Whether numeric motion is meaningful for hexagram genes is not established.',
 'V9 and S differ (V9 moves every coordinate by Gaussian noise; S changes one axis relative to a peer). Which one she means by "ABC" is not stated in her docs; both are kept.']);
export const V9_ROLE_SPLIT=Object.freeze({employed:0.4,onlooker:0.4,scout:0.2});
const lcg=seed=>()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296};
const gauss=r=>{let u=0;while(u===0)u=r();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*r())};
const clip01=x=>x<0?0:x>1?1:x;

// ---- V9: line-for-line port of swarm_os_full.py abc_phase() ----
export class AbcV9{
 constructor({dims,objective,agents=40,seed=719,initial=null,spread=null}){
  this.n=dims;this.objective=objective;this.rand=lcg(seed);this.evals=0;this.best={score:Infinity,pos:null};
  this.agents=Array.from({length:agents},(_,i)=>{
   const r=this.rand(),role=r<0.4?'employed':(r<0.8?'onlooker':'scout');
   const pos=Float64Array.from({length:dims},(_,k)=>i===0&&initial?initial[k]:(initial&&spread!==null?clip01(initial[k]+gauss(this.rand)*spread):this.rand()));
   const a={id:i,role,age:0,pos,pbestPos:pos.slice(),pbestScore:0};return a});
 }
 fit(a){const s=this.objective(a.pos);this.evals++;if(s<this.best.score)this.best={score:s,pos:a.pos.slice()};return 1/(1+s)}   // v9: 1/(1+dist) (+ words/warmth/trail: not used, see ABC_INVESTIGATING)
 step(){
  const A=this.agents,R=this.rand;
  for(const a of A)if(a.role==='employed'){                                   // employed: random walk 0.03, keep pbest if better
   for(let k=0;k<this.n;k++)a.pos[k]=clip01(a.pos[k]+gauss(R)*0.03);
   const s=this.fit(a);if(s>a.pbestScore){a.pbestScore=s;a.pbestPos=a.pos.slice()}}
  const fits=A.map(a=>this.fit(a)),total=fits.reduce((x,y)=>x+y,0);
  if(total>0)for(let c=0;c<Math.floor(A.length/2);c++){                          // onlookers: choose by fitness share, move 0.4 toward their pbest, noise 0.02
   let cur=R()*total,idx=A.length-1;for(let j=0;j<A.length;j++){cur-=fits[j];if(cur<=0){idx=j;break}}
   const a=A[idx];for(let k=0;k<this.n;k++)a.pos[k]=clip01(a.pos[k]+(a.pbestPos[k]-a.pos[k])*0.4+gauss(R)*0.02)}
  for(const a of A)if(a.role==='scout'){a.age++;if(a.age>15){for(let k=0;k<this.n;k++)a.pos[k]=R();a.age=0}}   // scouts: after 15 steps abandon and restart anywhere
  return this.best;
 }
}
// ---- S: port of morph-search.mjs abc()/bee() (lower score better; one axis changed relative to a peer; scouts replace sources stalled for 8 trials) ----
export class AbcSingleAxis{
 constructor({dims,objective,agents=12,seed=719,initial=null}){
  this.n=dims;this.objective=objective;this.rand=lcg(seed);this.evals=0;this.best={score:Infinity,pos:null};this.N=agents;
  this.src=Array.from({length:agents},(_,i)=>this.cand(i===0&&initial?Float64Array.from(initial):Float64Array.from({length:dims},()=>this.rand())));
 }
 cand(pos){const p=pos.map(clip01),s=this.objective(p);this.evals++;if(s<this.best.score)this.best={score:s,pos:p.slice()};return {pos:p,score:s,trials:0}}
 bee(i){const s=this.src[i],R=this.rand;let peer=Math.floor(R()*(this.N-1));if(peer>=i)peer++;const axis=Math.floor(R()*this.n),v=s.pos.slice();
  v[axis]+=(R()*2-1)*(v[axis]-this.src[peer].pos[axis]);const nx=this.cand(v);if(nx.score<s.score)this.src[i]=nx;else s.trials++}
 step(){
  for(let i=0;i<this.N;i++)this.bee(i);                                          // employed
  const min=Math.min(...this.src.map(c=>c.score)),w=this.src.map(c=>1/(1+c.score-min)),tot=w.reduce((a,b)=>a+b,0);
  for(let i=0;i<this.N;i++){let cur=this.rand()*tot,sel=this.N-1;for(let j=0;j<this.N;j++){cur-=w[j];if(cur<=0){sel=j;break}}this.bee(sel)}   // onlookers
  for(let i=0;i<this.N;i++)if(this.src[i].trials>=8)this.src[i]=this.cand(Float64Array.from({length:this.n},()=>this.rand()));                   // scouts
  return this.best;
 }
}
// ---- the connection: same call shape as swarm-fit.mjs fitParticles; atoms are 0..63 and positions are normalised to 0..1 ----
export const ABC_VARIANTS=Object.freeze({v9:AbcV9,'single-axis':AbcSingleAxis});
export function abcFit({bounds,initial,score,agents,iterations=40,seed=719,variant='v9',spread}){
 const n=bounds.length,toX=p=>Array.from(p,(v,k)=>bounds[k][0]+v*(bounds[k][1]-bounds[k][0])),init01=initial.map((v,k)=>(v-bounds[k][0])/(bounds[k][1]-bounds[k][0]));
 const Cls=ABC_VARIANTS[variant];if(!Cls)throw new RangeError('unknown ABC variant '+variant);
 const s=new Cls({dims:n,objective:p=>score(toX(p)),agents:agents??(variant==='v9'?40:12),seed,initial:init01,...(variant==='v9'&&spread!==undefined?{spread}:{})});
 const initialScore=score(initial);for(let i=0;i<iterations;i++)s.step();
 return {values:s.best.pos?toX(s.best.pos):initial.slice(),initialScore,score:Math.min(initialScore,s.best.score),evaluations:s.evals,variant,backend:'javascript-cpu',workers:false};
}
