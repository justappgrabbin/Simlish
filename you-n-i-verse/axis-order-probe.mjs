// Probe (prints counts): every axis order (720) × yang-pole choice (64) = 46,080 configs tested against constraints; orbits under the 120 dodecahedron symmetries.
import {FACES,AXES} from './dodeca-face-axes.mjs';
const P=(1+Math.sqrt(5))/2,dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2],n2=dot(FACES[0],FACES[0]);
const mul=(A,B)=>A.map((r,i)=>[0,1,2].map(j=>r[0]*B[0][j]+r[1]*B[1][j]+r[2]*B[2][j]));
const ap=(M,v)=>M.map(r=>dot(r,v));
const rot=(ax,t)=>{const n=Math.hypot(...ax),[x,y,z]=ax.map(a=>a/n),c=Math.cos(t),s=Math.sin(t),C=1-c;return [[c+x*x*C,x*y*C-z*s,x*z*C+y*s],[y*x*C+z*s,c+y*y*C,y*z*C-x*s],[z*x*C-y*s,z*y*C+x*s,c+z*z*C]]};
const gens=[rot(FACES[0],2*Math.PI/5),[[0,0,1],[1,0,0],[0,1,0]],[[-1,0,0],[0,-1,0],[0,0,-1]]];
const faceIdx0=v=>FACES.findIndex(f=>f.every((x,k)=>Math.abs(x-v[k])<1e-6));const key=M=>FACES.map(f=>faceIdx0(M.map(r=>r[0]*f[0]+r[1]*f[1]+r[2]*f[2]))).join();let G=[[[1,0,0],[0,1,0],[0,0,1]]],seen=new Set([key(G[0])]);
for(let i=0;i<G.length;i++)for(const g of gens){const M=mul(g,G[i]);if(!seen.has(key(M))){seen.add(key(M));G.push(M)}}
console.log('group order',G.length);
const faceIdx=v=>FACES.findIndex(f=>f.every((x,k)=>Math.abs(x-v[k])<1e-6));
const perms=G.map(M=>FACES.map(f=>faceIdx(ap(M,f)))); // face permutations
// axis of face, pole bit
const axOf=f=>AXES.findIndex(a=>a.includes(f)),pole=f=>AXES[axOf(f)].indexOf(f);
// configs: order (axis per line 0..5) + yangPole per line (0/1) => yang face per line
const orders=[];const pm=(a,r)=>{if(!a.length){orders.push(r);return}a.forEach((x,i)=>pm(a.filter((_,k)=>k!==i),[...r,x]))};pm([0,1,2,3,4,5],[]);
const adj=(i,j)=>Math.abs(dot(FACES[i],FACES[j])/n2-1/Math.sqrt(5))<1e-9;
// cube synthemes: 2-fold axes = midpoints of adjacent face pairs; pair of axes perpendicular to it
const cubes=[];// each 2-fold axis direction: f_i+f_j for adjacent faces
const two=[];for(let i=0;i<12;i++)for(let j=i+1;j<12;j++)if(adj(i,j)){const v=FACES[i].map((x,k)=>x+FACES[j][k]);if(!two.some(w=>Math.abs(Math.abs(dot(v,w))-Math.hypot(...v)*Math.hypot(...w))<1e-6))two.push(v)}
const perpPair=v=>AXES.map((a,i)=>i).filter(i=>Math.abs(dot(FACES[AXES[i][0]],v))<1e-6);
const pairs=two.map(perpPair);
// cube = 3 mutually perpendicular 2-fold axes
const synth=[];for(let a=0;a<15;a++)for(let b=a+1;b<15;b++)for(let c=b+1;c<15;c++)if([[a,b],[a,c],[b,c]].every(([x,y])=>Math.abs(dot(two[x],two[y]))<1e-6))synth.push([pairs[a],pairs[b],pairs[c]].map(p=>p.join('')).sort().join('|'));
console.log('2-fold axes',two.length,'each perpendicular to',new Set(pairs.map(p=>p.length)),'axis-pairs; cubes',synth.length,synth);
let total=0,c1=0,c2=0,c3=0,c23=0,c123=0,cap=0,all=[];
for(const o of orders)for(let y=0;y<64;y++){total++;const yang=o.map((a,l)=>AXES[a][(y>>l)&1]);
  const corr=[[0,3],[1,4],[2,5]];
  const k1=corr.every(([i,j])=>adj(yang[i],yang[j]))||corr.every(([i,j])=>!adj(yang[i],yang[j]));
  const k2=perms.some(p=>yang.every((f,l)=>p[f]===yang[5-l]));
  const syn=corr.map(([i,j])=>[o[i],o[j]].sort().join('')).sort().join('|');const k3=synth.includes(syn);
  const isCap=yang.some(f=>yang.filter(g=>adj(f,g)).length===5);
  if(k1)c1++;if(k2)c2++;if(k3)c3++;if(k2&&k3)c23++;if(k1&&k2&&k3){c123++;all.push([o,y])}if(k1&&k2&&k3&&isCap)cap++;}
console.log({total,k1_corrSameAngle:c1,k2_reversalIsSymmetry:c2,k3_corrPairsAreACube:c3,k2k3:c23,k1k2k3:c123,plusHeavenCap:cap});
// orbits of survivors under the group
const canon=([o,y])=>{const yang=o.map((a,l)=>AXES[a][(y>>l)&1]);return perms.map(p=>yang.map(f=>p[f]).join(',')).sort()[0]};
console.log('orbits of k1k2k3 under 120 symmetries',new Set(all.map(canon)).size);
const capS=all.filter(([o,y])=>{const yang=o.map((a,l)=>AXES[a][(y>>l)&1]);return yang.some(f=>yang.filter(g=>adj(f,g)).length===5)});
console.log('orbits with cap',new Set(capS.map(canon)).size);
// extra combos
const tri=(fs)=>adj(fs[0],fs[1])&&adj(fs[1],fs[2])&&adj(fs[0],fs[2]);
let R={k1k3:0,k3cap:0,k3triTri:0,k3capTri:0,k1k3cap:0,k1k3capTri:0};const keep={};
const axisPerm=new Set(perms.map(p=>AXES.map(a=>axOf(p[a[0]])).join('')));console.log('distinct axis permutations from symmetries',axisPerm.size,'; any fixed-point-free involution?',[...axisPerm].some(s=>[...s].every((x,i)=>+x!==i&&+s[+x]===i)));
for(const o of orders)for(let y=0;y<64;y++){const yang=o.map((a,l)=>AXES[a][(y>>l)&1]);const corr=[[0,3],[1,4],[2,5]];
  const k1=corr.every(([i,j])=>adj(yang[i],yang[j]))||corr.every(([i,j])=>!adj(yang[i],yang[j]));
  const k3=synth.includes(corr.map(([i,j])=>[o[i],o[j]].sort().join('')).sort().join('|'));
  const cap=yang.some(f=>yang.filter(g=>adj(f,g)).length===5);const tt=tri(yang.slice(0,3))&&tri(yang.slice(3));
  const add=(k,c)=>{if(c){R[k]++;(keep[k]??=[]).push([o,y])}};
  add('k1k3',k1&&k3);add('k3cap',k3&&cap);add('k3triTri',k3&&tt);add('k3capTri',k3&&cap&&tt);add('k1k3cap',k1&&k3&&cap);add('k1k3capTri',k1&&k3&&cap&&tt);}
console.log(R);for(const k in keep)console.log(k,'orbits',new Set(keep[k].map(canon)).size);
const ex=keep.k1k3capTri?.[0];if(ex){const [o,y]=ex;console.log('example order',o,'yang bits',y.toString(2).padStart(6,'0'),'yang faces',o.map((a,l)=>AXES[a][(y>>l)&1]))}
const res={};
for(const [o,y] of keep.k3cap){const yang=o.map((a,l)=>AXES[a][(y>>l)&1]);const c=yang.findIndex(f=>yang.filter(g=>adj(f,g)).length===5);
  const ring=[0,1,2,3,4,5].filter(l=>l!==c);const pathOK=ring.every((l,i)=>i===0||adj(yang[ring[i-1]],yang[l]));const cyc=pathOK&&adj(yang[ring[0]],yang[ring[4]]);
  const k=`centre=line${c+1} ringInOrder=${pathOK}`;(res[k]??=[]).push([o,y]);}
for(const k in res)console.log(k,res[k].length,'orbits',new Set(res[k].map(canon)).size);
const r2={};
for(const o of orders)for(let y=0;y<64;y++){const yang=o.map((a,l)=>AXES[a][(y>>l)&1]);const c=yang.findIndex(f=>yang.filter(g=>adj(f,g)).length===5);if(c<0)continue;
  const ring=[0,1,2,3,4,5].filter(l=>l!==c);if(!ring.every((l,i)=>i===0||adj(yang[ring[i-1]],yang[l])))continue;
  const k3=synth.includes([[0,3],[1,4],[2,5]].map(([i,j])=>[o[i],o[j]].sort().join('')).sort().join('|'));
  (r2[`centre=line${c+1} corrIsCube=${k3}`]??=[]).push([o,y]);}
for(const k in r2)console.log('capRing',k,r2[k].length,'orbits',new Set(r2[k].map(canon)).size);
