// Trident in pure JavaScript: reads her trident_syntia.pt (a PyTorch state_dict zip) and runs the forward pass of model.py (stellarproximology/Trident) on the CPU.
// No dependency, no backend, no worker. PORT of model.py (reference copy: integrated/reference/hf-synthia/); the weights are hers (stellarproximology/Synthia).
// Scope, stated plainly: forward + generate WITHOUT the RAG gate (rag=None path of model.py). The RAGFusionGate weights are loaded but unused because chunk embeddings come from rag.py, which is not ported.
// Tokenizer: train_syntia.py trains on ids = [ord(c) % 4096 for c in text[:64].lower()] (character ids), NOT on trident_vocab.json (a 4096-word list). tokenize() follows the training code.
import {readZip} from '../zip-lite.mjs';
export const CONFIG=Object.freeze({vocab:4096,maxSeq:256,d:128,heads:4,layers:4,dff:512,headLayers:1,headDff:256,headNames:['code','math','research'],eps:1e-5});
// ---- minimal unpickler (protocol 2 subset that torch.save writes for a state_dict) ----
function unpickle(b,onTensor){
 const dv=new DataView(b.buffer,b.byteOffset,b.byteLength),td=new TextDecoder();let p=0;const st=[],marks=[],memo=new Map();
 const MARK=Symbol('mark'),popMark=()=>{const m=marks.pop();const items=st.splice(m);return items};
 const G={'collections OrderedDict':()=>({__od:true,items:new Map()}),'torch._utils _rebuild_tensor_v2':(storage,off,size,stride)=>onTensor({storage,off,size,stride})};
 while(true){
  const op=b[p++];
  switch(op){
   case 0x80:p++;break;                                    // PROTO
   case 0x63:{const nl=(s)=>{const e=b.indexOf(10,s);return e};let e=nl(p);const mod=td.decode(b.subarray(p,e));p=e+1;e=nl(p);const name=td.decode(b.subarray(p,e));p=e+1;const key=mod+' '+name;st.push({global:key});break} // GLOBAL
   case 0x71:memo.set(b[p++],st[st.length-1]);break;      // BINPUT
   case 0x72:memo.set(dv.getUint32(p,true),st[st.length-1]);p+=4;break; // LONG_BINPUT
   case 0x68:st.push(memo.get(b[p++]));break;             // BINGET
   case 0x6a:st.push(memo.get(dv.getUint32(p,true)));p+=4;break; // LONG_BINGET
   case 0x29:st.push([]);break;                            // EMPTY_TUPLE
   case 0x85:{const a=st.pop();st.push([a]);break}         // TUPLE1
   case 0x86:{const y=st.pop(),x=st.pop();st.push([x,y]);break}
   case 0x87:{const z=st.pop(),y=st.pop(),x=st.pop();st.push([x,y,z]);break}
   case 0x28:marks.push(st.length);break;                  // MARK
   case 0x74:st.push(popMark());break;                     // TUPLE
   case 0x52:{const args=st.pop(),f=st.pop();const fn=f.global&&G[f.global];if(!fn)throw new Error('unpickle: unsupported callable '+JSON.stringify(f));st.push(fn(...args));break} // REDUCE
   case 0x58:{const n=dv.getUint32(p,true);p+=4;st.push(td.decode(b.subarray(p,p+n)));p+=n;break} // BINUNICODE
   case 0x4b:st.push(b[p++]);break;                        // BININT1
   case 0x4d:st.push(dv.getUint16(p,true));p+=2;break;     // BININT2
   case 0x4a:st.push(dv.getInt32(p,true));p+=4;break;      // BININT
   case 0x89:st.push(false);break;                         // NEWFALSE
   case 0x88:st.push(true);break;                          // NEWTRUE
   case 0x4e:st.push(null);break;                          // NONE
   case 0x7d:st.push({__dict:true,items:new Map()});break; // EMPTY_DICT
   case 0x73:{const v=st.pop(),k=st.pop(),d=st[st.length-1];(d.items).set(k,v);break} // SETITEM
   case 0x75:{const it=popMark(),d=st[st.length-1];for(let i=0;i<it.length;i+=2)d.items.set(it[i],it[i+1]);break} // SETITEMS
   case 0x62:{const state=st.pop();break}                  // BUILD (OrderedDict metadata) ignored
   case 0x51:{const pid=st.pop();if(pid[0]!=='storage')throw new Error('unpickle: unknown persistent id');st.push({storageKey:pid[2],dtype:pid[1].global});break} // BINPERSID
   case 0x2e:return st.pop();                              // STOP
   default:throw new Error('unpickle: unsupported opcode 0x'+op.toString(16)+' at '+(p-1));
  }
 }
}
// ---- load: returns {tensors: Map(name -> Float32Array)} ----
export async function loadStateDict(ptBytes){
 const files=await readZip(ptBytes),by=new Map(files.map(f=>[f.path,f.data]));
 const pkl=files.find(f=>/\/data\.pkl$/.test(f.path));if(!pkl)throw new Error('Not a torch zip checkpoint (no data.pkl)');
 const root=pkl.path.replace(/data\.pkl$/,''),tensors=new Map();
 const od=unpickle(pkl.data,t=>t);
 for(const [name,t] of od.items){
  if(t.storage.dtype!=='torch FloatStorage')throw new Error('Only float32 tensors supported, got '+t.storage.dtype+' for '+name);
  const raw=by.get(root+'data/'+t.storage.storageKey);if(!raw)throw new Error('missing storage '+t.storage.storageKey);
  const n=t.size.reduce((a,b)=>a*b,1),f=new Float32Array(raw.buffer.slice(raw.byteOffset+t.off*4,raw.byteOffset+t.off*4+n*4));
  tensors.set(name,{data:f,shape:t.size});
 }
 return {tensors};
}
// ---- math (float32 storage, float64 accumulation) ----
function erf(x){const s=x<0?-1:1;x=Math.abs(x);if(x>6)return s;let sum=x,term=x,n=0;if(x<3){while(Math.abs(term)>1e-17){n++;term*=-x*x/n;sum+=term/(2*n+1);if(n>200)break}return s*2/Math.sqrt(Math.PI)*sum}
 // continued fraction for larger x
 let f=0;for(let k=60;k>=1;k--)f=k/2/(x+f);return s*(1-Math.exp(-x*x)/Math.sqrt(Math.PI)/(x+f))}
const gelu=x=>0.5*x*(1+erf(x/Math.SQRT2));
function linear(x,T,inD,W,B,outD){const y=new Float32Array(T*outD);for(let t=0;t<T;t++){const xo=t*inD,yo=t*outD;for(let o=0;o<outD;o++){let s=B?B[o]:0;const wo=o*inD;for(let i=0;i<inD;i++)s+=x[xo+i]*W[wo+i];y[yo+o]=s}}return y}
function layerNorm(x,T,D,w,b,eps){const y=new Float32Array(T*D);for(let t=0;t<T;t++){let m=0;for(let i=0;i<D;i++)m+=x[t*D+i];m/=D;let v=0;for(let i=0;i<D;i++){const d=x[t*D+i]-m;v+=d*d}v/=D;const inv=1/Math.sqrt(v+eps);for(let i=0;i<D;i++)y[t*D+i]=(x[t*D+i]-m)*inv*w[i]+b[i]}return y}
function attention(x,T,pre,P,nHeads){ // nn.MultiheadAttention(batch_first, causal mask)
 const D=CONFIG.d,hd=D/nHeads,Wi=P.get(pre+'.in_proj_weight').data,Bi=P.get(pre+'.in_proj_bias').data;
 const qkv=linear(x,T,D,Wi,Bi,3*D),out=new Float32Array(T*D),sc=1/Math.sqrt(hd),s=new Float64Array(T);
 for(let h=0;h<nHeads;h++)for(let t=0;t<T;t++){let mx=-Infinity;for(let u=0;u<=t;u++){let d=0;for(let i=0;i<hd;i++)d+=qkv[t*3*D+h*hd+i]*qkv[u*3*D+D+h*hd+i];s[u]=d*sc;if(s[u]>mx)mx=s[u]}
  let z=0;for(let u=0;u<=t;u++){s[u]=Math.exp(s[u]-mx);z+=s[u]}
  for(let i=0;i<hd;i++){let a=0;for(let u=0;u<=t;u++)a+=s[u]*qkv[u*3*D+2*D+h*hd+i];out[t*D+h*hd+i]=a/z}}
 return linear(out,T,D,P.get(pre+'.out_proj.weight').data,P.get(pre+'.out_proj.bias').data,D);
}
function block(x,T,pre,P,dff,nHeads){ // TransformerBlock (post-LN)
 const D=CONFIG.d,a=attention(x,T,pre+'.attn',P,nHeads),r1=new Float32Array(T*D);for(let i=0;i<r1.length;i++)r1[i]=x[i]+a[i];
 const n1=layerNorm(r1,T,D,P.get(pre+'.ln1.weight').data,P.get(pre+'.ln1.bias').data,CONFIG.eps);
 const h=linear(n1,T,D,P.get(pre+'.ff.0.weight').data,P.get(pre+'.ff.0.bias').data,dff);for(let i=0;i<h.length;i++)h[i]=gelu(h[i]);
 const f=linear(h,T,dff,P.get(pre+'.ff.3.weight').data,P.get(pre+'.ff.3.bias').data,D),r2=new Float32Array(T*D);for(let i=0;i<r2.length;i++)r2[i]=n1[i]+f[i];
 return layerNorm(r2,T,D,P.get(pre+'.ln2.weight').data,P.get(pre+'.ln2.bias').data,CONFIG.eps);
}
export class Trident{
 constructor(state){this.P=state.tensors}
 static async fromBytes(ptBytes){return new Trident(await loadStateDict(ptBytes))}
 // ids: array of ints; head: null (router-weighted mix) | 'code'|'math'|'research'. Returns {logits(last position), router}
 forward(ids,{head=null}={}){
  const P=this.P,D=CONFIG.d,T=ids.length;if(!T||T>CONFIG.maxSeq)throw new RangeError('1..'+CONFIG.maxSeq+' ids');
  const E=P.get('embed.weight').data,PE=P.get('pos_enc.pe').data;let x=new Float32Array(T*D);
  for(let t=0;t<T;t++){if(ids[t]<0||ids[t]>=CONFIG.vocab)throw new RangeError('id out of range');for(let i=0;i<D;i++)x[t*D+i]=E[ids[t]*D+i]+PE[t*D+i]}
  for(let l=0;l<CONFIG.layers;l++)x=block(x,T,'backbone.'+l,P,CONFIG.dff,CONFIG.heads);
  x=layerNorm(x,T,D,P.get('ln_back.weight').data,P.get('ln_back.bias').data,CONFIG.eps);
  const mean=new Float32Array(D);for(let t=0;t<T;t++)for(let i=0;i<D;i++)mean[i]+=x[t*D+i]/T;
  let h=linear(mean,1,D,P.get('router.proj.0.weight').data,P.get('router.proj.0.bias').data,64);for(let i=0;i<64;i++)h[i]=gelu(h[i]);
  const z=linear(h,1,64,P.get('router.proj.2.weight').data,P.get('router.proj.2.bias').data,3),mx=Math.max(...z),ex=Array.from(z,v=>Math.exp(v-mx)),sm=ex.reduce((a,b)=>a+b,0),router=ex.map(v=>v/sm);
  const one=name=>{let y=x;for(let l=0;l<CONFIG.headLayers;l++)y=block(y,T,'heads.'+name+'.layers.'+l,P,CONFIG.headDff,CONFIG.heads);
   y=layerNorm(y,T,D,P.get('heads.'+name+'.ln_out.weight').data,P.get('heads.'+name+'.ln_out.bias').data,CONFIG.eps);
   const last=y.subarray((T-1)*D,T*D);return linear(last,1,D,P.get('heads.'+name+'.lm_head.weight').data,null,CONFIG.vocab)};
  if(head){if(!CONFIG.headNames.includes(head))throw new RangeError('unknown head '+head);return {logits:one(head),router}}
  const logits=new Float32Array(CONFIG.vocab);CONFIG.headNames.forEach((n,i)=>{const l=one(n);for(let v=0;v<CONFIG.vocab;v++)logits[v]+=router[i]*l[v]});
  return {logits,router};
 }
 // temp<=0 => greedy. rand is injectable for repeatable tests.
 generate(ids,{maxNew=64,temp=0.8,topK=40,head=null,rand=Math.random}={}){
  const out=ids.slice();
  for(let n=0;n<maxNew;n++){
   const {logits}=this.forward(out.slice(-CONFIG.maxSeq),{head});
   let next;
   if(temp<=0){next=0;for(let v=1;v<logits.length;v++)if(logits[v]>logits[next])next=v}
   else{const l=Array.from(logits,v=>v/temp);let cut=-Infinity;if(topK){cut=[...l].sort((a,b)=>b-a)[Math.min(topK,l.length)-1]}
    const mx=Math.max(...l),p=l.map(v=>v<cut?0:Math.exp(v-mx)),s=p.reduce((a,b)=>a+b,0);let r=rand()*s;next=0;for(let v=0;v<p.length;v++){r-=p[v];if(r<=0){next=v;break}}}
   out.push(next);
  }
  return out;
 }
}
export const tokenize=(text,{maxLen=64}={})=>Array.from(String(text).slice(0,maxLen).toLowerCase(),c=>c.codePointAt(0)%CONFIG.vocab);
// Training used ord(c)%4096, so ids are the characters themselves (ids < 4096 are the code points). Printable output is the inverse for ids that are code points.
export const detokenize=ids=>ids.map(i=>String.fromCodePoint(i)).join('');
// Same as generate() but yields to the event loop between tokens so a phone's UI thread stays responsive (no worker is used).
Trident.prototype.generateAsync=async function(ids,{yieldMs=0,onToken=null,...opts}={}){
 let out=ids.slice();const n=opts.maxNew??64;
 for(let i=0;i<n;i++){out=this.generate(out,{...opts,maxNew:1});if(onToken)onToken(out[out.length-1],i);await new Promise(r=>setTimeout(r,yieldMs))}
 return out;
};
