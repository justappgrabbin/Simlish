// Local neural action head over fixed graph message-passing features.
// This is not the full PyTorch QuantumHDNet port.
export class LocalGraphModel {
 constructor(seed=17){let s=seed;const rand=()=>{s=(Math.imul(s,1664525)+1013904223)>>>0;return s/2**32-.5};this.w=Array.from({length:3},()=>Array.from({length:4},rand));this.b=[0,0,0];this.steps=0;}
 features(bits,edges){if(bits.length!==64)throw Error('Expected 64 gate features');let x=bits.map(Number);for(let l=0;l<3;l++){const next=x.slice(),count=Array(64).fill(1);for(const [a,b] of edges){next[a]+=x[b];next[b]+=x[a];count[a]++;count[b]++;}x=next.map((v,i)=>Math.tanh(v/count[i]));}return [bits.reduce((a,b)=>a+b,0)/64,x.reduce((a,b)=>a+b,0)/64,x[0],x[7]];}
 predict(features){const logits=this.w.map((row,k)=>row.reduce((s,v,i)=>s+v*features[i],this.b[k]));const max=Math.max(...logits),e=logits.map(v=>Math.exp(v-max)),sum=e.reduce((a,b)=>a+b);return e.map(v=>v/sum);}
 train(examples,epochs=300,lr=.2){for(let e=0;e<epochs;e++)for(const {features,label} of examples){const p=this.predict(features);for(let k=0;k<3;k++){const d=p[k]-Number(k===label);for(let j=0;j<4;j++)this.w[k][j]-=lr*d*features[j];this.b[k]-=lr*d;}this.steps++;}return this;}
 export(){return {version:1,architecture:'fixed-3-round-graph/trainable-softmax-head',weights:this.w,bias:this.b,steps:this.steps,labels:['observe','transform','rest']};}
 static restore(data){if(data.version!==1||data.weights?.length!==3||data.weights.some(r=>r.length!==4||r.some(v=>!Number.isFinite(v)))||data.bias?.length!==3||data.bias.some(v=>!Number.isFinite(v)))throw Error('Invalid model');const m=new LocalGraphModel();m.w=data.weights.map(r=>r.slice());m.b=data.bias.slice();m.steps=data.steps;return m;}
}
