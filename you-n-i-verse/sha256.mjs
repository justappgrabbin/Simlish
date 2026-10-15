// Small synchronous SHA-256 (works in browser WebView and Node) for NSNF payload_hash / artifact_hash fields.
const K=new Uint32Array([0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2]);
export function sha256(input){
 const data=typeof input==='string'?new TextEncoder().encode(input):input instanceof Uint8Array?input:new Uint8Array(input);
 const l=data.length,total=((l+9+63)>>6)<<6,buf=new Uint8Array(total);buf.set(data);buf[l]=0x80;
 const dv=new DataView(buf.buffer);dv.setUint32(total-8,Math.floor(l/0x20000000));dv.setUint32(total-4,(l<<3)>>>0);
 const h=new Uint32Array([0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19]),w=new Uint32Array(64);
 const rr=(x,n)=>(x>>>n)|(x<<(32-n));
 for(let o=0;o<total;o+=64){
  for(let i=0;i<16;i++)w[i]=dv.getUint32(o+i*4);
  for(let i=16;i<64;i++){const a=rr(w[i-15],7)^rr(w[i-15],18)^(w[i-15]>>>3),b=rr(w[i-2],17)^rr(w[i-2],19)^(w[i-2]>>>10);w[i]=(w[i-16]+a+w[i-7]+b)>>>0}
  let [a,b,c,d,e,f,g,hh]=h;
  for(let i=0;i<64;i++){const s1=rr(e,6)^rr(e,11)^rr(e,25),ch=(e&f)^(~e&g),t1=(hh+s1+ch+K[i]+w[i])>>>0,s0=rr(a,2)^rr(a,13)^rr(a,22),mj=(a&b)^(a&c)^(b&c),t2=(s0+mj)>>>0;hh=g;g=f;f=e;e=(d+t1)>>>0;d=c;c=b;b=a;a=(t1+t2)>>>0}
  h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;
 }
 return [...h].map(x=>x.toString(16).padStart(8,'0')).join('');
}
