// Dependency-free ZIP reader/writer (browser WebView and Node 18+). Reads stored+deflate; writes stored.
const te=new TextEncoder(),td=new TextDecoder();
const CRC=(()=>{const t=new Uint32Array(256);for(let n=0;n<256;n++){let c=n;for(let k=0;k<8;k++)c=c&1?0xedb88320^(c>>>1):c>>>1;t[n]=c>>>0}return t})();
export function crc32(b){let c=0xffffffff;for(let i=0;i<b.length;i++)c=CRC[(c^b[i])&255]^(c>>>8);return (c^0xffffffff)>>>0}
async function inflateRaw(data){const s=new Blob([data]).stream().pipeThrough(new DecompressionStream('deflate-raw'));return new Uint8Array(await new Response(s).arrayBuffer())}
export function isZip(b){return b.length>=4&&b[0]===0x50&&b[1]===0x4b&&(b[2]===3||b[2]===5)&&(b[3]===4||b[3]===6)}
export async function readZip(bytes){
 const b=bytes instanceof Uint8Array?bytes:new Uint8Array(bytes),dv=new DataView(b.buffer,b.byteOffset,b.byteLength);
 let e=-1;for(let i=b.length-22;i>=Math.max(0,b.length-65557);i--)if(dv.getUint32(i,true)===0x06054b50){e=i;break}
 if(e<0)throw new Error('Not a readable zip (no end record)');
 const n=dv.getUint16(e+10,true);let p=dv.getUint32(e+16,true);const out=[];
 for(let i=0;i<n;i++){
  if(dv.getUint32(p,true)!==0x02014b50)throw new Error('Corrupt zip directory');
  const method=dv.getUint16(p+10,true),csize=dv.getUint32(p+20,true),nl=dv.getUint16(p+28,true),xl=dv.getUint16(p+30,true),cl=dv.getUint16(p+32,true),off=dv.getUint32(p+42,true);
  const path=td.decode(b.subarray(p+46,p+46+nl));p+=46+nl+xl+cl;
  if(path.endsWith('/'))continue;
  const lnl=dv.getUint16(off+26,true),lxl=dv.getUint16(off+28,true),raw=b.subarray(off+30+lnl+lxl,off+30+lnl+lxl+csize);
  if(method===0)out.push({path,data:raw.slice()});else if(method===8)out.push({path,data:await inflateRaw(raw)});else throw new Error('Unsupported zip method '+method+' for '+path);
 }
 return out;
}
export function writeZip(files){
 const parts=[],central=[];let offset=0;
 for(const f of files){
  const name=te.encode(f.path),data=typeof f.data==='string'?te.encode(f.data):f.data,crc=crc32(data);
  const h=new DataView(new ArrayBuffer(30));h.setUint32(0,0x04034b50,true);h.setUint16(4,20,true);h.setUint16(6,0x0800,true);h.setUint32(14,crc,true);h.setUint32(18,data.length,true);h.setUint32(22,data.length,true);h.setUint16(26,name.length,true);
  parts.push(new Uint8Array(h.buffer),name,data);
  const c=new DataView(new ArrayBuffer(46));c.setUint32(0,0x02014b50,true);c.setUint16(4,20,true);c.setUint16(6,20,true);c.setUint16(8,0x0800,true);c.setUint32(16,crc,true);c.setUint32(20,data.length,true);c.setUint32(24,data.length,true);c.setUint16(28,name.length,true);c.setUint32(42,offset,true);
  central.push(new Uint8Array(c.buffer),name);offset+=30+name.length+data.length;
 }
 const csize=central.reduce((a,x)=>a+x.length,0),end=new DataView(new ArrayBuffer(22));end.setUint32(0,0x06054b50,true);end.setUint16(8,files.length,true);end.setUint16(10,files.length,true);end.setUint32(12,csize,true);end.setUint32(16,offset,true);
 const all=[...parts,...central,new Uint8Array(end.buffer)],out=new Uint8Array(all.reduce((a,x)=>a+x.length,0));let o=0;for(const x of all){out.set(x,o);o+=x.length}return out;
}
