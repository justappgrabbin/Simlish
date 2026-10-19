// FILLED IN, no source: minimal global Buffer for her code in the browser (isBuffer, from, byteLength, read*LE, write, slice).
class Buf extends Uint8Array{
 static isBuffer(x){return x instanceof Buf}
 static from(x){if(typeof x==='string'){const b=new TextEncoder().encode(x);const o=new Buf(b.length);o.set(b);return o}const o=new Buf(x.length??x.byteLength);o.set(x instanceof ArrayBuffer?new Uint8Array(x):x);return o}
 static alloc(n){return new Buf(n)}
 static byteLength(s){return typeof s==='string'?new TextEncoder().encode(s).length:s.byteLength}
 dv(){return new DataView(this.buffer,this.byteOffset,this.byteLength)}
 readUInt32LE(o){return this.dv().getUint32(o,true)}readInt32LE(o){return this.dv().getInt32(o,true)}readUInt16LE(o){return this.dv().getUint16(o,true)}
 readUInt32BE(o){return this.dv().getUint32(o,false)}
 writeUInt32LE(v,o){this.dv().setUint32(o,v,true)}writeInt32LE(v,o){this.dv().setInt32(o,v,true)}writeUInt16LE(v,o){this.dv().setUint16(o,v,true)}
 write(s,o=0){const b=new TextEncoder().encode(s);this.set(b,o);return b.length}
 toString(enc){return new TextDecoder().decode(this)}
}
if(typeof globalThis.Buffer==='undefined')globalThis.Buffer=Buf;
export const Buffer=globalThis.Buffer;
