// Thin adapter: Alexis's Bantu 6-bit token representation, kept as she wrote it.
// Lines 5-9 of her bridge.js (integrated/bantu/bridge.js.txt; the file itself also contains browser/DOM code, so it cannot be imported as a module) are copied VERBATIM below.
// A test checks that each function text below occurs unchanged in the reference copy.
// Her rule (README): each 6-bit token is "six independently simulated primitive positions grouped 3-2-1 (two bits per level)".
function encodeBytes(bytes){let out=[];for(const b of bytes){if(b<63)out.push(b);else out.push(63,b>>6,b&63)}return out}
function decodeBytes(codes){let out=[];for(let i=0;i<codes.length;i++){let v=codes[i];if(v===63){if(i+2>=codes.length)throw Error('Truncated escape');const hi=codes[++i],lo=codes[++i];if(hi>3||lo>63)throw Error('Invalid escape');v=(hi<<6)|lo;if(v<63)throw Error('Noncanonical escape')}if(v>255||v<0)throw Error('Invalid byte');out.push(v)}return Uint8Array.from(out)}
function bits(v){return v.toString(2).padStart(6,'0')}
function encodeText(text){const bytes=new TextEncoder().encode(text),codes=encodeBytes(bytes);return {bytes,codes,groups:codes.map((v,i)=>({id:i,value:v,bits:bits(v),levels:[bits(v).slice(0,2),bits(v).slice(2,4),bits(v).slice(4,6)]}))}}
function decodeText(codes){return new TextDecoder('utf-8',{fatal:true}).decode(decodeBytes(codes))}
export {encodeBytes,decodeBytes,bits,encodeText,decodeText};
export const VERBATIM_FUNCTIONS=Object.freeze([encodeBytes,decodeBytes,bits,encodeText,decodeText].map(f=>f.toString()));
// 3-2-1 packs of a token (two bits per level), as her groups[].levels
export const packsOf=code=>{const b=bits(code);return [b.slice(0,2),b.slice(2,4),b.slice(4,6)].map(s=>parseInt(s,2))};
