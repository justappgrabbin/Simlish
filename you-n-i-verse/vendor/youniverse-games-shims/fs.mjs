// FILLED IN, no source: in-memory fs for her youniverse games in the browser (see README.md).
const KEY='youniverse.fs.v1',files=new Map(),dirs=new Set(['/']);
const ls=globalThis.localStorage;
try{const s=ls&&JSON.parse(ls.getItem(KEY)||'null');if(s){for(const [k,v] of s.f)files.set(k,v);for(const d of s.d)dirs.add(d)}}catch{}
const persist=()=>{try{ls&&ls.setItem(KEY,JSON.stringify({f:[...files].filter(([k])=>k.endsWith('.json')),d:[...dirs]}))}catch{}};
const norm=p=>String(p).replace(/\/+/g,'/').replace(/\/$/,'')||'/';
const enoent=p=>Object.assign(new Error('ENOENT: no such file or directory, '+p),{code:'ENOENT'});
export function existsSync(p){p=norm(p);return files.has(p)||dirs.has(p)}
export function mkdirSync(p){p=norm(p);const parts=p.split('/');for(let i=1;i<=parts.length;i++)dirs.add(parts.slice(0,i).join('/')||'/')}
export function writeFileSync(p,data){p=norm(p);mkdirSync(p.slice(0,p.lastIndexOf('/'))||'/');files.set(p,typeof data==='string'?data:new TextDecoder().decode(data));persist()}
export function readFileSync(p){p=norm(p);if(!files.has(p))throw enoent(p);return files.get(p)}
export function readdirSync(p){p=norm(p);const pre=p==='/'?'/':p+'/',out=new Set();for(const k of [...files.keys(),...dirs])if(k.startsWith(pre)&&k!==p)out.add(k.slice(pre.length).split('/')[0]);return [...out].sort()}
export function renameSync(a,b){a=norm(a);if(!files.has(a))throw enoent(a);files.set(norm(b),files.get(a));files.delete(a);persist()}
export function rmSync(p){files.delete(norm(p));persist()}
export const __reset=()=>{files.clear();dirs.clear();dirs.add('/');try{ls&&ls.removeItem(KEY)}catch{}};
export default {existsSync,mkdirSync,writeFileSync,readFileSync,readdirSync,renameSync,rmSync};
