// node scripts-build-apps.mjs  -> writes apps/apps.json (adds file lists). Run after editing apps/ folders.
import fs from 'node:fs'; import path from 'node:path';
const root=path.dirname(new URL(import.meta.url).pathname);
const src=JSON.parse(fs.readFileSync(path.join(root,'apps/apps.src.json')));
const walk=(d,b=d)=>fs.existsSync(d)?fs.readdirSync(d,{withFileTypes:true}).flatMap(e=>e.isDirectory()?walk(path.join(d,e.name),b):[path.relative(b,path.join(d,e.name))]):[];
const out=src.map(a=>({...a,files:walk(path.join(root,a.folder)).filter(f=>/\.(html?|m?js|css|json|md|txt|svg)$/.test(f)).slice(0,400)}));
fs.writeFileSync(path.join(root,'apps/apps.json'),JSON.stringify(out,null,1));console.log(out.map(a=>`${a.id}: ${a.files.length} files`).join('\n'));
