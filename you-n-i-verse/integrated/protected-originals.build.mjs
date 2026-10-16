// Writes protected-originals.json: the sha256 of every original the self-assembling loop must never modify (her reference copies, the primitive layers, her model weights).
// Run: node integrated/protected-originals.build.mjs   (pure Node built-ins; build tool, not shipped logic)
import {readFileSync,writeFileSync,readdirSync,statSync} from 'node:fs';
import {join,dirname} from 'node:path';import {fileURLToPath} from 'node:url';
import {sha256} from '../sha256.mjs';
const ROOT=join(dirname(fileURLToPath(import.meta.url)),'..');
const walk=d=>readdirSync(join(ROOT,d)).flatMap(n=>statSync(join(ROOT,d,n)).isDirectory()?walk(d+'/'+n):[d+'/'+n]);
const paths=['integrated/primitive-functions.json','integrated/primitive-attributes.json','integrated/primitive-sources.json','models/trident_syntia.pt','models/trident_vocab.json',...walk('integrated/reference')].sort();
const files=Object.fromEntries(paths.map(p=>[p,sha256(new Uint8Array(readFileSync(join(ROOT,p))))]));
writeFileSync(join(ROOT,'integrated/protected-originals.json'),JSON.stringify({note:'sha256 of originals the swarm loop may never modify. The constitution text is protected in code (swarm-loop.mjs CONSTITUTION). Regenerate only when an original is intentionally replaced, and say so in the commit.',files},null,1)+'\n');
console.log(Object.keys(files).length+' originals recorded');
