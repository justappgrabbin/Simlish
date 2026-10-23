import test from 'node:test';import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';import path from 'node:path';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
test('source modules remain byte-identical to manifest',async()=>{
 const manifest=JSON.parse(await readFile(path.join(root,'SOURCE-MANIFEST.json'),'utf8'));
 assert.ok(manifest.modules.length>=15);
 for(const f of manifest.modules){const actual=createHash('sha256').update(await readFile(path.join(root,f.path))).digest('hex');assert.equal(actual,f.sha256,f.path)}
});
