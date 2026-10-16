/**
 * Package the existing SynthAi Computer's *built* browser runtime into the
 * existing YOU-N-I-VERSE WebView assets, preserving both upstream sources.
 *
 * Usage:
 *   node android/stage-synthai-computer.mjs <SynthAi/public> <40-char-commit-sha>
 *
 * The resulting staged /assets/computer/ subtree is generated at build time;
 * its execution remains local in the existing WebViewAssetLoader HTTPS origin.
 * No server URLs, WebWorkers, LLMs, or CDN resources are required for ATO/Klein.
 */
import { cp, mkdir, rm, readFile, writeFile, readdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve, join, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const source = resolve(process.argv[2] || '');
const sha = process.argv[3];
if (!/^[a-f0-9]{40}$/i.test(sha || '')) throw new Error('Pinned 40-character SynthAi commit SHA required');
if (!process.argv[2] || !source.endsWith(sep + 'public'))
  throw new Error('Explicit SynthAi public/ directory required (run computer/build-web.mjs first)');
const target = fileURLToPath(new URL('./app/src/main/swarm-assets/computer/', import.meta.url));
await rm(target, { recursive: true, force: true });
await mkdir(target, { recursive: true });

const files = [
  'index.html', 'computer.css', 'computer-app.mjs', 'swarm-ui.mjs',
  'computer.html', 'computer-diagnostic.mjs', 'fitness-morph-ui.mjs',
];
for (const file of files) await cp(join(source, file), join(target, file));
await cp(join(source, 'computer-runtime'), join(target, 'computer-runtime'), { recursive: true });

const replaceExact = (body, oldText, newText, where) => {
  if (!body.includes(oldText)) throw new Error('Upstream Computer path changed: ' + where + ': ' + oldText);
  return body.replaceAll(oldText, newText);
};
const changes = {
  'index.html': [
    ['href="/computer.css"', 'href="./computer.css"'],
    ['src="/computer-app.mjs"', 'src="./computer-app.mjs"'],
    ['href="/computer.html"', 'href="./computer.html"'],
    ['<div class="statusline">', '<div class="statusline"><a class="pill" href="../index.html" aria-label="Return to YOU-N-I-VERSE Studio">← Studio</a>'],
  ],
  'computer-app.mjs': [
    ["from '/computer-runtime/", "from './computer-runtime/"],
    ["from '/swarm-ui.mjs'", "from './swarm-ui.mjs'"],
    ["from '/fitness-morph-ui.mjs'", "from './fitness-morph-ui.mjs'"],
  ],
  'computer.html': [['src="/computer-diagnostic.mjs"', 'src="./computer-diagnostic.mjs"']],
  'computer-diagnostic.mjs': [["from'/computer-runtime/", "from'./computer-runtime/"]],
};
for (const [name, transformations] of Object.entries(changes)) {
  let contents = await readFile(join(target, name), 'utf8');
  for (const [a,b] of transformations) contents = replaceExact(contents, a, b, name);
  if (/(?:href|src)=["']\/|from\s+["']\//.test(contents))
    throw new Error('Unrewritten root-origin dependency in ' + name);
  await writeFile(join(target, name), contents);
}

const inventory = [];
async function scan(folder, rel = '') {
  for (const item of await readdir(folder, { withFileTypes: true })) {
    const path = join(folder, item.name);
    const name = rel ? rel + '/' + item.name : item.name;
    if (item.isDirectory()) await scan(path, name);
    else {
      const bytes = await readFile(path);
      inventory.push({ path: name, bytes: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex') });
    }
  }
}
await scan(target);
if (!inventory.some(x => x.path.endsWith('ato-core/src/index.mjs')))
  throw new Error('Real recovered ATO Core was not packaged');
if (!inventory.some(x => x.path === 'computer-runtime/events/browser-event-emitter.mjs'))
  throw new Error('Native append-only IndexedDB event adapter was not packaged');
if (!inventory.some(x => x.path === 'computer-runtime/services/fitness-morph.mjs'))
  throw new Error('Real executable Simlish fitness adapter was not packaged');
if (!inventory.some(x => x.path === 'computer-runtime/donors/verified/simlish/frame-morph.mjs'))
  throw new Error('Original Simlish visual morph donor was not packaged');
if (!inventory.some(x => x.path === 'computer-runtime/donors/verified/simlish/swarm-fit.mjs'))
  throw new Error('Original Simlish CPU particle optimizer was not packaged');
if (!inventory.some(x => x.path === 'computer-runtime/services/native-klein-memory.mjs'))
  throw new Error('Native Klein memory adapter was not packaged');
if (inventory.some(x => /(?:node_modules|\.git\/)/.test(x.path)))
  throw new Error('Unexpected tool/development package copied to Android');
inventory.sort((a, b) => a.path.localeCompare(b.path));
const record = {
  manifest: 'synthai.swarm-android-local-assets.v1',
  upstream: 'justappgrabbin/SynthAi',
  upstreamCommit: sha,
  origin: 'https://appassets.androidplatform.net/assets/computer/index.html',
  files: inventory,
};
await writeFile(join(target, 'swarm-assets-manifest.json'), JSON.stringify(record, null, 2) + '\n');
console.log('Staged EXISTING SynthAi Computer ' + sha + ' into Android WebView assets: ' +
  inventory.length + ' files / ' + inventory.reduce((n, x) => n + x.bytes, 0) + ' bytes');
