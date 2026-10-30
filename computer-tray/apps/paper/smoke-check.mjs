/**
 * Offline smoke checks for Paper sovereign fixes.
 * Run: node smoke-check.mjs
 */
import { detectArtifactKind, normalizeArtifact } from './automata/artifact-normalizer.mjs';
import { AutomataIngestionExecutionKernel } from './automata/ingestion-execution-kernel.mjs';
import { MODEL_ID, LEGACY_MODEL_ID, MAX_TOKENS, isReady } from './neural.mjs';
import { readFileSync } from 'fs';

if (!globalThis.structuredClone) {
  globalThis.structuredClone = (v) => JSON.parse(JSON.stringify(v));
}

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

// Mirror of app.js infer for structural check
function infer(desc) {
  const d = String(desc || '');
  let list = ['Home'];
  if (/garden|plant|flower|landscape|botanic/i.test(d)) list.push('Garden', 'Visit', 'About');
  if (/portfolio|gallery|photograph|photo|work|artist|design/i.test(d)) list.push('Work', 'Gallery', 'About', 'Contact');
  if (/shop|store|product|ecommerce|sell/i.test(d)) list.push('Products', 'Cart', 'About', 'Contact');
  if (/blog|article|journal|writing|newsletter/i.test(d)) list.push('Articles', 'About');
  if (/docs|guide|manual|reference|documentation/i.test(d)) list = ['Overview', 'Guide', 'Reference', 'About'];
  if (/restaurant|cafe|food|menu|kitchen|bakery/i.test(d)) list.push('Menu', 'About', 'Contact');
  if (/music|band|album|concert|podcast/i.test(d)) list.push('Music', 'About', 'Contact');
  if (/event|wedding|conference|meetup/i.test(d)) list.push('Events', 'About', 'Contact');
  if (/school|course|learn|class|workshop/i.test(d)) list.push('Courses', 'About', 'Contact');
  if (/about|profile|bio/i.test(d)) list.push('About');
  if (/contact|booking|message|hire|email/i.test(d)) list.push('Contact');
  if (list.length === 1) {
    if (/\b(site|page|web|project)\b/i.test(d)) list.push('About', 'Contact');
    else list.push('About');
  }
  return [...new Set(list)];
}

function stripTags(html) {
  return String(html || '')
    .replace(/<script[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style[\s\S]*?<\/style>/gi, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/\s+/g, ' ')
    .trim();
}

const results = [];

// A. Classification
assert(detectArtifactKind({ name: 'page.html', mime: 'text/html', text: '<html><body>Hi</body></html>' }) === 'html', 'html kind');
assert(detectArtifactKind({ name: 'note.md', text: '# Title\nHello' }) === 'markdown', 'markdown kind');
assert(detectArtifactKind({ name: 'theme.css', mime: 'text/css', text: 'body{color:red}' }) === 'css', 'css kind');
assert(detectArtifactKind({ name: 'pic.png', mime: 'image/png', bytes: new Uint8Array([137, 80, 78, 71]) }) === 'image', 'image kind');
assert(detectArtifactKind({ name: 'app.js', text: 'console.log(1)' }) === 'javascript', 'js kind');
assert(detectArtifactKind({ name: 'data.json', text: '{"a":1}' }) === 'json', 'json kind');
assert(detectArtifactKind({ name: 'plain.txt', text: 'hello' }) === 'text', 'text kind');
results.push('A classification ok');

// B. Kernel execute paths
const kernel = new AutomataIngestionExecutionKernel();
const html = await kernel.process(
  { name: 'about.html', content: '<!doctype html><html><head><title>About Us</title></head><body><h1>About</h1><p>Hello garden.</p></body></html>', type: 'text/html' },
  { execute: true },
);
assert(html.artifact.kind === 'html', `expected html got ${html.artifact.kind}`);
assert(html.execution.path !== 'compiled-without-execution', `html still without-execution: ${html.execution.path}`);
assert(html.execution.ok === true, 'html execution not ok');
results.push(`B html execute path=${html.execution.path}`);

const json = await kernel.process(
  { name: 'meta.json', content: '{"ok":true,"n":3}' },
  { execute: true },
);
assert(json.artifact.kind === 'json', 'json kind from kernel');
assert(json.execution.ok === true, 'json parse ok');
assert(json.execution.path === 'browser-native-json', `json path ${json.execution.path}`);
results.push('B json execute ok');

const jsPrep = await kernel.process(
  { name: 'tool.js', content: 'const x = 1 + 1; x;' },
  { execute: false },
);
assert(jsPrep.artifact.kind === 'javascript', 'js kind');
assert(jsPrep.execution.path === 'compiled-without-execution', 'js compile-only on upload');
results.push('B js compile-without-execution on upload ok');

// Materialize helpers
const body = stripTags('<html><head><title>X</title></head><body><h1>Hello</h1><p>World &amp; friends</p></body></html>');
assert(body.includes('Hello') && body.includes('World & friends'), `stripTags failed: ${body}`);
results.push('B stripTags ok');

// C. Infer
const garden = infer('A small garden site');
assert(garden.includes('Home') && garden.includes('Garden'), `garden pages: ${garden}`);
assert(garden.length >= 3, `garden too anemic: ${garden}`);
const portfolio = infer('A photographer portfolio with gallery and contact');
assert(portfolio.includes('Work') || portfolio.includes('Gallery'), `portfolio: ${portfolio}`);
assert(portfolio.includes('Contact'), `portfolio missing contact: ${portfolio}`);
const shop = infer('An online shop for handmade products');
assert(shop.includes('Products'), `shop: ${shop}`);
results.push(`C infer garden=${garden.join('|')} portfolio=${portfolio.join('|')} shop=${shop.join('|')}`);

// F. Neural module wiring (no GPU download in Node)
assert(MODEL_ID === 'TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC', 'MODEL_ID');
assert(LEGACY_MODEL_ID === 'TinyLlama-1.1B-Chat-v1.0-q4f16_1', 'LEGACY_MODEL_ID');
assert(MAX_TOKENS === 256, 'MAX_TOKENS');
assert(isReady() === false, 'engine should start unloaded');
const appSrc = readFileSync(new URL('./app.js', import.meta.url), 'utf8');
assert(appSrc.includes("['neural','🧠 Neural']"), 'Neural nav tab');
assert(appSrc.includes('load-neural'), 'load-neural action');
assert(appSrc.includes('run-neural'), 'run-neural action');
assert(appSrc.includes('Draft with TinyLlama'), 'Draft with TinyLlama');
assert(appSrc.includes('max_tokens:NEURAL_MAX_TOKENS') || appSrc.includes('NEURAL_MAX_TOKENS'), 'max_tokens wired');
results.push('F neural wiring ok');

console.log(JSON.stringify({ ok: true, results }, null, 2));
