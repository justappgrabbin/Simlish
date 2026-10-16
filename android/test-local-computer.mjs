/**
 * Verify the actual Android WebView asset layout through Chromium, using the
 * same /assets/ paths served by the existing WebViewAssetLoader.
 * The test server is static-only and binds to loopback: no app backend.
 */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { resolve, join, extname, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const repo = fileURLToPath(new URL('../', import.meta.url));
const studio = resolve(repo, 'you-n-i-verse');
const computer = resolve(repo, 'android/app/src/main/swarm-assets/computer');
const types = new Map(Object.entries({
  '.mjs': 'text/javascript', '.js': 'text/javascript', '.json': 'application/json',
  '.html': 'text/html', '.css': 'text/css', '.wasm': 'application/wasm',
  '.svg': 'image/svg+xml', '.png': 'image/png', '.jpg': 'image/jpeg',
  '.webp': 'image/webp', '.woff2': 'font/woff2',
}));
const server = createServer(async (req, res) => {
  const url = new URL(req.url, 'http://localhost');
  const path = decodeURIComponent(url.pathname);
  if (!path.startsWith('/assets/')) { res.writeHead(404).end(); return; }
  const nested = path.slice('/assets/'.length);
  const isComputer = nested.startsWith('computer/');
  const root = isComputer ? computer : studio;
  const local = isComputer ? nested.slice('computer/'.length) : nested;
  const file = resolve(root, local);
  if (!(file === root || file.startsWith(root + sep))) { res.writeHead(403).end(); return; }
  try {
    const bytes = await readFile(file);
    res.writeHead(200, { 'Content-Type': types.get(extname(file)) ?? 'application/octet-stream' });
    res.end(bytes);
  } catch {
    res.writeHead(404).end('Missing packaged asset: ' + local);
  }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const base = 'http://127.0.0.1:' + server.address().port;
const browser = await chromium.launch({ args: ['--no-sandbox'] });
const page = await browser.newPage({ viewport: { width: 420, height: 850 } });
const runtimeErrors = [];
const offDevice = [];
page.on('pageerror', error => runtimeErrors.push(error.message));
page.on('request', req => { if (!req.url().startsWith(base + '/')) offDevice.push(req.url()); });
try {
  // The Android host provides this existing JavaScript bridge only locally.
  await page.addInitScript(() => { window.NativeDownloads = {}; });
  let response = await page.goto(base + '/assets/index.html', { waitUntil: 'domcontentloaded' });
  assert.equal(response.status(), 200);
  const computerLink = page.locator('#synthaiComputerLauncher');
  assert.equal(await computerLink.isVisible(), true, 'existing studio exposes native Computer');
  await computerLink.click();
  await page.waitForURL(/\/assets\/computer\/index\.html/);
  await page.waitForFunction(() => Boolean(window.SynthAIComputer?.nativeKlein) &&
    document.getElementById('bootLabel')?.textContent === 'runtime online',
  null, { timeout: 25000 });
  await page.locator('[data-view="systems"]').click();
  await page.locator('#assembleKlein').click();
  await page.waitForFunction(() => document.getElementById('formationStatus')?.textContent === '3 executable');
  await page.locator('#runFormation').click();
  await page.waitForFunction(() => document.getElementById('formationOutput')?.textContent.includes('Recognized HAS_PROPERTY'));
  assert.match(await page.locator('#formationMemory').innerText(), /1 native lexical records/);
  const bottom = await page.locator('.rail').evaluate(element => element.getBoundingClientRect().bottom);
  assert.equal(Math.round(bottom), 850, 'existing Computer bottom navigation works');
  await page.route('**/*', route => route.abort());
  await page.locator('#formationText').fill('owl is patient');
  await page.locator('#runFormation').click();
  await page.waitForFunction(() => document.getElementById('formationMemory')?.textContent.includes('2 native lexical records'));
  await page.unroute('**/*');
  // Simulates Android WebView process recreation on the exact same asset URL.
  await page.reload({ waitUntil: 'load' });
  await page.waitForFunction(() => Boolean(window.SynthAIComputer?.nativeKlein) &&
    document.getElementById('bootLabel')?.textContent === 'runtime online',
  null, { timeout: 25000 });
  await page.locator('[data-view="systems"]').click();
  assert.match(await page.locator('#formationMemory').innerText(), /2 native lexical records/);
  assert.equal(await page.locator('#formationStatus').innerText(), '3 executable');
  const events = await page.evaluate(() => window.SynthAIComputer.formationHistory('klein:autoling+diseminer'));
  assert.ok(events.some(e => e.input.action === 'execute'));
  assert.deepEqual(runtimeErrors, [], 'no uncaught module or runtime errors in embedded Computer');
  assert.deepEqual(offDevice, [], 'all required modules and requests remain within packaged assets');
  console.log('Android asset-layout browser integration passed: studio > Computer > native ATO execution > offline operation > persisted restart');
} finally {
  await page.close();
  await browser.close();
  await new Promise(resolve => server.close(resolve));
}
