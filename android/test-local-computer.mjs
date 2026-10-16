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
  // Actual pixel inputs and actual donor math, through Android's packaged
  // assets path. No generated decorative output is counted as morphing.
  const photos = await page.evaluate(() => {
    function render(offset) {
      const canvas = document.createElement('canvas');
      canvas.width = 96; canvas.height = 128;
      const ctx = canvas.getContext('2d'), img = ctx.createImageData(96, 128);
      for (let y = 0; y < 128; y++) for (let x = 0; x < 96; x++) {
        const dx = 2 * x / 95 - 1, dy = 2 * y / 127 - 1;
        const disc = Math.max(0, 65 - Math.hypot(x - 45 - offset * 2, y - 62) * 2);
        const p = (y * 96 + x) * 4;
        const base = [90 + disc + x * .3, 115 + disc / 2 + y * .3, 145 + disc / 3];
        const light = offset ? [11 + 13 * dx + 7 * dy, 6 - 8 * dx + 12 * dy, -4 + 6 * dx - 7 * dy] : [0, 0, 0];
        for (let channel = 0; channel < 3; channel++)
          img.data[p + channel] = Math.max(0, Math.min(255, base[channel] + light[channel]));
        img.data[p + 3] = 255;
      }
      ctx.putImageData(img, 0, 0);
      return canvas.toDataURL('image/png').split(',')[1];
    }
    return { from: render(0), to: render(1) };
  });
  await page.locator('#fitnessMorphLab summary').click();
  await page.locator('#fitnessSource').setInputFiles({
    name: 'source.png', mimeType: 'image/png', buffer: Buffer.from(photos.from, 'base64'),
  });
  await page.locator('#fitnessTarget').setInputFiles({
    name: 'target.png', mimeType: 'image/png', buffer: Buffer.from(photos.to, 'base64'),
  });
  await page.locator('#runSwarmFit').click();
  await page.waitForFunction(() => /^(Accepted:|Not accepted:)/.test(document.getElementById('fitnessStatus').textContent),
    null, { timeout: 35000 });
  assert.match(await page.locator('#fitnessEvidence').innerText(), /held-out error/);
  await page.locator('#runFrameMorph').click();
  await page.waitForFunction(() => document.getElementById('fitnessStatus').textContent.includes('Executable frame morph rendered'),
    null, { timeout: 35000 });
  assert.match(await page.locator('#fitnessEvidence').innerText(), /native Simlish mesh vertices/);
  const renderedPixel = await page.locator('#fitnessPreview').evaluate(c =>
    Array.from(c.getContext('2d').getImageData(60, 60, 1, 1).data));
  assert.equal(renderedPixel[3], 255);
  assert.equal((await page.evaluate(() => window.SynthAIComputer.fitnessMorph.history())).length, 1);
  const mountedFitness = await page.evaluate(() => window.SynthAIComputer.formations.list().map(f => f.id));
  assert.ok(mountedFitness.includes('simlish:swarm-lighting'));
  assert.ok(mountedFitness.includes('simlish:frame-geometry'));
  await page.reload({ waitUntil: 'load' });
  await page.waitForFunction(() => window.SynthAIComputer?.fitnessMorph &&
    document.getElementById('bootLabel')?.textContent === 'runtime online');
  assert.equal((await page.evaluate(() => window.SynthAIComputer.fitnessMorph.history())).length, 1);
  assert.ok((await page.evaluate(() => window.SynthAIComputer.formations.list().map(f => f.id)))
    .includes('simlish:frame-geometry'), 'executable frame morph persisted across native asset reload');
  assert.deepEqual(runtimeErrors, [], 'no uncaught module or runtime errors in embedded Computer');
  assert.deepEqual(offDevice, [], 'all required modules and requests remain within packaged assets');
  console.log('Android asset-layout browser integration passed: studio > Computer > native ATO execution > offline operation > persisted restart');
} finally {
  await page.close();
  await browser.close();
  await new Promise(resolve => server.close(resolve));
}
