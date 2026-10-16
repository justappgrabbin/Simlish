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
  // Integration across the real on-device Studio project database:
  // saved character → ATO fitness selection → append-only visible layer → Studio.
  const studioOriginal = 'data:image/png;base64,' + photos.from;
  await page.evaluate(async original => {
    const { saveProject, readProject } = await import('/assets/project-store.mjs');
    const controls = {
      zoom: 1, rotation: 0, blend: 1, faceX: .46, faceY: .105,
      faceSize: .30, influence: .5,
    };
    await saveProject({
      type: 'you-n-i-verse-project', version: 1,
      savedAt: new Date().toISOString(), world: 'cosmic',
      source: 'on-device-test-pixel-art', controls,
      pan: { x: 0, y: 0 }, photo: null, body: original, sprites: null,
      morphedImage: null, morphedSprites: null,
    });
    const persisted = await readProject();
    if (persisted.body !== original) throw new Error('Original Studio character project not retained');
  }, studioOriginal);
  await page.locator('#fitnessMorphLab summary').click();
  assert.equal(await page.locator('#studioCharacterControls').evaluate(e => e.hidden), false,
    'native Computer exposes original Studio bridge');
  await page.locator('#useStudioCharacter').click();
  await page.waitForFunction(() =>
    document.getElementById('studioCharacterStatus').textContent.includes('character loaded locally'));
  await page.locator('#fitnessTarget').setInputFiles({
    name: 'studio-reference.png', mimeType: 'image/png',
    buffer: Buffer.from(photos.to, 'base64'),
  });
  await page.locator('#evolveLighting').click();
  await page.waitForFunction(() => /^(Accepted:|Rejected:)/.test(
    document.getElementById('fitnessStatus')?.textContent || ''),
    null, { timeout: 40000 });
  const evolutionary = await page.evaluate(() =>
    window.SynthAIComputer.fitnessMorph.evolutionHistory());
  assert.equal(evolutionary.length, 1);
  assert.equal(evolutionary[0].result.candidates.length, 2);
  assert.equal(evolutionary[0].result.accepted, true,
    'Observed character-target pixels must produce a measurable improvement');
  assert.match(await page.locator('#fitnessEvidence').innerText(), /candidate selected/);
  await page.locator('#applyStudioFitness').click();
  await page.waitForFunction(() =>
    document.getElementById('studioCharacterStatus').textContent.includes('Original artwork'));
  const saved = await page.evaluate(async () => {
    const { readProject } = await import('/assets/project-store.mjs');
    const project = await readProject();
    return {
      body: project.body, count: project.fitnessMorphLayers.length,
      active: project.activeFitnessMorphId,
      layers: project.fitnessMorphLayers.map(layer => ({
        id: layer.id, sourceLayerId: layer.sourceLayerId,
        image: layer.image, revision: layer.receipt.activeRevision,
      })),
    };
  });
  assert.equal(saved.body, studioOriginal, 'original character artwork was not overwritten');
  assert.equal(saved.count, 1);
  assert.equal(saved.layers[0].id, saved.active);
  assert.equal(saved.layers[0].sourceLayerId, null);
  assert.match(saved.layers[0].revision, /^formation:simlish:lighting-active@/);
  assert.notEqual(saved.layers[0].image, saved.body, 'accepted character layer is a changed bitmap');
  const privacy = await page.evaluate(async () => {
    const all = await window.SynthAIComputer.eventEmitter.readAll();
    return all.some(event => JSON.stringify(event).includes('data:image'));
  });
  assert.equal(privacy, false, 'no portrait data URLs in Computer append-only event records');

  await page.locator('#returnToStudioFitness').click();
  await page.waitForURL(/\/assets\/index\.html/);
  await page.waitForFunction(() =>
    document.getElementById('fitnessMorphStatus')?.textContent.includes('Fitted expression visible'),
    null, { timeout: 20000 });
  assert.equal(await page.locator('#toggleFitnessMorph').isVisible(), true);
  const imageChecksum = async () => page.locator('#scene').evaluate(surface => {
    const ctx = surface.getContext('2d');
    const samples = ctx.getImageData(0, 0, surface.width, surface.height).data;
    let sum = 0;
    for (let y = 100; y < 850; y += 11) for (let x = 170; x < 550; x += 11) {
      const i = (y * surface.width + x) * 4;
      sum += samples[i] * 3 + samples[i + 1] * 5 + samples[i + 2] * 7;
    }
    return sum;
  });
  const acceptedCanvasChecksum = await imageChecksum();
  await page.locator('#toggleFitnessMorph').click();
  await page.waitForFunction(() =>
    document.getElementById('fitnessMorphStatus').textContent.includes('Original appearance visible'));
  const originalCanvasChecksum = await imageChecksum();
  assert.notEqual(acceptedCanvasChecksum, originalCanvasChecksum,
    'Studio canvas must visibly change when accepted character appearance is toggled');
  await page.waitForFunction(async () => {
    const { readProject } = await import('/assets/project-store.mjs');
    const p = await readProject();
    return p?.activeFitnessMorphId == null && p?.fitnessMorphLayers.length === 1;
  }, null, { timeout: 12000 });
  const persistedAfterToggle = await page.evaluate(async () => {
    const { readProject } = await import('/assets/project-store.mjs');
    const p = await readProject();
    return { source: p.body, count: p.fitnessMorphLayers.length };
  });
  assert.equal(persistedAfterToggle.source, studioOriginal);
  assert.equal(persistedAfterToggle.count, 1);
  console.log('Original Studio character roundtrip passed: saved bitmap → ATO fitness-evolved graph → appended real image layer → visible reversible Studio rendering');
  assert.deepEqual(runtimeErrors, [], 'no uncaught module or runtime errors in embedded Computer');
  assert.deepEqual(offDevice, [], 'all required modules and requests remain within packaged assets');
  console.log('Android asset-layout browser integration passed: studio > Computer > native ATO execution > offline operation > persisted restart');
} finally {
  await page.close();
  await browser.close();
  await new Promise(resolve => server.close(resolve));
}
