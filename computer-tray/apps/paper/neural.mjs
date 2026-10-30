/**
 * Paper Neural — TinyLlama in the browser via WebLLM. No backend. No API key.
 *
 * Alexis's original Paper-synth path:
 *   <script src="https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm/dist/index.min.js"></script>
 *   engine = await webllm.createEngine("TinyLlama-1.1B-Chat-v1.0-q4f16_1");
 *   await engine.chat.completions.create({ messages:[{role:'user',content:prompt}], max_tokens:256 });
 *
 * Today's CDN dropped dist/index.min.js; CreateMLCEngine + "-MLC" model id is the live equivalent.
 * We still try her createEngine path first, then fall through.
 */

export const LEGACY_MODEL_ID = 'TinyLlama-1.1B-Chat-v1.0-q4f16_1';
export const MODEL_ID = 'TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC';
export const MAX_TOKENS = 256;

const WEBLLM_ESM = 'https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.86/+esm';
const WEBLLM_UMD_CANDIDATES = [
  'https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm/dist/index.min.js',
  'https://unpkg.com/@mlc-ai/web-llm/dist/index.min.js',
];

let engine = null;
let loadPromise = null;
let lastModelId = null;

export function getEngine() {
  return engine;
}

export function isReady() {
  return !!engine;
}

export function loadedModelId() {
  return lastModelId;
}

function formatProgress(report) {
  if (report == null) return 'Loading TinyLlama…';
  if (typeof report === 'string') return report;
  const text = report.text || report.msg || 'Loading TinyLlama…';
  if (typeof report.progress === 'number' && Number.isFinite(report.progress)) {
    return `${text} (${Math.round(report.progress * 100)}%)`;
  }
  return text;
}

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const existing = [...document.querySelectorAll('script[data-paper-webllm]')].find(el => el.dataset.paperWebllm === src);
    if (existing) {
      if (existing.dataset.loaded === '1') resolve();
      else {
        existing.addEventListener('load', () => resolve(), { once: true });
        existing.addEventListener('error', () => reject(new Error(`Failed to load ${src}`)), { once: true });
      }
      return;
    }
    const s = document.createElement('script');
    s.src = src;
    s.async = true;
    s.dataset.paperWebllm = src;
    s.onload = () => {
      s.dataset.loaded = '1';
      resolve();
    };
    s.onerror = () => reject(new Error(`Failed to load ${src}`));
    document.head.appendChild(s);
  });
}

function globalWebLLM() {
  return globalThis.webllm || globalThis.webLLM || null;
}

async function tryLegacyCreateEngine(onProgress) {
  const existing = globalWebLLM();
  if (typeof existing?.createEngine === 'function') {
    onProgress?.(formatProgress({ text: `createEngine("${LEGACY_MODEL_ID}")…`, progress: 0 }));
    const eng = await existing.createEngine(LEGACY_MODEL_ID, {
      initProgressCallback: (r) => onProgress?.(formatProgress(r)),
    });
    lastModelId = LEGACY_MODEL_ID;
    return eng;
  }
  for (const src of WEBLLM_UMD_CANDIDATES) {
    try {
      await loadScript(src);
      const w = globalWebLLM();
      if (typeof w?.createEngine === 'function') {
        onProgress?.(formatProgress({ text: `createEngine("${LEGACY_MODEL_ID}")…`, progress: 0 }));
        const eng = await w.createEngine(LEGACY_MODEL_ID, {
          initProgressCallback: (r) => onProgress?.(formatProgress(r)),
        });
        lastModelId = LEGACY_MODEL_ID;
        return eng;
      }
    } catch {
      /* UMD path gone on current CDN — fall through */
    }
  }
  return null;
}

/**
 * Load TinyLlama once. Weights download into the browser cache (WebGPU).
 * @param {(text: string) => void} onProgress
 */
export async function loadTinyLlama(onProgress = () => {}) {
  if (engine) return engine;
  if (loadPromise) return loadPromise;

  loadPromise = (async () => {
    try {
      const legacy = await tryLegacyCreateEngine(onProgress);
      if (legacy) {
        engine = legacy;
        onProgress?.(formatProgress({ text: 'TinyLlama ready (legacy createEngine).', progress: 1 }));
        return engine;
      }
    } catch (err) {
      onProgress?.(formatProgress({ text: `Legacy createEngine skipped: ${err.message}` }));
    }

    onProgress?.(formatProgress({ text: `Importing WebLLM → CreateMLCEngine("${MODEL_ID}")…`, progress: 0 }));
    const mod = await import(/* @vite-ignore */ WEBLLM_ESM);
    const CreateMLCEngine = mod.CreateMLCEngine || mod.createEngine || mod.default?.CreateMLCEngine;
    if (typeof CreateMLCEngine !== 'function') {
      throw new Error('WebLLM CreateMLCEngine missing from CDN module.');
    }
    engine = await CreateMLCEngine(MODEL_ID, {
      initProgressCallback: (r) => onProgress?.(formatProgress(r)),
    });
    lastModelId = MODEL_ID;
    onProgress?.(formatProgress({ text: 'TinyLlama ready.', progress: 1 }));
    return engine;
  })();

  try {
    return await loadPromise;
  } catch (err) {
    loadPromise = null;
    throw err;
  }
}

/**
 * Same chat call as Alexis's original Foundry box.
 * @param {string} prompt
 * @param {{ max_tokens?: number }} [opts]
 */
export async function chat(prompt, opts = {}) {
  if (!engine) throw new Error('Load TinyLlama first.');
  const max_tokens = opts.max_tokens ?? MAX_TOKENS;
  const reply = await engine.chat.completions.create({
    messages: [{ role: 'user', content: String(prompt ?? '') }],
    max_tokens,
  });
  const choice = reply?.choices?.[0];
  const content = choice?.message?.content ?? choice?.text;
  if (content == null) throw new Error('TinyLlama returned an empty completion.');
  return String(content);
}
