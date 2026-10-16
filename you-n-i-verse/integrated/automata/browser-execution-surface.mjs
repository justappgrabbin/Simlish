function safeValue(value) {
  try { return structuredClone(value); } catch { return String(value); }
}

function hasModuleSyntax(text) {
  return /(^|\n)\s*(?:import|export)\b/.test(String(text ?? ''));
}

async function executeJson(normalized) {
  try {
    return {
      ok: true,
      kind: 'json',
      path: 'browser-native-json',
      backendUsed: false,
      workerUsed: false,
      result: { engine: 'JSON.parse', stdout: [], returnValue: JSON.parse(normalized.text) },
    };
  } catch (error) {
    return { ok: false, kind: 'json', path: 'browser-native-json', error: String(error.message), backendUsed: false, workerUsed: false };
  }
}

async function executeWasm(normalized, imports = {}) {
  try {
    const module = await WebAssembly.compile(normalized.bytes);
    const required = WebAssembly.Module.imports(module);
    const unresolved = required
      .filter((entry) => !Object.prototype.hasOwnProperty.call(imports, entry.module))
      .map((entry) => `${entry.module}.${entry.name}`);
    if (unresolved.length) {
      return {
        ok: false,
        kind: 'wasm',
        path: 'browser-native-wasm',
        error: 'WASM imports are unresolved',
        unresolvedDependencies: unresolved,
        backendUsed: false,
        workerUsed: false,
      };
    }
    const instance = await WebAssembly.instantiate(module, imports);
    return {
      ok: true,
      kind: 'wasm',
      path: 'browser-native-wasm',
      backendUsed: false,
      workerUsed: false,
      result: {
        engine: 'WebAssembly',
        stdout: [],
        returnValue: { exports: Object.keys(instance.exports).sort() },
      },
    };
  } catch (error) {
    return { ok: false, kind: 'wasm', path: 'browser-native-wasm', error: String(error.message), backendUsed: false, workerUsed: false };
  }
}

async function executeJavascriptInSandbox(normalized, { timeoutMs = 2500, globals = {} } = {}) {
  if (hasModuleSyntax(normalized.text)) {
    return {
      ok: false,
      kind: 'javascript',
      path: 'browser-sandbox-js',
      error: 'module graph requires dependency linking before execution',
      unresolvedDependencies: ['module-loader'],
      backendUsed: false,
      workerUsed: false,
    };
  }
  if (typeof document === 'undefined' || typeof window === 'undefined') {
    return {
      ok: false,
      kind: 'javascript',
      path: 'browser-sandbox-unavailable',
      error: 'document/window are unavailable in this host; delegate to the existing runtime bridge',
      unresolvedDependencies: ['browser-document-surface'],
      backendUsed: false,
      workerUsed: false,
    };
  }

  const token = `synthia-${Math.random().toString(36).slice(2)}-${Date.now()}`;
  const iframe = document.createElement('iframe');
  iframe.setAttribute('sandbox', 'allow-scripts');
  iframe.style.display = 'none';
  iframe.srcdoc = `<!doctype html><meta charset="utf-8"><script>
    addEventListener('message', async (event) => {
      const data = event.data || {};
      if (data.type !== 'synthia-execute') return;
      const stdout = [];
      const wrap = (name) => (...args) => stdout.push({ level:name, args:args.map((v) => {
        try { return structuredClone(v); } catch { return String(v); }
      }) });
      const consoleProxy = { log:wrap('log'), info:wrap('info'), warn:wrap('warn'), error:wrap('error') };
      try {
        const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor;
        const fn = new AsyncFunction('console','synthiaGlobals', '"use strict";\\n' + data.code);
        const returnValue = await fn(consoleProxy, data.globals || {});
        parent.postMessage({ type:'synthia-result', token:data.token, ok:true, stdout, returnValue }, '*');
      } catch (error) {
        parent.postMessage({ type:'synthia-result', token:data.token, ok:false, stdout, error:String(error && error.message || error) }, '*');
      }
    });
  <\/script>`;

  document.documentElement.appendChild(iframe);
  return new Promise((resolve) => {
    let settled = false;
    const finish = (result) => {
      if (settled) return;
      settled = true;
      window.removeEventListener('message', onMessage);
      clearTimeout(timer);
      iframe.remove();
      resolve(result);
    };
    const onMessage = (event) => {
      if (event.source !== iframe.contentWindow || event.data?.type !== 'synthia-result' || event.data?.token !== token) return;
      finish({
        ok: Boolean(event.data.ok),
        kind: 'javascript',
        path: 'browser-sandbox-js',
        backendUsed: false,
        workerUsed: false,
        ...(event.data.ok
          ? { result: { engine: 'sandboxed-iframe-js', stdout: safeValue(event.data.stdout ?? []), returnValue: safeValue(event.data.returnValue) } }
          : { error: String(event.data.error ?? 'execution failed'), result: { engine: 'sandboxed-iframe-js', stdout: safeValue(event.data.stdout ?? []) } }),
      });
    };
    window.addEventListener('message', onMessage);
    const timer = setTimeout(() => finish({
      ok: false,
      kind: 'javascript',
      path: 'browser-sandbox-js-timeout',
      error: `execution exceeded ${timeoutMs}ms`,
      backendUsed: false,
      workerUsed: false,
    }), timeoutMs);
    iframe.addEventListener('load', () => {
      iframe.contentWindow.postMessage({ type: 'synthia-execute', token, code: normalized.text, globals: safeValue(globals) }, '*');
    }, { once: true });
  });
}

export class BrowserExecutionSurface {
  constructor({ providers = [] } = {}) {
    this.providers = [...providers];
  }

  register(provider) {
    if (!provider || typeof provider.canExecute !== 'function' || typeof provider.execute !== 'function') {
      throw new TypeError('execution provider requires canExecute() and execute()');
    }
    this.providers.push(provider);
    return provider;
  }

  async execute(normalized, context = {}) {
    for (const provider of this.providers) {
      if (await provider.canExecute(normalized, context)) return provider.execute(normalized, context);
    }
    if (normalized.kind === 'json') return executeJson(normalized);
    if (normalized.kind === 'text') {
      return {
        ok: true,
        kind: 'text',
        path: 'browser-native-text',
        backendUsed: false,
        workerUsed: false,
        result: { engine: 'TextDecoder', stdout: [], returnValue: normalized.text },
      };
    }
    if (normalized.kind === 'wasm') return executeWasm(normalized, context.wasmImports ?? {});
    if (normalized.kind === 'javascript') return executeJavascriptInSandbox(normalized, context);
    return {
      ok: false,
      kind: normalized.kind,
      path: 'browser-provider-required',
      error: `no browser-local execution provider registered for ${normalized.kind}`,
      unresolvedDependencies: [`provider:${normalized.kind}`],
      backendUsed: false,
      workerUsed: false,
    };
  }
}

export default BrowserExecutionSurface;
