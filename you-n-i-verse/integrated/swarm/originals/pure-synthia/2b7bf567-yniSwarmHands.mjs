function validHttpUrl(value) {
  try { const url = new URL(String(value)); return ['http:', 'https:'].includes(url.protocol) ? url.href : null; }
  catch { return null; }
}

export function bindYNIToSwarm(swarm, {
  host = globalThis.YNIHost || {},
  permissions = globalThis.SynthiaPermissions || {},
  document = globalThis.document,
  fetchImpl = globalThis.fetch?.bind(globalThis),
} = {}) {
  if (!swarm?.registerExternal) throw new TypeError('bindYNIToSwarm requires a swarm body');
  const allowed = (name) => permissions[name] === true;
  const ids = [];

  ids.push(swarm.registerExternal({
    id: 'hand:yni-dom', group: 'browser-hands', maxConcurrency: 2,
    capabilities: ['device.dom.click','device.dom.focus','device.dom.fill','device.dom.inspect'],
    execute: async (task = {}) => {
      if (!allowed('allowExecution')) throw new Error('EXECUTION_PERMISSION_REQUIRED');
      if (task.op === 'inspect') return host.inspect ? host.inspect(task) : { title: document?.title || null, url: globalThis.location?.href || null };
      if (task.op === 'click') { if (host.click) return host.click(task.selector); const el = document?.querySelector(task.selector); if (!el) throw new Error(`No element matches ${task.selector}`); el.click(); return { clicked: task.selector }; }
      if (task.op === 'focus') { if (host.focus) return host.focus(task.selector); const el = document?.querySelector(task.selector); if (!el) throw new Error(`No element matches ${task.selector}`); el.focus(); return { focused: task.selector }; }
      if (task.op === 'fill') { if (host.fill) return host.fill(task.selector, task.value); const el = document?.querySelector(task.selector); if (!el) throw new Error(`No element matches ${task.selector}`); el.value = task.value; el.dispatchEvent?.(new Event('input',{bubbles:true})); return { filled: task.selector }; }
      throw new Error(`Unknown DOM hand op: ${task.op}`);
    },
  }).id);

  ids.push(swarm.registerExternal({
    id: 'hand:yni-browser', group: 'browser-hands', maxConcurrency: 4,
    capabilities: ['browser.open','browser.navigate'],
    execute: async (task = {}) => {
      if (!allowed('allowNetwork')) throw new Error('NETWORK_PERMISSION_REQUIRED');
      const url = validHttpUrl(task.url); if (!url) throw new Error('Invalid URL');
      if (task.op === 'navigate' && host.navigate) return host.navigate(url);
      if (host.open) return host.open(url);
      globalThis.open?.(url, '_blank'); return { opened: url };
    },
  }).id);

  ids.push(swarm.registerExternal({
    id: 'hand:yni-network', group: 'network-hands', maxConcurrency: 6,
    capabilities: ['network.request','network.download'],
    execute: async (task = {}) => {
      if (!allowed('allowNetwork')) throw new Error('NETWORK_PERMISSION_REQUIRED');
      const url = validHttpUrl(task.url); if (!url) throw new Error('Invalid URL');
      if (host.request) return host.request(url, task);
      if (!fetchImpl) throw new Error('Network host unavailable');
      const response = await fetchImpl(url, { method: task.method || 'GET', headers: task.headers, body: task.data });
      if (!response.ok) throw new Error(`Network request failed: ${response.status}`);
      return { status: response.status, data: task.responseType === 'text' ? await response.text() : await response.arrayBuffer() };
    },
  }).id);

  ids.push(swarm.registerExternal({
    id: 'hand:yni-files', group: 'file-hands', maxConcurrency: 3,
    capabilities: ['file.read','file.write'],
    execute: async (task = {}) => {
      if (!allowed('allowFiles')) throw new Error('FILE_PERMISSION_REQUIRED');
      if (task.op === 'read' && host.read) return host.read(task.name);
      if (task.op === 'write' && host.save) return host.save(task.name, task.data);
      throw new Error(`File operation unavailable: ${task.op}`);
    },
  }).id);

  return Object.freeze({ registered: Object.freeze(ids), snapshot: swarm.snapshot() });
}

export default bindYNIToSwarm;
