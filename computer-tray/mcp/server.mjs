#!/usr/bin/env node
// Swarm Computer MCP server — zero dependencies, Node 18+.
//   stdio:  node mcp/server.mjs
//   HTTP:   node mcp/server.mjs --http [--port 8787]   (Streamable HTTP at /mcp, legacy SSE at /sse + /messages)
// Every file write passes the address engine (../address-engine/resolver.mjs) first.
import fs from 'node:fs'; import path from 'node:path'; import http from 'node:http'; import { randomUUID } from 'node:crypto';
import { state, encode } from '../address-engine/resolver.mjs';
const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const APPS = () => JSON.parse(fs.readFileSync(path.join(ROOT, 'apps/apps.json'), 'utf8'));
const LEDGER = path.join(ROOT, 'apps/address-ledger.jsonl');
const app = id => { const a = APPS().find(x => x.id === id); if (!a) throw new Error(`unknown app ${id}`); return a; };
const safe = (a, file) => { const base = path.join(ROOT, a.folder); const p = path.resolve(base, file); if (!p.startsWith(base + path.sep)) throw new Error('path escapes app folder'); return p; };
const walk = d => fs.existsSync(d) ? fs.readdirSync(d, { withFileTypes: true }).flatMap(e => e.isDirectory() ? walk(path.join(d, e.name)).map(f => e.name + '/' + f) : [e.name]) : [];
function gate(kind, id, address) {
  if (!address) throw new Error('address required: every app or file must pass the address engine first (gate.line.color.tone.base)');
  const st = state(address); if (st.unresolved) throw new Error(`address unresolved: ${st.reason}`);
  const rec = { at: new Date().toISOString(), kind, id, address: st.address, index: st.index, gate: st.gateState?.names?.[0] ?? null };
  fs.appendFileSync(LEDGER, JSON.stringify(rec) + '\n'); return rec;
}
const S = (type, props, req = []) => ({ type: 'object', properties: props, required: req, additionalProperties: false });
const str = d => ({ type: 'string', description: d });
const TOOLS = {
  list_apps: { description: 'List the apps in the Swarm Computer tray (id, name, folder, entry, status).', inputSchema: S('object', {}),
    run: () => APPS().map(({ files, ...a }) => ({ ...a, fileCount: files?.length ?? 0 })) },
  resolve_address: { description: "Resolve a gate.line.color.tone.base address through Alexis's state-space address engine. Returns the state plus the BHANTUGH layers (binary, HTML, ASCII, NATO, transliteration, Unicode, geometry, hex).", inputSchema: S('object', { address: str('e.g. 18.5.3.4.2') }, ['address']),
    run: ({ address }) => { const st = state(address); return st.unresolved ? st : { state: st, layers: encode(address) }; } },
  list_app_files: { description: 'List files inside one app folder.', inputSchema: S('object', { app_id: str('app id from list_apps') }, ['app_id']),
    run: ({ app_id }) => walk(path.join(ROOT, app(app_id).folder)) },
  read_app_file: { description: 'Read a text file from an app folder.', inputSchema: S('object', { app_id: str('app id'), file: str('path inside the app folder, e.g. index.html') }, ['app_id', 'file']),
    run: ({ app_id, file }) => fs.readFileSync(safe(app(app_id), file), 'utf8') },
  write_app_file: { description: 'Create or overwrite a text file in an app folder. The address is resolved through the address engine BEFORE the write; an unresolvable address blocks the write.', inputSchema: S('object', { app_id: str('app id'), file: str('path inside the app folder'), content: str('full new file content'), address: str('gate.line.color.tone.base for this file') }, ['app_id', 'file', 'content', 'address']),
    run: ({ app_id, file, content, address }) => { const a = app(app_id); const p = safe(a, file); const rec = gate('file', `${app_id}/${file}`, address); fs.mkdirSync(path.dirname(p), { recursive: true }); fs.writeFileSync(p, content); return { written: path.relative(ROOT, p), bytes: Buffer.byteLength(content), gate: rec }; } },
  search: { description: 'Search app names, notes and file names (ChatGPT connector compatible). Returns ids usable with fetch.', inputSchema: S('object', { query: str('search text') }, ['query']),
    run: ({ query }) => { const q = query.toLowerCase(); const results = APPS().flatMap(a => [{ id: `${a.id}:${a.entry}`, title: a.name, text: a.note || '', hit: (a.name + ' ' + (a.note || '')).toLowerCase().includes(q) }, ...(a.files || []).filter(f => f.toLowerCase().includes(q)).map(f => ({ id: `${a.id}:${f}`, title: `${a.name} / ${f}`, hit: true }))]).filter(r => r.hit).slice(0, 50).map(({ hit, ...r }) => ({ ...r, url: `computer://apps/${r.id}` })); return { results }; } },
  fetch: { description: 'Fetch one search result by id ("app_id:file") (ChatGPT connector compatible).', inputSchema: S('object', { id: str('id from search') }, ['id']),
    run: ({ id }) => { const [a, ...f] = id.split(':'); const file = f.join(':'); return { id, title: id, text: fs.readFileSync(safe(app(a), file), 'utf8').slice(0, 200000), url: `computer://apps/${id}` }; } },
};
const INFO = { name: 'swarm-computer', version: '0.1.0' };
async function handle(msg) {
  const { id, method, params } = msg; const ok = result => ({ jsonrpc: '2.0', id, result }); const err = (code, message) => ({ jsonrpc: '2.0', id, error: { code, message } });
  if (id === undefined) return null; // notification
  if (method === 'initialize') return ok({ protocolVersion: params?.protocolVersion || '2025-06-18', capabilities: { tools: {} }, serverInfo: INFO });
  if (method === 'ping') return ok({});
  if (method === 'tools/list') return ok({ tools: Object.entries(TOOLS).map(([name, t]) => ({ name, description: t.description, inputSchema: t.inputSchema })) });
  if (method === 'tools/call') { const t = TOOLS[params?.name]; if (!t) return err(-32602, `unknown tool ${params?.name}`);
    try { const r = await t.run(params.arguments || {}); return ok({ content: [{ type: 'text', text: typeof r === 'string' ? r : JSON.stringify(r, null, 1) }] }); }
    catch (e) { return ok({ isError: true, content: [{ type: 'text', text: e.message }] }); } }
  return err(-32601, `method not found: ${method}`);
}
const args = process.argv.slice(2);
if (!args.includes('--http')) {
  let buf = ''; process.stdin.setEncoding('utf8');
  process.stdin.on('data', async d => { buf += d; let i; while ((i = buf.indexOf('\n')) >= 0) { const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1); if (!line) continue;
    let m; try { m = JSON.parse(line); } catch { process.stdout.write(JSON.stringify({ jsonrpc: '2.0', id: null, error: { code: -32700, message: 'parse error' } }) + '\n'); continue; }
    const r = await handle(m); if (r) process.stdout.write(JSON.stringify(r) + '\n'); } });
} else {
  const port = +(args[args.indexOf('--port') + 1] || process.env.PORT || 8787); const sse = new Map();
  const body = req => new Promise(r => { let b = ''; req.on('data', c => b += c); req.on('end', () => r(b)); });
  http.createServer(async (req, res) => {
    res.setHeader('Access-Control-Allow-Origin', '*'); res.setHeader('Access-Control-Allow-Headers', '*'); res.setHeader('Access-Control-Expose-Headers', 'Mcp-Session-Id');
    if (req.method === 'OPTIONS') return res.end();
    const url = new URL(req.url, 'http://x');
    if (url.pathname === '/mcp' && req.method === 'POST') { const m = JSON.parse(await body(req) || '{}'); const batch = Array.isArray(m) ? m : [m];
      const out = (await Promise.all(batch.map(handle))).filter(Boolean); if (batch.some(x => x.method === 'initialize')) res.setHeader('Mcp-Session-Id', randomUUID());
      if (!out.length) { res.writeHead(202); return res.end(); } res.writeHead(200, { 'Content-Type': 'application/json' }); return res.end(JSON.stringify(Array.isArray(m) ? out : out[0])); }
    if (url.pathname === '/mcp' && req.method === 'GET') { res.writeHead(405); return res.end(); }
    if (url.pathname === '/sse' && req.method === 'GET') { const sid = randomUUID(); res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', Connection: 'keep-alive' });
      res.write(`event: endpoint\ndata: /messages?sessionId=${sid}\n\n`); sse.set(sid, res); const k = setInterval(() => res.write(': ping\n\n'), 15000); req.on('close', () => { clearInterval(k); sse.delete(sid); }); return; }
    if (url.pathname === '/messages' && req.method === 'POST') { const s = sse.get(url.searchParams.get('sessionId')); if (!s) { res.writeHead(404); return res.end('no session'); }
      const r = await handle(JSON.parse(await body(req))); if (r) s.write(`event: message\ndata: ${JSON.stringify(r)}\n\n`); res.writeHead(202); return res.end('accepted'); }
    if (url.pathname === '/') { res.writeHead(200, { 'Content-Type': 'text/plain' }); return res.end('swarm-computer MCP: POST /mcp (streamable HTTP) or GET /sse (legacy SSE)'); }
    res.writeHead(404); res.end();
  }).listen(port, () => console.error(`swarm-computer MCP on http://localhost:${port}/mcp  (SSE: /sse)`));
}
