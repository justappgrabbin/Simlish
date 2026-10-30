import { ComputerRuntime } from './computer-runtime/ComputerRuntime.mjs';
import { LocalStoragePersistence } from './computer-runtime/core/kernel.mjs';
import { mountSwarmComputerUI } from './swarm-ui.mjs';

const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];
const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, char => ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;' }[char]));
const titleFor = id => ({ home:'Computer Home', apps:'Apps', build:'Build', files:'Files', github:'GitHub', systems:'Systems', activity:'Activity' }[id] || 'SynthAI Computer');

const computer = await new ComputerRuntime({
  persistence: new LocalStoragePersistence(),
  namespace: 'synthai-computer-web'
}).boot();

globalThis.SynthAIComputer = computer;
const swarmUI = mountSwarmComputerUI({ computer });

let currentProjectId = localStorage.synthaiCurrentProject || '';
let activity = [];

function log(type, payload = {}) {
  activity.unshift({ at: Date.now(), type, payload });
  activity = activity.slice(0, 120);
  renderActivity();
}

computer.bus.on('*', event => log(event.type, event.payload));

function show(view) {
  $$('.view').forEach(el => el.classList.toggle('active', el.id === view));
  $$('.nav').forEach(el => el.classList.toggle('active', el.dataset.view === view));
  $('#viewTitle').textContent = titleFor(view);
  if (view === 'files') renderFiles();
  if (view === 'systems') { renderSystems(); void swarmUI.render(); }
}

$$('[data-view]').forEach(button => button.addEventListener('click', () => show(button.dataset.view)));
$$('[data-go]').forEach(button => button.addEventListener('click', () => show(button.dataset.go)));

function project() {
  return currentProjectId ? computer.projects.get(currentProjectId) : null;
}

function setCurrentProject(id) {
  currentProjectId = id || '';
  localStorage.synthaiCurrentProject = currentProjectId;
  renderProjects();
  loadCurrentFile();
}

function starterHtml(name) {
  const safeName = escapeHtml(name);
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>${safeName}</title>
  <style>
    body{font-family:system-ui;margin:0;min-height:100vh;display:grid;place-items:center;background:#100b18;color:#f8f3ff}
    main{max-width:700px;padding:32px}
    h1{font-size:clamp(2rem,8vw,4.5rem);margin:0 0 12px}
    p{color:#c8b8d7;line-height:1.6}
  </style>
</head>
<body><main><h1>${safeName}</h1><p>This project was created inside SynthAI Computer. Edit this file, preview it locally, then publish the same files to GitHub.</p></main></body>
</html>`;
}

function mimeFor(path) {
  if (path.endsWith('.html')) return 'text/html';
  if (path.endsWith('.css')) return 'text/css';
  if (path.endsWith('.js') || path.endsWith('.mjs')) return 'text/javascript';
  if (path.endsWith('.json')) return 'application/json';
  return 'text/plain';
}

async function createProject() {
  const name = $('#projectName').value.trim();
  if (!name) return message('#builderMessage', 'Project name is required.', false);
  try {
    const p = await computer.projects.create({ name, description: $('#projectDescription').value.trim() });
    await computer.projects.writeFile(p.id, 'index.html', starterHtml(name), { type: 'text/html' });
    setCurrentProject(p.id);
    $('#filePath').value = 'index.html';
    loadCurrentFile();
    message('#builderMessage', 'Project created in the Computer and persisted locally.', true);
  } catch (error) {
    message('#builderMessage', error.message, false);
  }
}

async function saveFile() {
  const p = project();
  if (!p) return message('#builderMessage', 'Choose or create a project first.', false);
  const path = $('#filePath').value.trim();
  if (!path) return message('#builderMessage', 'File path is required.', false);
  try {
    await computer.projects.writeFile(p.id, path, $('#editor').value, { type: mimeFor(path) });
    message('#builderMessage', `${path} saved to the Computer VFS.`, true);
    renderProjects();
    renderFiles();
  } catch (error) {
    message('#builderMessage', error.message, false);
  }
}

function loadCurrentFile() {
  const p = project();
  $('#projectState').textContent = p ? p.status.toUpperCase() : 'NO PROJECT';
  $('#fileProjectName').textContent = p ? p.name : 'choose a project';
  if (!p) return;
  $('#projectName').value = p.name;
  $('#projectDescription').value = p.description || '';
  const requested = $('#filePath').value.trim() || 'index.html';
  const file = computer.projects.readFile(p.id, requested) || computer.projects.listFiles(p.id)[0];
  if (file) {
    $('#filePath').value = file.path;
    $('#editor').value = file.content;
  } else {
    $('#editor').value = '';
  }
  renderFiles();
}

function preview() {
  const p = project();
  if (!p) return message('#builderMessage', 'Choose a project first.', false);
  const file = computer.projects.readFile(p.id, 'index.html');
  if (!file) return message('#builderMessage', 'This project has no index.html to preview.', false);
  $('#preview').srcdoc = file.content;
  message('#builderMessage', 'Preview rendered from the Computer VFS.', true);
}

async function publish() {
  const p = project();
  if (!p) return message('#builderMessage', 'Choose a project first.', false);
  try {
    await saveFile();
    const result = await computer.projects.publish(p.id, { backend: 'github' });
    renderProjects();
    const url = result?.repo?.url;
    message('#builderMessage', url ? `Published: ${url}` : 'GitHub publish completed.', true);
    await loadRepos();
  } catch (error) {
    message('#builderMessage', `Publish failed: ${error.message}`, false);
  }
}

function renderProjects() {
  const projects = computer.projects.list();
  $('#projectCount').textContent = `${projects.length} project${projects.length === 1 ? '' : 's'}`;
  $('#projectPicker').innerHTML = '<option value="">Choose project</option>' + projects.map(p =>
    `<option value="${escapeHtml(p.id)}" ${p.id === currentProjectId ? 'selected' : ''}>${escapeHtml(p.name)} · ${escapeHtml(p.status)}</option>`
  ).join('');
  $('#recentProjects').classList.toggle('empty', projects.length === 0);
  $('#recentProjects').innerHTML = projects.length ? projects.slice(0, 6).map(p =>
    `<div class="system-row"><div><strong>${escapeHtml(p.name)}</strong><small>${Object.keys(p.files || {}).length} files</small></div><span class="state">${escapeHtml(p.status.toUpperCase())}</span></div>`
  ).join('') : 'No local projects yet.';
}

function renderFiles() {
  const p = project();
  $('#fileProjectName').textContent = p ? p.name : 'choose a project';
  if (!p) {
    $('#fileList').className = 'file-list empty';
    $('#fileList').textContent = 'No project selected.';
    return;
  }
  const files = computer.projects.listFiles(p.id);
  $('#fileList').className = files.length ? 'file-list' : 'file-list empty';
  $('#fileList').innerHTML = files.length ? files.map(file =>
    `<button class="file-row" data-file="${escapeHtml(file.path)}"><div><strong>${escapeHtml(file.path)}</strong><small>${escapeHtml(file.type)}</small></div><span>Open</span></button>`
  ).join('') : 'No files in this project.';
  $$('#fileList [data-file]').forEach(button => button.addEventListener('click', () => {
    $('#filePath').value = button.dataset.file;
    loadCurrentFile();
    show('build');
  }));
}

function renderSystems() {
  const systems = computer.systems.list();
  const rows = systems.map(system =>
    `<div class="card-mini"><div><strong>${escapeHtml(system.name)}</strong><small>${escapeHtml(system.role)} · ${escapeHtml(system.relationship)}</small></div><span class="state">${system.id === 'synthai-computer' ? 'WIRED' : 'PRESENT'}</span></div>`
  ).join('');
  $('#systemsList').innerHTML = rows;
  $('#homeSystems').innerHTML = systems.slice(0, 5).map(system =>
    `<div class="system-row"><div><strong>${escapeHtml(system.name)}</strong><small>${escapeHtml(system.role)}</small></div><span class="state">${system.id === 'synthai-computer' ? 'WIRED' : 'PRESENT'}</span></div>`
  ).join('');
}

function renderActivity() {
  const el = $('#activityList');
  if (!el) return;
  el.innerHTML = activity.length ? activity.map(item =>
    `<div class="activity-row"><time>${new Date(item.at).toLocaleTimeString()}</time><strong>${escapeHtml(item.type)}</strong><div>${escapeHtml(JSON.stringify(item.payload).slice(0, 500))}</div></div>`
  ).join('') : '<div class="empty">No visible events yet.</div>';
}

function message(selector, text, good) {
  const el = $(selector);
  el.textContent = text;
  el.className = `message ${good === true ? 'good' : good === false ? 'bad' : ''}`;
}

async function connectGitHub() {
  const baseUrl = $('#serverUrl').value.trim().replace(/\/$/, '');
  const token = $('#computerToken').value;
  localStorage.synthaiServerUrl = baseUrl;
  localStorage.synthaiComputerToken = token;
  computer.configureGitHub({ baseUrl, token });
  $('#githubPanelState').textContent = 'WIRED';
  try {
    const result = await computer.backends.request('github', { action: 'status' });
    $('#githubStatus').textContent = 'VERIFIED';
    $('#githubPanelState').textContent = 'VERIFIED';
    $('#githubDetail').textContent = `connected as ${result.user}`;
    message('#githubMessage', `Verified GitHub connection as ${result.user}.`, true);
    await loadRepos();
  } catch (error) {
    $('#githubStatus').textContent = 'PARTIALLY WIRED';
    $('#githubPanelState').textContent = 'PARTIALLY WIRED';
    $('#githubDetail').textContent = 'bridge configured, verification failed';
    message('#githubMessage', error.message, false);
  }
}

async function loadRepos() {
  try {
    const result = await computer.backends.request('github', { action: 'list-repos' });
    const repos = result.repos || [];
    $('#repoCount').textContent = String(repos.length);
    $('#repoList').className = repos.length ? 'repo-list' : 'repo-list empty';
    $('#repoList').innerHTML = repos.length ? repos.map(repo =>
      `<div class="repo-row"><div><a class="repo-link" target="_blank" rel="noreferrer" href="${escapeHtml(repo.url)}"><strong>${escapeHtml(repo.fullName)}</strong></a><small>${escapeHtml(repo.defaultBranch)} · ${repo.private ? 'private' : 'public'}</small></div><span class="state">PRESENT</span></div>`
    ).join('') : 'No repositories returned.';
  } catch (error) {
    message('#githubMessage', error.message, false);
  }
}

$('#createProject').addEventListener('click', createProject);
$('#saveFile').addEventListener('click', saveFile);
$('#previewFile').addEventListener('click', preview);
$('#publishProject').addEventListener('click', publish);
$('#connectGitHub').addEventListener('click', connectGitHub);
$('#refreshRepos').addEventListener('click', loadRepos);
$('#projectPicker').addEventListener('change', event => setCurrentProject(event.target.value));
$('#clearActivity').addEventListener('click', () => { activity = []; renderActivity(); });

const savedServer = localStorage.synthaiServerUrl;
const savedToken = localStorage.synthaiComputerToken;
if (savedServer) $('#serverUrl').value = savedServer;
if (savedToken) $('#computerToken').value = savedToken;

$('#bootDot').classList.add('good');
$('#bootLabel').textContent = 'runtime online';
$('#runtimeStatus').textContent = 'LOCAL';
$('#runtimeDetail').textContent = computer.snapshot().version;
$('#mountCount').textContent = `${computer.shellManager.listMounted().length} mounted`;

renderProjects();
renderSystems();
await swarmUI.render();
renderActivity();
if (currentProjectId && computer.projects.get(currentProjectId)) loadCurrentFile();
if (savedServer && savedToken) connectGitHub();
