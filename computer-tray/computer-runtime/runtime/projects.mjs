const cleanPath = value => String(value || '').replaceAll('\\','/').replace(/^\/+/, '').replace(/\/{2,}/g,'/');

export class ProjectWorkspace {
  constructor({ bus, state, vfs, backends } = {}) {
    Object.assign(this, { bus, state, vfs, backends });
  }

  async create({ name, description = '' } = {}) {
    if (!name?.trim()) throw new Error('project name required');
    const id = `project-${crypto.randomUUID?.() ?? Math.random().toString(36).slice(2)}`;
    const record = {
      id,
      name: name.trim(),
      description: String(description || ''),
      status: 'local',
      files: {},
      createdAt: Date.now(),
      updatedAt: Date.now(),
      publications: []
    };
    await this.state.set(`projects.items.${id}`, record, { source: 'project-workspace' });
    this.bus?.emit('project:created', record);
    return structuredClone(record);
  }

  get(id) {
    return this.state.get(`projects.items.${id}`, null);
  }

  list() {
    return Object.values(this.state.get('projects.items', {}))
      .filter(Boolean)
      .sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
  }

  async writeFile(projectId, path, content, { type = 'text/plain' } = {}) {
    const project = this.get(projectId);
    if (!project) throw new Error(`unknown project: ${projectId}`);
    const filePath = cleanPath(path);
    if (!filePath) throw new Error('file path required');
    const vfsPath = `/projects/${projectId}/${filePath}`;
    await this.vfs.write(vfsPath, content ?? '', { type, source: 'project-workspace', projectId, filePath });
    project.files[filePath] = { path: filePath, vfsPath, type, updatedAt: Date.now() };
    project.updatedAt = Date.now();
    await this.state.set(`projects.items.${projectId}`, project, { source: 'project-workspace' });
    this.bus?.emit('project:file-written', { projectId, path: filePath, vfsPath });
    return { ...project.files[filePath], content: String(content ?? '') };
  }

  readFile(projectId, path) {
    const project = this.get(projectId);
    if (!project) throw new Error(`unknown project: ${projectId}`);
    const filePath = cleanPath(path);
    const meta = project.files?.[filePath];
    if (!meta) return null;
    const record = this.vfs.read(meta.vfsPath);
    return record ? { ...meta, content: record.content } : null;
  }

  listFiles(projectId) {
    const project = this.get(projectId);
    if (!project) throw new Error(`unknown project: ${projectId}`);
    return Object.keys(project.files || {}).sort().map(path => this.readFile(projectId, path));
  }

  snapshot(projectId) {
    const project = this.get(projectId);
    if (!project) throw new Error(`unknown project: ${projectId}`);
    return {
      ...project,
      files: this.listFiles(projectId).map(file => ({ path: file.path, type: file.type, content: file.content }))
    };
  }

  async publish(projectId, { backend = 'github', private: isPrivate = false } = {}) {
    if (!this.backends) throw new Error('backend broker unavailable');
    const project = this.snapshot(projectId);
    const result = await this.backends.request(backend, { action: 'create-project', project, private: Boolean(isPrivate) });
    const current = this.get(projectId);
    current.status = 'published';
    current.updatedAt = Date.now();
    current.publications = [...(current.publications || []), { backend, result, at: Date.now() }];
    await this.state.set(`projects.items.${projectId}`, current, { source: 'project-workspace' });
    this.bus?.emit('project:published', { projectId, backend, result });
    return result;
  }
}
