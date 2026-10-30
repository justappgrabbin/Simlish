export class GitHubWorkspaceAdapter {
  constructor({ baseUrl = '', token = '', fetchImpl = globalThis.fetch } = {}) {
    if (typeof fetchImpl !== 'function') throw new Error('fetch implementation required');
    this.baseUrl = String(baseUrl || '').replace(/\/$/, '');
    this.token = String(token || '');
    this.fetchImpl = fetchImpl;
  }

  configure({ baseUrl = this.baseUrl, token = this.token } = {}) {
    this.baseUrl = String(baseUrl || '').replace(/\/$/, '');
    this.token = String(token || '');
    return this;
  }

  async #request(method, path, body) {
    if (!this.baseUrl) throw new Error('Synthia Server URL not configured');
    if (!this.token) throw new Error('Computer auth token not configured');
    const response = await this.fetchImpl(`${this.baseUrl}${path}`, {
      method,
      headers: {
        'content-type': 'application/json',
        'x-terminal-token': this.token
      },
      body: body === undefined ? undefined : JSON.stringify(body)
    });
    const text = await response.text();
    let data;
    try { data = text ? JSON.parse(text) : null; } catch { data = { raw: text }; }
    if (!response.ok) {
      const error = new Error(data?.error || data?.message || `Computer GitHub bridge failed: ${response.status}`);
      error.status = response.status;
      error.details = data;
      throw error;
    }
    return data;
  }

  async request(input = {}) {
    switch (input.action) {
      case 'status':
        return this.#request('GET', '/computer/github/status');
      case 'list-repos':
        return this.#request('GET', '/computer/github/repos');
      case 'create-project': {
        const project = input.project || {};
        return this.#request('POST', '/computer/github/projects', {
          name: project.name,
          description: project.description,
          private: Boolean(input.private),
          files: Array.isArray(project.files)
            ? project.files.map(file => ({ path: file.path, content: file.content }))
            : []
        });
      }
      case 'dispatch-workflow':
        return this.#request('POST', '/computer/github/workflows/dispatch', input);
      case 'latest-workflow': {
        const q = new URLSearchParams();
        for (const key of ['owner','repo','workflowId','branch']) if (input[key]) q.set(key, input[key]);
        return this.#request('GET', `/computer/github/workflows/latest?${q}`);
      }
      default:
        throw new Error(`unsupported GitHub adapter action: ${input.action}`);
    }
  }
}
