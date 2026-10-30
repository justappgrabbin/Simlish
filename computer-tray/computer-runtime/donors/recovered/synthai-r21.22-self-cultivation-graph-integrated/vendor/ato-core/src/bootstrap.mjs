import { AutomataMesh } from './automaton.mjs';
import { bootstrapKleinTools } from './klein-tools.mjs';
import { successAutomaton } from './success.mjs';
import { conversationAutomaton } from './families.mjs';
import { browserFormAutomaton } from './browser-form.mjs';
import { researchBrowserAutomaton } from './research-browser.mjs';
import { computationalGrammarCoderAutomaton } from './computational-grammar-coder.mjs';
export const SEED_PROFILE = Object.freeze({
    id: 'generic-seed',
    version: 'ato.seed-profile.v1',
    personalized: false,
    qualities: Object.freeze({}),
    provenance: 'portable-default',
});
export function bootstrapATO({ mesh = new AutomataMesh(), profile = SEED_PROFILE, includeConversation = true, includeSuccess = true, includeBrowser = true } = {}) {
    const tools = [...bootstrapKleinTools(mesh)];
    if (includeSuccess && !mesh.automatons.has('success')) {
        const tool = successAutomaton();
        mesh.add(tool);
        tools.push(tool);
    }
    if (includeConversation && !mesh.automatons.has('conversation')) {
        const tool = conversationAutomaton();
        mesh.add(tool);
        tools.push(tool);
    }
    if (includeBrowser && !mesh.automatons.has('browser-form')) {
        const tool = browserFormAutomaton();
        mesh.add(tool);
        tools.push(tool);
    }
    if (includeBrowser && !mesh.automatons.has('research-browser')) {
        const tool = researchBrowserAutomaton();
        mesh.add(tool);
        tools.push(tool);
    }
    if (!mesh.automatons.has('computational-grammar-coder')) {
        const tool = computationalGrammarCoderAutomaton();
        mesh.add(tool);
        tools.push(tool);
    }
    return Object.freeze({ status: 'ready', profile: Object.freeze({ ...profile, qualities: Object.freeze({ ...profile.qualities }) }), mesh, tools: Object.freeze(tools), manifests: Object.freeze(tools.map(tool => tool.manifest())) });
}
