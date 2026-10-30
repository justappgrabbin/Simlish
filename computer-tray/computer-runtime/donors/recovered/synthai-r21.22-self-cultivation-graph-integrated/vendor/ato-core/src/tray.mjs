export class TrayError extends Error {
    constructor(code, message) { super(message); this.name = 'TrayError'; this.code = code; }
}
export const TRAY_FORMS = Object.freeze(['compact', 'palette', 'radial', 'bottom-sheet', 'grid', 'hidden']);
export class CapabilityTray {
    constructor({ formResolver = null } = {}) { this.formResolver = formResolver; }
    resolve({ mesh, workspace, context = {}, notifications = [], historyAvailable = true } = {}) {
        if (!mesh || !workspace)
            throw new TrayError('MISSING_CONTEXT', 'Tray requires mesh and workspace');
        const tools = mesh.snapshot().automatons.map((tool) => Object.freeze({
            id: tool.id,
            label: tool.metadata?.label || tool.metadata?.family || tool.id,
            family: tool.metadata?.family || null,
            address: tool.addressKey,
            state: tool.lifecycle,
            functionalLevel: tool.functionalLevel,
            available: tool.lifecycle !== 'error',
        }));
        const form = this.formResolver ? this.formResolver({ tools, workspace: workspace.current, context }) : this.#defaultForm(context, tools.length);
        if (!TRAY_FORMS.includes(form))
            throw new TrayError('INVALID_TRAY_FORM', form);
        return Object.freeze({
            form,
            expression: true,
            items: Object.freeze(tools),
            anchors: Object.freeze({
                presence: 'companion',
                currentTask: workspace.current.focus,
                history: historyAvailable,
                notifications: notifications.length,
                controls: true,
            }),
        });
    }
    #defaultForm(context, count) {
        if (context.mode === 'focus')
            return 'compact';
        if (context.mode === 'conversation')
            return 'bottom-sheet';
        if (count > 12)
            return 'palette';
        return 'grid';
    }
}
