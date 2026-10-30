export const QUALITY_TRANSFER_MODES = Object.freeze(['observe', 'borrow', 'modulate', 'learn', 'cultivate']);
export class QualityTransferError extends Error {
    constructor(code, message, details = {}) { super(message); this.name = 'QualityTransferError'; this.code = code; this.details = details; }
}
const qualityPorts = (tool, direction) => Object.freeze((tool.metadata?.qualityPorts || []).filter(port => port.direction === direction));
export class QualityTransferLedger {
    constructor() { this.active = new Map(); this.history = []; this.sequence = 0; }
    compatible(source, target, association) {
        const outputs = qualityPorts(source, 'output'), inputs = qualityPorts(target, 'input'), relations = new Set(association?.relations?.map(item => item.id) || []);
        return Object.freeze(outputs.flatMap(output => inputs.filter(input => input.quality === output.quality && input.schemaVersion === output.schemaVersion).map(input => Object.freeze({ output, input, resonant: !input.relations?.length || input.relations.some(relation => relations.has(relation)) }))).filter(pair => pair.resonant));
    }
    transfer({ source, target, quality, mode = 'observe', value, association, policy = {}, context = null } = {}) {
        if (!QUALITY_TRANSFER_MODES.includes(mode))
            throw new QualityTransferError('INVALID_TRANSFER_MODE', mode);
        const pair = this.compatible(source, target, association).find(item => item.output.quality === quality);
        if (!pair)
            throw new QualityTransferError('QUALITY_NOT_REACHABLE', quality);
        if (pair.output.modes && !pair.output.modes.includes(mode) || pair.input.modes && !pair.input.modes.includes(mode))
            throw new QualityTransferError('MODE_NOT_ALLOWED', mode);
        if (policy.allowed === false)
            throw new QualityTransferError('POLICY_DENIED', quality);
        const id = `quality:${++this.sequence}:${source.id}->${target.id}:${quality}`;
        const event = Object.freeze({ id, sequence: this.sequence, source: source.id, target: target.id, sourceAddress: source.addressKey, targetAddress: target.addressKey, quality, mode, value, association: Object.freeze({ mappingVersion: association.mappingVersion, operator: association.gateRelation.operator, emergenceSpot: association.emergenceSpot.addressKey }), context, status: 'active', retention: pair.input.retention || 'event' });
        this.history.push(event);
        if (mode !== 'observe')
            this.active.set(id, event);
        return event;
    }
    rollback(id, reason = 'rolled-back') { const event = this.active.get(id); if (!event)
        return false; this.active.delete(id); this.history.push(Object.freeze({ sequence: ++this.sequence, type: 'rollback', transferId: id, reason, previous: event })); return true; }
    lineage() { return Object.freeze([...this.history]); }
}
export const withQualityPorts = (tool, ports) => { tool.metadata = Object.freeze({ ...tool.metadata, qualityPorts: Object.freeze(ports.map(port => Object.freeze({ schemaVersion: '1', modes: Object.freeze([...(port.modes || QUALITY_TRANSFER_MODES)]), ...port }))) }); return tool; };
