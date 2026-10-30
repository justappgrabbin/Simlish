import { Automaton } from './automaton.mjs';
import { StateSpaceAssociationResolver } from './association.mjs';
export class GenerativeError extends Error {
    constructor(code, message, details = {}) { super(message); this.name = 'GenerativeError'; this.code = code; this.details = details; }
}
const outputPorts = (tool) => tool.ports.filter((port) => port.direction === 'output');
const inputPorts = (tool) => tool.ports.filter((port) => port.direction === 'input');
const compatiblePair = (left, right) => {
    for (const output of outputPorts(left))
        for (const input of inputPorts(right)) {
            const guarantees = new Set(output.guarantees);
            if (output.type === input.type && output.schemaVersion === input.schemaVersion && input.requires.every((item) => guarantees.has(item)))
                return { output, input };
        }
    return null;
};
export class GenerativeEmergence {
    constructor({ resolver = new StateSpaceAssociationResolver(), manifestVersion = 'ato.generated-tool.v1' } = {}) {
        this.resolver = resolver;
        this.manifestVersion = manifestVersion;
        this.candidates = new Map();
        this.generated = new Map();
        this.revisions = [];
    }
    consider(left, right, { context = null, direction = 'left-to-right' } = {}) {
        const association = this.resolver.resolve(left, right, { context });
        const source = direction === 'left-to-right' ? left : right;
        const target = direction === 'left-to-right' ? right : left;
        const pair = compatiblePair(source, target);
        if (!pair)
            return Object.freeze({ status: 'unresolved', reason: 'NO_EXECUTABLE_CONTRACT', association });
        if (!association.emergenceSpot.resolved)
            return Object.freeze({ status: 'unresolved', reason: 'EMERGENCE_ADDRESS_INCOMPLETE', association });
        const exposedInput = inputPorts(source)[0];
        const exposedOutput = outputPorts(target)[0];
        if (!exposedInput || !exposedOutput)
            return Object.freeze({ status: 'unresolved', reason: 'INCOMPLETE_STANDALONE_CONTRACT', association });
        const id = `generated:${association.mappingVersion}:${association.gateRelation.operator}:${[left.id, right.id].sort().join('+')}`;
        const candidate = Object.freeze({
            id,
            status: 'candidate',
            manifestVersion: this.manifestVersion,
            association,
            members: Object.freeze([left.id, right.id]),
            memberAddresses: Object.freeze([left.addressKey, right.addressKey]),
            direction,
            source,
            target,
            contract: Object.freeze({ bridgeOutput: pair.output, bridgeInput: pair.input, exposedInput, exposedOutput }),
            generation: 1 + Math.max(left.metadata?.generation || 0, right.metadata?.generation || 0),
        });
        this.candidates.set(id, candidate);
        return candidate;
    }
    synthesize(candidateId, { functionalLevel = 'space', expressionField = null } = {}) {
        const candidate = this.candidates.get(candidateId);
        if (!candidate)
            throw new GenerativeError('CANDIDATE_NOT_FOUND', candidateId);
        const { source, target, association } = candidate;
        const implementation = async (input, context) => {
            const intermediate = await source.call(input, { ...context, generatedBy: candidate.id });
            return target.call(intermediate, { ...context, generatedBy: candidate.id });
        };
        const tool = new Automaton({
            id: candidate.id,
            address: association.emergenceSpot.address,
            structure: 'bigram',
            activeLevels: [1],
            functionalLevel,
            ports: [
                { id: 'in', direction: 'input', type: candidate.contract.exposedInput.type, schemaVersion: candidate.contract.exposedInput.schemaVersion, requires: candidate.contract.exposedInput.requires, guarantees: [] },
                { id: 'out', direction: 'output', type: candidate.contract.exposedOutput.type, schemaVersion: candidate.contract.exposedOutput.schemaVersion, requires: [], guarantees: candidate.contract.exposedOutput.guarantees },
            ],
            implementation,
            manifestVersion: this.manifestVersion,
            metadata: {
                family: 'generated-tool',
                generation: candidate.generation,
                emergent: true,
                decomposable: true,
                members: candidate.members,
                memberAddresses: candidate.memberAddresses,
                mappingVersion: association.mappingVersion,
                gateOperator: association.gateRelation,
                relations: association.relations,
            },
        });
        this.generated.set(tool.id, Object.freeze({ tool, candidate, active: true }));
        this.candidates.delete(candidateId);
        expressionField?.express({ source: tool, mode: 'creation', qualities: { emergent: true, generation: candidate.generation }, content: { toolId: tool.id, members: candidate.members }, context: association });
        return tool;
    }
    dissolve(toolId, reason = 'dissolved') {
        const record = this.generated.get(toolId);
        if (!record)
            return false;
        this.generated.set(toolId, Object.freeze({ ...record, active: false, reason }));
        this.revisions.push(Object.freeze({ type: 'dissolve', toolId, reason, previous: record }));
        return true;
    }
    restore(toolId) {
        const record = this.generated.get(toolId);
        if (!record)
            throw new GenerativeError('GENERATED_TOOL_NOT_FOUND', toolId);
        this.generated.set(toolId, Object.freeze({ ...record, active: true, reason: null }));
        this.revisions.push(Object.freeze({ type: 'restore', toolId }));
        return record.tool;
    }
    decompose(toolId) {
        const record = this.generated.get(toolId);
        if (!record)
            throw new GenerativeError('GENERATED_TOOL_NOT_FOUND', toolId);
        return Object.freeze({ toolId, members: record.candidate.members, memberAddresses: record.candidate.memberAddresses, association: record.candidate.association });
    }
}
