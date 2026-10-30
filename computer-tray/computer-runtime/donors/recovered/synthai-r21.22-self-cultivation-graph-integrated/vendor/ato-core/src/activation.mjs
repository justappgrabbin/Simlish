export class ActivationError extends TypeError {
    constructor(code, message, details = {}) {
        super(message);
        this.name = 'ActivationError';
        this.code = code;
        this.details = details;
    }
}
export const STRUCTURES = Object.freeze({
    bigram: Object.freeze({ levelCount: 2, minimumActive: 1, canonicalActive: 1 }),
    trigram: Object.freeze({ levelCount: 3, minimumActive: 2, canonicalActive: 2 }),
    hexagram: Object.freeze({ levelCount: 6, minimumActive: 5, canonicalActive: 5 }),
  decagram: Object.freeze({ levelCount: 10, minimumActive: 8, canonicalActive: 8 }),
});
export function activationSignature(structure, activeLevels) {
    const definition = typeof structure === 'string' ? STRUCTURES[structure] : structure;
    if (!definition || !Number.isInteger(definition.levelCount) || definition.levelCount < 1) {
        throw new ActivationError('INVALID_STRUCTURE', 'A valid structure and level count are required');
    }
    if (!Array.isArray(activeLevels)) {
        throw new ActivationError('MISSING_ACTIVE_LEVELS', 'activeLevels must be supplied explicitly');
    }
    const unique = [...new Set(activeLevels)].sort((a, b) => a - b);
    const invalid = unique.filter((level) => !Number.isInteger(level) || level < 1 || level > definition.levelCount);
    if (invalid.length) {
        throw new ActivationError('INVALID_ACTIVE_LEVEL', 'Active levels fall outside the occupied structure', { invalid });
    }
    if (unique.length < (definition.minimumActive ?? 0)) {
        throw new ActivationError('INSUFFICIENT_ACTIVATION', 'The structure does not contain enough active levels', {
            required: definition.minimumActive,
            received: unique.length,
        });
    }
    const active = new Set(unique);
    const inactiveLevels = Array.from({ length: definition.levelCount }, (_, index) => index + 1)
        .filter((level) => !active.has(level));
    const activationMask = Array.from({ length: definition.levelCount }, (_, index) => active.has(index + 1) ? 1 : 0);
    return Object.freeze({
        structure: typeof structure === 'string' ? structure : 'custom',
        levelCount: definition.levelCount,
        activeCount: unique.length,
        activeLevels: Object.freeze(unique),
        inactiveLevels: Object.freeze(inactiveLevels),
        activationMask: Object.freeze(activationMask),
        mask: activationMask.join(''),
    });
}
export function canonicalActivation(structure, inactiveLevel) {
    const definition = STRUCTURES[structure];
    if (!definition)
        throw new ActivationError('UNKNOWN_STRUCTURE', `Unknown structure: ${structure}`);
    if (!Number.isInteger(inactiveLevel) || inactiveLevel < 1 || inactiveLevel > definition.levelCount) {
        throw new ActivationError('INVALID_INACTIVE_LEVEL', 'inactiveLevel must occupy the selected structure');
    }
    const activeLevels = Array.from({ length: definition.levelCount }, (_, index) => index + 1)
        .filter((level) => level !== inactiveLevel);
    return activationSignature(structure, activeLevels);
}
export function activationTransition(source, target) {
    if (!source?.activationMask || !target?.activationMask || source.levelCount !== target.levelCount) {
        throw new ActivationError('INCOMPATIBLE_SIGNATURES', 'Activation transitions require signatures with equal occupancy');
    }
    const changedLevels = source.activationMask
        .map((value, index) => value === target.activationMask[index] ? null : index + 1)
        .filter(Boolean);
    return Object.freeze({
        from: source.mask,
        to: target.mask,
        changedLevels: Object.freeze(changedLevels),
        deactivated: Object.freeze(changedLevels.filter((level) => source.activationMask[level - 1] === 1)),
        activated: Object.freeze(changedLevels.filter((level) => target.activationMask[level - 1] === 1)),
    });
}
