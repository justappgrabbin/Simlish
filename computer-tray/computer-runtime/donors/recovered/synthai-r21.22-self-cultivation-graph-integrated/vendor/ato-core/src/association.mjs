import { canonicalAddress, normalizeAddress, gateState, ADDRESS_FIELDS } from './address-space.mjs';
import { operator, toBitString } from './boolean-ato.mjs';
import { locateHexagram } from './klein-iching.mjs';
export class AssociationError extends Error {
    constructor(code, message, details = {}) { super(message); this.name = 'AssociationError'; this.code = code; this.details = details; }
}
const freezeRecord = (value) => Object.freeze(Object.fromEntries(Object.entries(value)));
export class StateSpaceAssociationResolver {
    constructor({ mappingVersion = 'ato.state-space-association.v1', gateOperator = 'equivalence' } = {}) {
        this.mappingVersion = mappingVersion;
        this.gateOperator = gateOperator;
    }
    resolve(left, right, { context = null } = {}) {
        if (!left?.address || !right?.address)
            throw new AssociationError('UNADDRESSED_TOOL', 'Both tools require dimensional addresses');
        const a = normalizeAddress(left.address, { mode: left.address.mode || 'macro', allowUnresolved: true });
        const b = normalizeAddress(right.address, { mode: right.address.mode || 'macro', allowUnresolved: true });
        const shared = {}, differences = {};
        for (const field of ADDRESS_FIELDS) {
            if (a[field] === undefined && b[field] === undefined)
                continue;
            if (a[field] === b[field])
                shared[field] = a[field];
            else
                differences[field] = Object.freeze({ left: a[field] ?? null, right: b[field] ?? null });
        }
        const leftGate = gateState(a), rightGate = gateState(b);
        const transform = operator(leftGate.bits, rightGate.bits, this.gateOperator);
        const transformBits = toBitString(transform);
        const transformGate = Number.parseInt(transformBits, 2) + 1;
        const leftLocation = locateHexagram(leftGate.bits);
        const rightLocation = locateHexagram(rightGate.bits);
        const relations = [];
        if (a.gate === b.gate)
            relations.push(Object.freeze({ id: 'same-gate', gate: a.gate }));
        if (leftLocation.house === rightLocation.house)
            relations.push(Object.freeze({ id: 'same-house', house: leftLocation.house, houseId: leftLocation.houseId }));
        if (leftLocation.row === rightLocation.row)
            relations.push(Object.freeze({ id: 'corresponding-row', row: leftLocation.row }));
        if (Object.keys(shared).length)
            relations.push(Object.freeze({ id: 'shared-coordinates', fields: Object.freeze(Object.keys(shared).sort()) }));
        relations.push(Object.freeze({ id: `${this.gateOperator}-operator`, bits: transformBits, resultantGate: transformGate }));
        const unresolved = ['line', 'color', 'tone', 'base'].filter((field) => shared[field] === undefined);
        const emergenceAddress = normalizeAddress({ mode: 'macro', gate: transformGate, ...shared }, { mode: 'macro', allowUnresolved: true });
        return Object.freeze({
            mappingVersion: this.mappingVersion,
            left: canonicalAddress(a),
            right: canonicalAddress(b),
            shared: freezeRecord(shared),
            differences: freezeRecord(differences),
            gateRelation: Object.freeze({ mode: this.gateOperator, left: leftGate.bits, right: rightGate.bits, operator: transformBits, resultantGate: transformGate }),
            klein: Object.freeze({ left: leftLocation, right: rightLocation }),
            relations: Object.freeze(relations),
            emergenceSpot: Object.freeze({ address: emergenceAddress, addressKey: canonicalAddress(emergenceAddress), resolved: unresolved.length === 0, unresolved: Object.freeze(unresolved) }),
            context,
        });
    }
}
