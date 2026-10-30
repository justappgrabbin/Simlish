import { equivalence, normalizeVector, toBitString } from './boolean-ato.mjs';
export const TRIGRAMS = Object.freeze({
    creative: Object.freeze({ bits: '111', name: 'The Creative', image: 'Heaven' }),
    abysmal: Object.freeze({ bits: '010', name: 'The Abysmal', image: 'Water' }),
    keepingStill: Object.freeze({ bits: '100', name: 'Keeping Still', image: 'Mountain' }),
    arousing: Object.freeze({ bits: '001', name: 'The Arousing', image: 'Thunder' }),
    gentle: Object.freeze({ bits: '110', name: 'The Gentle', image: 'Wind' }),
    clinging: Object.freeze({ bits: '101', name: 'The Clinging', image: 'Fire' }),
    receptive: Object.freeze({ bits: '000', name: 'The Receptive', image: 'Earth' }),
    joyous: Object.freeze({ bits: '011', name: 'The Joyous', image: 'Lake' }),
});
const HOUSES_SOURCE = [
    ['creative', ['111111', '111110', '111100', '111000', '110000', '100000', '101000', '101111']],
    ['abysmal', ['010010', '010011', '010001', '010101', '011101', '001101', '000101', '000010']],
    ['keepingStill', ['100100', '100101', '100111', '100011', '101011', '111011', '110011', '110100']],
    ['arousing', ['001001', '001000', '001010', '001110', '000110', '010110', '011110', '011001']],
    ['gentle', ['110110', '110111', '110101', '110001', '111001', '101001', '100001', '100110']],
    ['clinging', ['101101', '101100', '101110', '101010', '100010', '110010', '111010', '111101']],
    ['receptive', ['000000', '000001', '000011', '000111', '001111', '011111', '010111', '010000']],
    ['joyous', ['011011', '011010', '011000', '011100', '010100', '000100', '001100', '001011']],
];
export const HOUSES = Object.freeze(HOUSES_SOURCE.map(([id, members], houseIndex) => Object.freeze({
    id,
    number: houseIndex + 1,
    name: `House of ${TRIGRAMS[id].name}`,
    base: members[0],
    members: Object.freeze(members.map((bits, rowIndex) => Object.freeze({ house: houseIndex + 1, row: rowIndex + 1, bits }))),
})));
const BY_BITS = new Map(HOUSES.flatMap((house) => house.members.map((member) => [member.bits, Object.freeze({ ...member, houseId: house.id, houseName: house.name })])));
export function locateHexagram(value) {
    const bits = toBitString(normalizeVector(value, 6));
    const location = BY_BITS.get(bits);
    if (!location)
        throw new Error(`Hexagram is absent from Klein's eight-house state space: ${bits}`);
    return location;
}
export function houseOperator(sourceHouse, targetHouse) {
    const source = typeof sourceHouse === 'number' ? HOUSES[sourceHouse - 1] : HOUSES.find((house) => house.id === sourceHouse);
    const target = typeof targetHouse === 'number' ? HOUSES[targetHouse - 1] : HOUSES.find((house) => house.id === targetHouse);
    if (!source || !target)
        throw new RangeError('Unknown source or target house');
    return Object.freeze({ source: source.id, target: target.id, bits: toBitString(equivalence(source.base, target.base)) });
}
export function transformHouse(sourceHouse, targetHouse) {
    const source = typeof sourceHouse === 'number' ? HOUSES[sourceHouse - 1] : HOUSES.find((house) => house.id === sourceHouse);
    const target = typeof targetHouse === 'number' ? HOUSES[targetHouse - 1] : HOUSES.find((house) => house.id === targetHouse);
    const ato = houseOperator(source.id, target.id).bits;
    const products = source.members.map((member, index) => Object.freeze({
        source: member,
        operator: ato,
        resultBits: toBitString(equivalence(member.bits, ato)),
        expected: target.members[index],
        valid: toBitString(equivalence(member.bits, ato)) === target.members[index].bits,
    }));
    return Object.freeze({ source: source.id, target: target.id, operator: ato, products: Object.freeze(products), valid: products.every((item) => item.valid) });
}
export function rowAnalogy(subjectHouse, subjectRow, targetHouse) {
    const source = typeof subjectHouse === 'number' ? HOUSES[subjectHouse - 1] : HOUSES.find((house) => house.id === subjectHouse);
    const target = typeof targetHouse === 'number' ? HOUSES[targetHouse - 1] : HOUSES.find((house) => house.id === targetHouse);
    if (!source || !target || !Number.isInteger(subjectRow) || subjectRow < 1 || subjectRow > 8)
        throw new RangeError('A valid house, row, and target house are required');
    const sourceBase = source.members[0].bits;
    const sourceMember = source.members[subjectRow - 1].bits;
    const targetBase = target.members[0].bits;
    const relation = toBitString(equivalence(sourceBase, sourceMember));
    const result = toBitString(equivalence(targetBase, relation));
    return Object.freeze({ sourceHouse: source.id, sourceRow: subjectRow, targetHouse: target.id, relation, result, location: locateHexagram(result) });
}
export function trigramAnalogy(a, b, c) {
    const relation = toBitString(equivalence(a, b));
    const result = toBitString(equivalence(c, relation));
    const match = Object.entries(TRIGRAMS).find(([, value]) => value.bits === result);
    return Object.freeze({ a, b, c, relation, result, trigram: match ? Object.freeze({ id: match[0], ...match[1] }) : null });
}
