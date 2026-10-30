import { completeAnalogy, normalizeVector, operator, toBitString } from './boolean-ato.mjs';
export const HEXAGRAM_COUNT = 64;
export const LINE_COUNT = 6;
export function hexagram(value) {
    const lines = normalizeVector(value, LINE_COUNT);
    const index = Number.parseInt(toBitString(lines), 2);
    return Object.freeze({ index, number: index + 1, lines, bits: toBitString(lines) });
}
export function fromNumber(number) {
    if (!Number.isInteger(number) || number < 1 || number > HEXAGRAM_COUNT)
        throw new RangeError('Hexagram number must be 1 through 64');
    return hexagram(number - 1);
}
export function changeLines(value, lineNumbers = []) {
    const source = hexagram(value);
    const selected = new Set(lineNumbers);
    if ([...selected].some((line) => !Number.isInteger(line) || line < 1 || line > LINE_COUNT))
        throw new RangeError('Changing lines must be 1 through 6');
    // Conventional line numbering is bottom-up; bit strings are displayed top-down.
    const lines = source.lines.map((bit, index) => selected.has(LINE_COUNT - index) ? bit ^ 1 : bit);
    return Object.freeze({ source, changingLines: Object.freeze([...selected].sort()), result: hexagram(lines) });
}
export function relation(a, b, mode = 'equivalence') {
    const left = hexagram(a);
    const right = hexagram(b);
    const transform = operator(left.lines, right.lines, mode);
    return Object.freeze({ left, right, transform, bits: toBitString(transform), mode });
}
export function analogy(a, b, c, mode = 'equivalence') {
    const left = hexagram(a);
    const right = hexagram(b);
    const context = hexagram(c);
    const completed = completeAnalogy(left.lines, right.lines, context.lines, mode);
    return Object.freeze({ a: left, b: right, c: context, relation: completed.relation, result: hexagram(completed.result), mode });
}
export function enumerateSpace() {
    return Object.freeze(Array.from({ length: HEXAGRAM_COUNT }, (_, index) => hexagram(index)));
}
export function neighbors(value) {
    const source = hexagram(value);
    return Object.freeze(Array.from({ length: LINE_COUNT }, (_, index) => changeLines(source.lines, [index + 1]).result));
}
export function shortestPath(start, end) {
    const source = hexagram(start);
    const target = hexagram(end);
    const changingLines = [];
    for (let index = 0; index < LINE_COUNT; index += 1) {
        if (source.lines[index] !== target.lines[index])
            changingLines.push(LINE_COUNT - index);
    }
    return Object.freeze({ source, target, changingLines: Object.freeze(changingLines.sort((a, b) => a - b)), distance: changingLines.length });
}
