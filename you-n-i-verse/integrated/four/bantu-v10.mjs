// Port of constants and formulas from her SWARM-OS v10 BANTU (integrated/reference/her/swarm_os_bantu.v10.py.txt,
// sha256 eb303f8c…, from swarm-os-v14/x/v10/swarm_os_bantu.py). Values are copied verbatim; line numbers in comments.
// Her file draws these with random.choice/random.random; here the INPUT x is supplied by the caller (see qualities.mjs)
// and nothing is random.
// v10 line 63
export const BANTU_ASCII_SHAPES = Object.freeze([...' .:-=+*#%@$&', ...'()[]{}<>|/\\^~_', ...'─│┌┐└┘├┤┬┴┼═║╔╗╚╝']);
// v10 line 76-77 / 78
export const BANTU_LAYER_NAMES = Object.freeze(['BINARY', 'ASCII', 'NATO', 'TRANSLIT', 'UNICODE']);
export const BANTU_LAYER_COLORS = Object.freeze([[255, 107, 107], [78, 205, 196], [69, 183, 209], [150, 206, 180], [254, 202, 87]]);
// v10 line 101: `20 * (1000 ** random.random())`; line 317: `20 * (1000 ** (idx/8))`. x in [0,1].
export const bantuSoundFreq = (x) => 20 * (1000 ** x);
