import { inflateSync } from 'node:zlib';

const SIGNATURE = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
const CRC_TABLE = Array.from({ length: 256 }, (_, n) => {
  let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});
function crc(bytes) { let c = 0xffffffff; for (const b of bytes) c = CRC_TABLE[(c ^ b) & 255] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; }
const paeth = (a, b, c) => { const p = a + b - c, da = Math.abs(p - a), db = Math.abs(p - b), dc = Math.abs(p - c); return da <= db && da <= dc ? a : db <= dc ? b : c; };

// Bounded true-pixel decoder for normalized 8-bit PNGs. General uploads go
// through Pillow's asynchronous normalization before reaching this engine.
export function decodePNG(buf) {
  if (!buf.subarray(0, 8).equals(SIGNATURE)) throw new Error('Invalid PNG signature');
  let width, height, color, offset = 8, ended = false; const parts = [];
  while (offset < buf.length) {
    if (offset + 12 > buf.length) throw new Error('Truncated PNG chunk');
    const len = buf.readUInt32BE(offset), end = offset + 12 + len;
    if (end > buf.length) throw new Error('Truncated PNG chunk data');
    const type = buf.toString('ascii', offset + 4, offset + 8), data = buf.subarray(offset + 8, offset + 8 + len);
    if (crc(buf.subarray(offset + 4, offset + 8 + len)) !== buf.readUInt32BE(offset + 8 + len)) throw new Error('PNG checksum failed');
    if (type === 'IHDR') {
      if (offset !== 8 || len !== 13) throw new Error('Invalid PNG header');
      width = data.readUInt32BE(0); height = data.readUInt32BE(4); color = data[9];
      if (width < 1 || height < 1 || width * height > 16000000) throw new Error('PNG exceeds 16 megapixels or has invalid dimensions');
      if (data[8] !== 8 || ![0, 2, 4, 6].includes(color) || data[10] || data[11] || data[12]) throw new Error('Normalize this PNG with createPlayerFromPhoto() before intake');
    } else if (type === 'IDAT') parts.push(data);
    else if (type === 'IEND') { ended = true; break; }
    offset = end;
  }
  if (!ended || !width || !parts.length) throw new Error('Incomplete PNG');
  const channels = { 0: 1, 2: 3, 4: 2, 6: 4 }[color], rowBytes = width * channels;
  const expected = (rowBytes + 1) * height;
  const filtered = inflateSync(Buffer.concat(parts), { maxOutputLength: expected });
  if (filtered.length !== expected) throw new Error('PNG pixel length mismatch');
  const raw = new Uint8Array(width * height * channels);
  for (let y = 0; y < height; y++) {
    const f = filtered[y * (rowBytes + 1)]; if (f > 4) throw new Error('Invalid PNG filter');
    for (let x = 0; x < rowBytes; x++) {
      const i = y * rowBytes + x, left = x >= channels ? raw[i - channels] : 0;
      const up = y ? raw[i - rowBytes] : 0, diagonal = y && x >= channels ? raw[i - rowBytes - channels] : 0;
      const prediction = [0, left, up, (left + up) >> 1, paeth(left, up, diagonal)][f];
      raw[i] = (filtered[y * (rowBytes + 1) + 1 + x] + prediction) & 255;
    }
  }
  const pixels = new Uint8Array(width * height * 3);
  for (let i = 0; i < width * height; i++) {
    const source = i * channels, alpha = color === 4 ? raw[source + 1] / 255 : color === 6 ? raw[source + 3] / 255 : 1;
    for (let c = 0; c < 3; c++) pixels[i * 3 + c] = Math.round(raw[source + (color === 0 || color === 4 ? 0 : c)] * alpha + 255 * (1 - alpha));
  }
  return { width, height, pixels };
}
