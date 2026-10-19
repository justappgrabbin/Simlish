// Photo ingestion pipeline.
// Accepts a file path or Buffer (BMP 24-bit, binary/ascii PPM, or PNG/JPEG via
// real pixel decoding), validates magic bytes + minimum size, extracts a
// VisualIdentity. Errors throw visibly — no silent fallbacks.

import { readFileSync } from 'node:fs';
import { decodePNG } from './pngDecoder.js';
import { sha256Hex } from '../util/hash.js';

const MIN_BYTES = 54; // smaller than any real BMP header + a pixel

function detectFormat(buf) {
  if (buf.length < 4) throw new Error('photoIngestion: buffer too small to detect format');
  if (buf[0] === 0x42 && buf[1] === 0x4d) return 'bmp'; // 'BM'
  if (buf[0] === 0x89 && buf[1] === 0x50 && buf[2] === 0x4e && buf[3] === 0x47) return 'png';
  if (buf[0] === 0xff && buf[1] === 0xd8) return 'jpeg';
  if (buf[0] === 0x50 && (buf[1] === 0x36 || buf[1] === 0x33)) return 'ppm'; // P6 / P3
  throw new Error('photoIngestion: unsupported image format (magic bytes not BMP/PNG/JPEG/PPM)');
}

// Decode uncompressed 24-bit BMP into RGB pixels.
function decodeBmp(buf) {
  if (buf.length < 54) throw new Error('photoIngestion: truncated BMP header');
  const dataOffset = buf.readUInt32LE(10);
  const width = buf.readInt32LE(18);
  const height = buf.readInt32LE(22);
  const bpp = buf.readUInt16LE(28);
  const compression = buf.readUInt32LE(30);
  if (bpp !== 24 || compression !== 0) {
    throw new Error(`photoIngestion: BMP must be 24-bit uncompressed (got bpp=${bpp}, compression=${compression})`);
  }
  if (width <= 0 || height === 0) throw new Error('photoIngestion: BMP has invalid dimensions');
  const h = Math.abs(height);
  validateDimensions(width, h);
  if (dataOffset < 54) throw new Error('photoIngestion: invalid BMP pixel offset');
  const rowSize = Math.ceil((width * 3) / 4) * 4;
  if (dataOffset + rowSize * h > buf.length) throw new Error('photoIngestion: BMP pixel data truncated');
  const pixels = new Uint8Array(width * h * 3);
  for (let y = 0; y < h; y++) {
    const srcRow = height > 0 ? h - 1 - y : y; // bottom-up when height positive
    for (let x = 0; x < width; x++) {
      const s = dataOffset + srcRow * rowSize + x * 3;
      const d = (y * width + x) * 3;
      pixels[d] = buf[s + 2]; pixels[d + 1] = buf[s + 1]; pixels[d + 2] = buf[s]; // BGR->RGB
    }
  }
  return { width, height: h, pixels };
}

// Decode binary (P6) or ascii (P3) PPM.
function decodePpm(buf) {
  const text = buf.toString('latin1');
  const headerMatch = text.match(/^P([36])\s+(\d+)\s+(\d+)\s+(\d+)\s/);
  if (!headerMatch) throw new Error('photoIngestion: malformed PPM header');
  const [, kind, w, h, max] = headerMatch;
  const width = +w, height = +h, maxval = +max;
  if (width <= 0 || height <= 0 || maxval <= 0 || maxval > 255) {
    throw new Error('photoIngestion: invalid PPM dimensions/maxval');
  }
  validateDimensions(width, height);
  const pixels = new Uint8Array(width * height * 3);
  if (kind === '6') {
    const start = headerMatch[0].length;
    if (start + pixels.length > buf.length) throw new Error('photoIngestion: PPM pixel data truncated');
    for (let i = 0; i < pixels.length; i++) pixels[i] = (buf[start + i] * 255) / maxval;
  } else {
    const body = text.slice(headerMatch[0].length).trim().split(/\s+/).map(Number);
    if (body.length < pixels.length) throw new Error('photoIngestion: P3 PPM pixel data truncated');
    for (let i = 0; i < pixels.length; i++) pixels[i] = (body[i] * 255) / maxval;
  }
  return { width, height, pixels };
}

// Decode actual image pixels; compressed file bytes are never a pixel substitute.
function decodeCompressed(buf, format) {
  if (format === 'png') return decodePNG(buf);
  throw new Error('JPEG intake requires asynchronous createPlayerFromPhoto() normalization');
}

function validateDimensions(width, height) {
  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width < 1 || height < 1 || width * height > 16000000) {
    throw new Error('photoIngestion: invalid dimensions or image exceeds 16 megapixels');
  }
}

// 4-bits-per-channel histogram palette: top-N quantized colors by frequency.
function extractPalette(pixels, topN = 5) {
  const bins = new Map();
  for (let i = 0; i < pixels.length; i += 3) {
    const key = `${pixels[i] >> 4},${pixels[i + 1] >> 4},${pixels[i + 2] >> 4}`;
    bins.set(key, (bins.get(key) || 0) + 1);
  }
  const sorted = [...bins.entries()].sort((a, b) => b[1] - a[1]).slice(0, topN);
  const total = pixels.length / 3;
  return sorted.map(([key, count]) => {
    const [r, g, b] = key.split(',').map((v) => (parseInt(v, 10) << 4) + 8);
    return {
      rgb: [r, g, b],
      hex: '#' + [r, g, b].map((v) => v.toString(16).padStart(2, '0')).join(''),
      share: +(count / total).toFixed(4),
    };
  });
}

function meanLuminance(pixels) {
  let sum = 0;
  const n = pixels.length / 3;
  for (let i = 0; i < pixels.length; i += 3) {
    sum += 0.2126 * pixels[i] + 0.7152 * pixels[i + 1] + 0.0722 * pixels[i + 2];
  }
  return +(sum / n / 255).toFixed(4);
}

function shannonEntropy(bytes) {
  const counts = new Array(256).fill(0);
  for (const b of bytes) counts[b]++;
  let e = 0;
  for (const c of counts) {
    if (c === 0) continue;
    const p = c / bytes.length;
    e -= p * Math.log2(p);
  }
  return +e.toFixed(4);
}

/**
 * Ingest a photo.
 * @param {string|Buffer} input - file path or image buffer
 * @param {object} [opts] - { sourceRef } recorded on the VisualIdentity
 * @returns {VisualIdentity} {sourceImageRef, format, width, height, palette,
 *   luminance, entropy, aspect, fingerprint, features}
 */
// Decode any supported input to an RGB raster {width, height, pixels, approximate?}.
// Exported for identity/visualDNA.js (facial anchor extraction); ingestPhoto's
// public contract is unchanged.
export function decodePhotoRaster(input) {
  let buf;
  if (Buffer.isBuffer(input)) buf = input;
  else if (typeof input === 'string') buf = readFileSync(input); // throws visibly if missing
  else throw new Error('photoIngestion: input must be a file path or Buffer');
  if (buf.length < MIN_BYTES) {
    throw new Error(`photoIngestion: image too small (${buf.length} bytes < ${MIN_BYTES})`);
  }
  if (buf.length > 12 * 1024 * 1024) throw new Error('photoIngestion: image exceeds 12 MB');
  const format = detectFormat(buf);
  let raster;
  if (format === 'bmp') raster = decodeBmp(buf);
  else if (format === 'ppm') raster = decodePpm(buf);
  else raster = decodeCompressed(buf, format);
  return { ...raster, format, buffer: buf };
}

export function ingestPhoto(input, opts = {}) {
  const decoded = decodePhotoRaster(input);
  const buf = decoded.buffer;
  const raster = decoded;
  const format = decoded.format;
  const sourceRef = Buffer.isBuffer(input)
    ? (opts.sourceRef || `buffer:${sha256Hex(buf).slice(0, 12)}`)
    : input;

  const palette = extractPalette(raster.pixels);
  const luminance = meanLuminance(raster.pixels);
  const entropy = shannonEntropy(buf);
  const aspect = +(raster.width / raster.height).toFixed(4);
  const fingerprint = sha256Hex({
    format, w: raster.width, h: raster.height,
    palette: palette.map((p) => p.hex), luminance, entropy, bytes: sha256Hex(buf).slice(0, 32),
  });

  return {
    sourceImageRef: sourceRef,
    format,
    width: raster.width,
    height: raster.height,
    approximateDecode: !!raster.approximate,
    palette,
    luminance,
    entropy,
    aspect,
    fingerprint,
    features: {
      dominantHue: palette[0].hex,
      paletteDiversity: palette.length,
      darkImage: luminance < 0.35,
      vividImage: entropy > 6.5,
      portraitAspect: aspect < 0.9,
    },
  };
}
