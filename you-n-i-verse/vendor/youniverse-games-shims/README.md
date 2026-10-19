FILLED IN, no source — browser stand-ins for the Node built-ins her youniverse games import
(node:fs, node:path, node:crypto, node:zlib) and the global Buffer. Used only by play.html through an
import map; Node and the tests use the real built-ins. Her files are not edited.
- fs: in-memory files, .json files mirrored to localStorage key "youniverse.fs.v1" (sync API subset her code
  calls: existsSync, readFileSync, writeFileSync, mkdirSync, readdirSync, renameSync, rmSync).
- path: join, dirname (POSIX).
- crypto: createHash('sha256') via the repo's sha256.mjs; randomUUID is a deterministic counter
  (her code uses it only for temporary file names).
- zlib: inflateSync throws (the avatar is always handed to her photo intake as a 24-bit BMP).
- buffer.mjs: global Buffer subset (isBuffer, from, alloc, byteLength, read/write Int/UInt LE, write, toString).
