// The human's own avatar. The chosen image (their studio morph, a photo they pick, or a tile of her supplied
// avatar-states board) is turned into HER visualIdentity by HER photo intake (vendor/youniverse-games/photo/photoIngestion.js).
// FILLED (no source): rgbaToBmp — pixels are written as an uncompressed 24-bit BMP, the same layout her e2e test
// (test/e2e.test.mjs makeTestBmp) writes, because her intake decodes BMP without native code.
import {ingestPhoto} from '../../vendor/youniverse-games/photo/photoIngestion.js';
export function rgbaToBmp(w,h,rgba){const row=Math.ceil(w*3/4)*4,size=54+row*h,b=Buffer.alloc(size);
 b.write('BM',0);b.writeUInt32LE(size,2);b.writeUInt32LE(54,10);b.writeUInt32LE(40,14);b.writeInt32LE(w,18);b.writeInt32LE(h,22);b.writeUInt16LE(1,26);b.writeUInt16LE(24,28);b.writeUInt32LE(0,30);b.writeUInt32LE(row*h,34);
 for(let y=0;y<h;y++)for(let x=0;x<w;x++){const s=((h-1-y)*w+x)*4,o=54+y*row+x*3;b[o]=rgba[s+2];b[o+1]=rgba[s+1];b[o+2]=rgba[s]}return b}
export const avatarIdentity=(bmp,sourceRef)=>ingestPhoto(bmp,{sourceRef});
