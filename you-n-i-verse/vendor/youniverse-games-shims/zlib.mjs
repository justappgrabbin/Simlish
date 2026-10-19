// FILLED IN, no source: her pngDecoder imports inflateSync; the play page always hands her intake a 24-bit BMP, so PNG is not decoded in the browser.
export function inflateSync(){throw new Error('PNG decoding is not available in the browser shim; the avatar is passed as BMP')}
export default {inflateSync};
