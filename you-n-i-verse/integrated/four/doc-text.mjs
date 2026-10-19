// Document text for the four primitives. FILLED IN by the builder (not her code): md read as UTF-8,
// docx read by a minimal zip reader (central directory + deflate-raw via DecompressionStream) and
// word/document.xml paragraphs (<w:p>) joined with newlines. PDF: no pure-JS PDF parser is in her code;
// the box ran poppler `pdftotext -layout` and the text is committed next to the PDF's sha256 (INVESTIGATING an in-app PDF path).
const u16=(b,o)=>b[o]|b[o+1]<<8, u32=(b,o)=>(b[o]|b[o+1]<<8|b[o+2]<<16|b[o+3]<<24)>>>0;
async function inflateRaw(bytes){const s=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate-raw'));return new Uint8Array(await new Response(s).arrayBuffer());}
export async function unzipEntry(bytes,name){
  let e=bytes.length-22; while(e>=0&&u32(bytes,e)!==0x06054b50)e--; if(e<0)throw new Error('not a zip');
  let p=u32(bytes,e+16); const n=u16(bytes,e+10); const dec=new TextDecoder();
  for(let i=0;i<n;i++){const method=u16(bytes,p+10),csize=u32(bytes,p+20),nl=u16(bytes,p+28),xl=u16(bytes,p+30),cl=u16(bytes,p+32),lo=u32(bytes,p+42);
    const fname=dec.decode(bytes.subarray(p+46,p+46+nl)); p+=46+nl+xl+cl;
    if(fname!==name)continue; const ds=lo+30+u16(bytes,lo+26)+u16(bytes,lo+28); const data=bytes.subarray(ds,ds+csize);
    if(method===0)return data; if(method===8)return inflateRaw(data); throw new Error('zip method '+method);}
  throw new Error('missing '+name);
}
const ENT={amp:'&',lt:'<',gt:'>',quot:'"',apos:"'"};
export function docxXmlToText(xml){return xml.split(/<\/w:p>/).map(p=>p.replace(/<w:tab\/>/g,'\t').replace(/<w:br\/>/g,'\n').replace(/<[^>]+>/g,'').replace(/&(#x?[0-9a-fA-F]+|\w+);/g,(m,k)=>k[0]==='#'?String.fromCodePoint(k[1]==='x'?parseInt(k.slice(2),16):+k.slice(1)):(ENT[k]??m))).join('\n').replace(/\n{3,}/g,'\n\n').trim();}
export async function docText(name,bytes){
  if(/\.docx$/i.test(name))return docxXmlToText(new TextDecoder().decode(await unzipEntry(bytes,'word/document.xml')));
  if(/\.(md|txt)$/i.test(name))return new TextDecoder().decode(bytes);
  throw new Error('INVESTIGATING: no in-app reader for '+name+' (PDF text is box-extracted by pdftotext)');
}
// Semantic units: sentences/lines (split on . ! ? or newline), trimmed, >= 3 word characters. FILLED IN unit rule.
export function semanticUnits(text){return text.split(/(?<=[.!?])\s+|\n+/).map(s=>s.replace(/\s+/g,' ').trim()).filter(s=>(s.match(/\w/g)||[]).length>=3);}
