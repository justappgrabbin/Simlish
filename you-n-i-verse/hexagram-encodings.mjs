import {gateToFuXiDecimal,fuXiDecimalToGate} from './vendor/kingwen.mjs';
export const HEXAGRAM_ENCODING_SOURCE='user-hexagram-csv/2026-10-12';
const trigrams={'111':'heaven','000':'earth','100':'thunder','010':'water','001':'mountain','011':'wind','110':'lake','101':'fire'};
const nato='Alfa Bravo Charlie Delta Echo Foxtrot Golf Hotel India Juliett Kilo Lima Mike November Oscar Papa Quebec Romeo Sierra Tango Uniform Victor Whiskey Xray Yankee Zulu'.split(' ');
export function hexagramEncoding(row){
 const decimal=gateToFuXiDecimal(row.gate),binaryBottomToTop=decimal.toString(2).padStart(6,'0').split('').reverse().join('');
 if(typeof row.pinyin!=='string'||!row.pinyin.trim())throw Error('Pinyin required');
 // The supplied ASCII/NATO columns transliterate ü to u; retain both names.
 const asciiName=row.pinyin.replaceAll('ü','u');if(!/^[a-z ]+$/.test(asciiName))throw Error('Unsupported ASCII transliteration');
 if(!/^#[\dA-Fa-f]{6}$/.test(row.illustrativeColor))throw Error('Invalid illustrative RGB value');
 const unicodeDecimal=0x4dc0+row.gate-1;
 return Object.freeze({...row,representation:'embodied',source:HEXAGRAM_ENCODING_SOURCE,hexagram:String.fromCodePoint(unicodeDecimal),binaryBottomToTop,displayDecimal:parseInt(binaryBottomToTop,2),fuxiDecimalBottomBitZero:decimal,asciiName,asciiBytes:Object.freeze([...asciiName].map(c=>c.charCodeAt(0))),nato:asciiName.split(' ').map(w=>[...w].map(c=>nato[c.charCodeAt(0)-97]).join(' ')).join(' / '),unicodeCodePoint:'U+'+unicodeDecimal.toString(16).toUpperCase(),unicodeDecimal,illustrativeSoundHz:220+5*parseInt(binaryBottomToTop,2),illustrative:true,shape:trigrams[binaryBottomToTop.slice(3)]+' over '+trigrams[binaryBottomToTop.slice(0,3)]});
}
export function displayDecimalToGate(value){if(!Number.isInteger(value)||value<0||value>63)throw RangeError('Display decimal must be 0..63');return fuXiDecimalToGate(parseInt(value.toString(2).padStart(6,'0').split('').reverse().join(''),2));}
export function changeEncodingLine(row,line){if(!Number.isInteger(line)||line<1||line>6)throw RangeError('Line must be 1..6');const decimal=gateToFuXiDecimal(row.gate)^(1<<(line-1));return fuXiDecimalToGate(decimal);}
