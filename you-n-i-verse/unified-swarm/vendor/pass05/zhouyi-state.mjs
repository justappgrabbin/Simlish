/** Zhouyi six-line operations. Bit zero is the BOTTOM line; 1=solid, 0=broken.
 * This is a modern computational encoding, not a claim about ancient binary notation.
 */
export const TRIGRAMS=Object.freeze(['Earth','Thunder','Water','Lake','Mountain','Fire','Wind','Heaven']);
export function assertSix(n){if(!Number.isInteger(n)||n<0||n>63)throw new RangeError('six-line state must be integer 0..63');return n;}
export function lines(n){assertSix(n);return Array.from({length:6},(_,i)=>(n>>i)&1);}
export function fromLines(xs){if(!Array.isArray(xs)||xs.length!==6||xs.some(x=>x!==0&&x!==1))throw new TypeError('six binary lines required');return xs.reduce((n,b,i)=>n|(b<<i),0);}
export function changeLine(n,bottomBasedLine){assertSix(n);if(!Number.isInteger(bottomBasedLine)||bottomBasedLine<1||bottomBasedLine>6)throw new RangeError('line must be 1..6, bottom first');return n^(1<<(bottomBasedLine-1));}
export function complement(n){return assertSix(n)^63;}
export function invert(n){return fromLines(lines(n).reverse());}
export function trigrams(n){assertSix(n);return {lower:TRIGRAMS[n&7],upper:TRIGRAMS[(n>>3)&7],lowerBits:n&7,upperBits:(n>>3)&7};}
export function transform(n,movingMask){assertSix(n);assertSix(movingMask);return n^movingMask;}
export function binaryBottomFirst(n){return lines(n).join('');}
export function binaryTopFirst(n){return lines(n).reverse().join('');}
export class ZhouyiStateJournal {
  #events=[];#value;
  constructor(initial=0){this.#value=assertSix(initial);this.#events.push(Object.freeze({index:0,action:'init',from:null,to:initial}));}
  get value(){return this.#value;}
  history(){return this.#events.map(e=>({...e}));}
  apply(action,argument){const from=this.#value;let to;
    if(action==='line')to=changeLine(from,argument);
    else if(action==='complement')to=complement(from);
    else if(action==='invert')to=invert(from);
    else if(action==='moving-mask')to=transform(from,argument);
    else if(action==='set')to=assertSix(argument);
    else throw Error('unknown action');
    this.#value=to;const event=Object.freeze({index:this.#events.length,action,argument:argument??null,from,to});this.#events.push(event);return {...event};
  }
}
