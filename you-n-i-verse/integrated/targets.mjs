// THREE unrelated targets, written only as organizations of the shared primitives (data: netlists). No function in this file executes anything.
// Each is reduced to a six-bit stream, resolved to an address, composed (definitions instantiated as primitives), executed by population.mjs, remembered and reorganized.
import {assemble} from './assemble.mjs';
const HA={name:'HalfAdder',inputs:['a','b'],cells:[{n:'s',d:'Movement',h:'XOR',a:'a',b:'b'},{n:'c',d:'Movement',h:'AND',a:'a',b:'b'}],outputs:['s','c']};
const FA={name:'FullAdder',inputs:['a','b','cin'],cells:[
 {n:'h1',call:'HalfAdder',args:['a','b'],outs:['s','c']},{n:'h2',call:'HalfAdder',args:['h1.s','cin'],outs:['s','c']},
 {n:'cout',d:'Movement',h:'OR',a:'h1.c',b:'h2.c'}],outputs:['h2.s','cout']};
// 1) NUMBER: an 8-bit running total. Evolution cells accumulate (state advances by the change applied); full adders (instantiated organizations) make the carries.
const total=(()=>{const x=[...Array(8)].map((_,i)=>'x'+i),cells=[{n:'zero',d:'Movement',h:'ZERO',a:'x0',b:'x0'}];
 for(let i=0;i<8;i++){const cin=i?'fa'+(i-1)+'.cout':'zero';
  cells.push({n:'fa'+i,call:'FullAdder',args:['x'+i,'acc'+i,cin],outs:['s','cout']});          // carry from (input, previous total bit, carry in); 'acc' is later = previous tick
  cells.push({n:'ch'+i,d:'Movement',h:'XOR',a:'x'+i,b:cin});                                  // the change to apply
  cells.push({n:'acc'+i,d:'Evolution',h:'XOR',a:'ch'+i,b:'zero'})}                             // state advances by that change
 return assemble([HA,FA,{name:'RunningTotal',inputs:x,cells,outputs:[...Array(8)].map((_,i)=>'acc'+i)}])})();
// 2) TEXT: capitalise the first letter of each word. A Being cell holds "a word boundary was seen"; Movement cells are the logic; a Design cell routes bit 5.
const words=(()=>{const b=[...Array(8)].map((_,i)=>'b'+i),cells=[
 {n:'start',d:'Movement',h:'NOTA',a:'started',b:'started'},                                   // started is later: its previous value; 0 on the first tick
 {n:'bnd',d:'Movement',h:'OR',a:'start',b:'bnd_held'},                                       // first character, or a boundary is being held
 {n:'o01',d:'Movement',h:'OR',a:'b0',b:'b1'},{n:'o23',d:'Movement',h:'OR',a:'b2',b:'b3'},{n:'o4_',d:'Movement',h:'OR',a:'b4',b:'b6'},{n:'o_7',d:'Movement',h:'OR',a:'o4_',b:'b7'},
 {n:'oA',d:'Movement',h:'OR',a:'o01',b:'o23'},{n:'oB',d:'Movement',h:'OR',a:'oA',b:'o_7'},
 {n:'isSp',d:'Movement',h:'ANDNOTB',a:'b5',b:'oB'},                                         // 0x20
 {n:'isLo',d:'Movement',h:'AND',a:'b5',b:'b6'},                                              // 0x60..0x7f
 {n:'cap',d:'Movement',h:'AND',a:'isLo',b:'bnd'},
 {n:'one',d:'Movement',h:'ONE',a:'b0',b:'b0'},{n:'zero',d:'Movement',h:'ZERO',a:'b0',b:'b0'},
 {n:'seen',d:'Movement',h:'OR',a:'isSp',b:'b6'},                                              // a character class that updates the hold
 {n:'bnd_held',d:'Being',h:0,a:'isSp',en:'seen'},                                            // holds the boundary flag through characters that are not letters or spaces
 {n:'out5',d:'Design',h:0,a:'b5',b:'zero',sel:'cap'},
 {n:'started',d:'Movement',h:'ONE',a:'b0',b:'b0'}];
 return assemble([{name:'WordCase',inputs:b,cells,outputs:['b0','b1','b2','b3','b4','out5','b6','b7']}])})();
// 3) PATTERN: a six-bit counter (Evolution cells toggling by carry), one pixel per tick: on where x AND y = 0 for x=low 3 bits, y=high 3 bits.
const patternDef=(()=>{ // every read of a counter bit n_i comes BEFORE n_i in the order, so it sees the previous tick (index 0 on the first tick)
 const cells=[{n:'one',d:'Movement',h:'ONE',a:'n0',b:'n0'},{n:'c1',d:'Movement',h:'A',a:'n0',b:'n0'}];
 for(let i=2;i<6;i++)cells.push({n:'c'+i,d:'Movement',h:'AND',a:'c'+(i-1),b:'n'+(i-1)});
 cells.push({n:'z0',d:'Movement',h:'AND',a:'n0',b:'n3'},{n:'z1',d:'Movement',h:'AND',a:'n1',b:'n4'},{n:'z2',d:'Movement',h:'AND',a:'n2',b:'n5'},
  {n:'z01',d:'Movement',h:'OR',a:'z0',b:'z1'},{n:'any',d:'Movement',h:'OR',a:'z01',b:'z2'},{n:'pixel',d:'Movement',h:'NOTA',a:'any',b:'any'});
 for(let i=0;i<6;i++)cells.push({n:'n'+i,d:'Evolution',h:'A',a:i===0?'one':'c'+i,b:'one'});   // state advances by the carry into it
 return assemble([{name:'Pattern',inputs:[],cells,outputs:['pixel']}])})();
export const TARGETS=Object.freeze({
 number:{name:'running-total.prim',atoms:total,note:'8-bit running total; input per tick: 8 bits (LSB first); output per tick: the total'},
 text:{name:'word-case.prim',atoms:words,note:'capitalise the first letter of each word; input per tick: one byte (LSB first); output: the byte'},
 pattern:{name:'pattern.prim',atoms:patternDef,note:'one pixel per tick, 64 ticks make an 8x8 picture; no input'}});
