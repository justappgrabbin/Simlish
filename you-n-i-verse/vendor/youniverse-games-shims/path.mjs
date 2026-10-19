// FILLED IN, no source: POSIX join/dirname for her youniverse games in the browser.
export function join(...p){const out=[];for(const s of p.join('/').split('/')){if(s===''||s==='.')continue;if(s==='..')out.pop();else out.push(s)}return (String(p[0]).startsWith('/')?'/':'')+out.join('/')}
export function dirname(p){p=String(p);const i=p.replace(/\/+$/,'').lastIndexOf('/');return i<0?'.':i===0?'/':p.slice(0,i)}
export default {join,dirname};
