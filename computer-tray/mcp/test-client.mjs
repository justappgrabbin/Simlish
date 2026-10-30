// Tests stdio + HTTP + SSE transports end to end.
import { spawn } from 'node:child_process';
const call=(id,method,params)=>({jsonrpc:'2.0',id,method,params});
const msgs=[call(1,'initialize',{protocolVersion:'2025-06-18',capabilities:{},clientInfo:{name:'t',version:'1'}}),{jsonrpc:'2.0',method:'notifications/initialized'},call(2,'tools/list',{}),
 call(3,'tools/call',{name:'list_apps',arguments:{}}),call(4,'tools/call',{name:'resolve_address',arguments:{address:'18.5.3.4.2'}}),
 call(5,'tools/call',{name:'write_app_file',arguments:{app_id:'paper',file:'mcp-test.txt',content:'hello',address:'99.1.1.1.1'}}),
 call(6,'tools/call',{name:'write_app_file',arguments:{app_id:'paper',file:'mcp-test.txt',content:'hello from MCP',address:'18.5.3.4.2'}}),
 call(7,'tools/call',{name:'read_app_file',arguments:{app_id:'paper',file:'mcp-test.txt'}}),call(8,'tools/call',{name:'read_app_file',arguments:{app_id:'paper',file:'../../index.html'}}),
 call(9,'tools/call',{name:'search',arguments:{query:'paper'}})];
const show=(r)=>console.log(r.id, r.error?.message || (r.result.tools? r.result.tools.map(t=>t.name).join(','): r.result.serverInfo? 'init '+r.result.serverInfo.name : (r.result.isError?'ERROR: ':'')+(r.result.content?.[0]?.text||'').slice(0,110).replace(/\s+/g,' ')));
console.log('--- stdio');
const p=spawn('node',[new URL('./server.mjs',import.meta.url).pathname]);let buf='';const got=[];
p.stdout.on('data',d=>{buf+=d;let i;while((i=buf.indexOf('\n'))>=0){got.push(JSON.parse(buf.slice(0,i)));buf=buf.slice(i+1)}});
for(const m of msgs)p.stdin.write(JSON.stringify(m)+'\n');await new Promise(r=>setTimeout(r,1500));got.forEach(show);p.kill();
console.log('--- http');
const h=spawn('node',[new URL('./server.mjs',import.meta.url).pathname,'--http','--port','8799']);await new Promise(r=>setTimeout(r,800));
for(const m of msgs){const r=await fetch('http://localhost:8799/mcp',{method:'POST',headers:{'content-type':'application/json',accept:'application/json, text/event-stream'},body:JSON.stringify(m)});if(r.status===200)show(await r.json());}
console.log('--- sse');
const ctl=new AbortController();const s=await fetch('http://localhost:8799/sse',{signal:ctl.signal});const rd=s.body.getReader();let txt='';const dec=new TextDecoder();
const next=async(pat)=>{while(!pat.test(txt)){const {value}=await rd.read();txt+=dec.decode(value)}};await next(/endpoint\ndata: (.*)\n/);const ep=txt.match(/data: (.*)\n/)[1];
await fetch('http://localhost:8799'+ep,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(msgs[3])});await next(/event: message/);show(JSON.parse(txt.match(/event: message\ndata: (.*)\n/)[1]));
ctl.abort();h.kill();process.exit(0);
