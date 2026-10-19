// Dimension term lists: hand-curated JUT-source seed terms (original, verified against the
// real source text) plus real GoEmotions-derived vocabulary appended below each seed list.
// GoEmotions (google-research-datasets/go_emotions, Apache-2.0, 211,225 real annotated rows,
// downloaded 2026-08-10) -- per-category word-vs-corpus log-odds computed locally, top terms
// per category, categories mapped onto these 5 dimensions by reasoned editorial judgement
// (not itself sourced from the JUT text), then hand-filtered for proper-noun/profanity noise
// and apostrophe-bearing tokens. 'neutral' deliberately excluded -- no category was forced
// into Space, matching the established 'Space does not play a part in these formulae'
// principle. Full method + unfiltered output: see DATASET_PROVENANCE.md.
//
// v1.4.0 addition: each dimension's own real seed-gate vocabulary, pulled directly from
// Bradford Hatcher's "Yijing, Word By Word" (real Key Words + Glossary sections, extracted
// and verified against the actual uploaded 1100-page book -- see yijing_hatcher_dataset.json).
// This is the project's OWN gate assigned to each dimension (Movement=Gate1/Qian,
// Evolution=Gate2/Kun, Being=Gate6/Song, Design=Gate14/Da You, Space=Gate20/Guan) --
// more directly authoritative than the GoEmotions mapping above, since it's not an inferred
// correspondence, it's this project's own already-established seedGate for each dimension.
//
// v1.5.0 addition: Design's seed gate (14) also carries real vocabulary from The Gnostic
// Book of Changes' compiled 'Other titles' list for that hexagram (extracted and verified
// against the actual uploaded 973-page book) -- see gnostic_book_of_changes_dataset.json.
// Only the hexagram name + its synonym-title list were extracted, NOT the book's extensive
// per-line multi-translator quotations (Wilhelm/Baynes, Blofeld, Liu, Ritsema/Karcher,
// Shaughnessy, Cleary, Wu, etc.) -- those are largely other people's still-copyrighted
// translations being compared, not this book's own original material, so they were not
// bulk-extracted. See DATASET_PROVENANCE.md.
//
// v1.6.0 addition: Evolution's seed gate (2) also carries a small, hand-verified vocabulary
// sample from Hua-Ching Ni's 'The Book of Changes and the Unchanging Truth' (the real
// uploaded scan). Unlike Movement/Design, this was NOT scaled to all 64 hexagrams this pass --
// the scan's OCR quality is materially worse (bracket/number corruption made a reliable
// per-hexagram header anchor unachievable in one pass), so only Gate 2 itself was extracted,
// by directly reading the real page range rather than trusting a regex sweep. See
// DATASET_PROVENANCE.md.
const DIMENSIONS = Object.freeze({
  Movement: { interrogative: 'WHERE', operation: 'transition', seedGate: 1, terms: ['movement','energy','creation','seeing','landscape','environment','where','transition','move','wish','desire','desperately','crown','wishes','pray','wanted','wanna','envy','dream','want','wishing','praying','hope','gods','battle','hoping','marry','needed','excited','merrily','exciting','stoked','outstanding','excitement','ecstatic','adventure','yay','interesting','horny','cheers','congratulations','happy','enjoyable','wow','interested','miracle','scariest','terrifying','scary','scared','scares','afraid','terrified','frightening','horrific','horrifying','dreadful','fear','fears','dangerously','scare','horror','horrible','haunting','dangerous','terrible','terror','nightmare','horrid','horrendous','worried','pleasantly','shocks','shocked','surprisingly','surprises','surprised','shockingly','suprised','shocking','surprising','wondered','wondering','amazes','amazed','wonder','surprise','ohhhhh','unbelievable','unexpected','omg','gosh','whoa','shock','potentially','nervous','anxious','shaking','worrying','anxiety','stressed','stressful','edgy','breathing','insecure','depression','panic','worry','depressing','depressed','trouble','infuriating','aggressively','irrationally','dare','angry','frustrated','shut','hate','furious','higher','purpose','self','actualizing','drives','autonomy','calling','vocation','star','quality','sovereignty','command','mastery','dragonhood','genius','authority','cogency','diligence','drive','lasting','energy','enduring','vigor','persistence','duration','time','orders','design','innovation','authoring','infinite','dynamic','life','positing','originality','initiative','sublimation','sunlight','transforming','water','vapor','perspective','from','outside','humanity','attunement','rhythms','purposes','qian','creative','vigorous','energetic','potent','constant','dry','clean','exhausted','heavenly','creation','persis','tence','endurance','order','activity','living','heaven','warmth','sun','appearance','male','gang','yang','principle','create','initiate','author','master','persist','endure','creativity','gan','dried'] },
  Evolution: { interrogative: 'WHAT', operation: 'transform', seedGate: 2, terms: ['evolution','gravity','memory','taste','love','light','what','transform','change','love','loved','loving','loves','adore','unconditional','fav','lovely','favourite','favorite','blanket','favorites','cuddle','hugs','sending','excellent','gorgeous','awesome','cuteness','salute','admirable','artwork','applaud','amazing','beautiful','fantastic','impressive','brilliant','cinematic','soundtrack','admire','cutie','handsome','remarkable','appreciated','great','beautifully','eloquent','thankyou','enlightening','thanks','thank','affirmation','clarification','clarifying','feedback','recommendation','sharing','thankful','input','thankfully','correcting','contribution','kindly','confirming','encouragement','informative','clearing','distinction','thx','greatly','condolences','rip','died','loss','passed','death','dead','peace','die','killed','crying','lost','killing','pain','heart','shame','poor','sorry','family','miss','kill','rest','wife','sad','ago','proud','pride','accomplishment','congrats','congratulations','won','glad','myself','worth','community','truly','strong','deserve','job','dad','super','played','big','country','hopefully','working','single','finally','fan','hopeful','hoping','hope','optimism','luck','someday','optimistic','hoped','finals','confident','hopes','rapid','recovery','cheer','improve','opportunities','consistent','stays','soon','convert','shine','grasp','outlive','careful','bless','sins','concerned','worry','safe','praying','rescue','pray','caring','worries','encourage','stay','concern','protect','succeed','medication','sympathy','wishing','receiving','tolerance','gentleness','patience','openness','accommodation','gratitude','assent','contentment','comprehension','understanding','embrace','room','allowance','endurance','perseverance','acquiescence','compliance','groundedness','support','care','potential','capacity','raw','material','substance','suchness','realism','consent','upholding','simplicity','naturalness','surety','latitude','range','breadth','largesse','fields','options','power','possibility','first','accept','givens','absorbing','learning','growing','accessing','kun','receive','absorb','substantiate','realize','acceptance','matter','field','basis','ground','earth','subordinate','humble','passive','accepting','compliant','female','rou','yin','principle','receptiveness','receptive','motherly','gentle','giving','support','selfless','harmonious','wholeness','nourished','westward','obstacles','achievement','wish','serious'] },
  Being: { interrogative: 'WHEN', operation: 'instantiate', seedGate: 6, terms: ['being','matter','touch','sex','survival','when','instantiate','runtime','execute','enjoyed','joy','constructive','happily','happiest','enjoying','happy','enjoy','glad','cheered','merrily','enjoyable','gladly','cheers','happier','bliss','celebrate','anniversary','enjoys','musical','happiness','fun','saddest','sad','heartbreaking','sadly','painfully','heartbroken','painful','sadness','disheartening','hurts','pains','cried','stressful','crying','cries','condolences','ouch','lonely','cry','devastating','loss','hurting','severe','shaking','disgusting','disgusted','disgust','yuck','hideous','horrendous','awful','nasty','repulsive','smells','worst','worse','filthy','creepy','vomit','dirty','gross','tastes','obnoxious','weirdo','weirdest','greasy','weird','embarrassing','embarassing','embarrassment','embarrassed','awkward','shameful','ashamed','cringy','shame','shamed','cringey','uncomfortable','insulting','oddly','accidentally','pants','apologies','insult','weight','kills','cringe','hahahahahaha','funniest','hahah','hahahaha','hilarious','funnier','hahahah','laughed','hahaha','lol','giggled','hahahahaha','haha','lmaoooo','funny','laughing','hehe','hilariously','laugh','amusing','lmao','ahaha','diss','relieved','relief','relax','thankfully','solved','goodness','finally','luckily','worries','oof','peace','worry','cool','safe','god','saved','helps','helpful','least','lucky','anxiety','sleep','wasn','easier','advocacy','adversarialism','partiality','partisanship','taking','promoting','one','side','presumption','challenge','competition','ambivalence','approach','conflicts','conflict','disparity','dissent','dissonance','points','view','within','bigger','picture','resistance','friction','strife','grievance','litigation','dispute','contest','confrontation','arbitration','diplomacy','conciliation','reconsideration','mid','course','corrections','revisiting','postulates','reference','frames','using','feedback','finding','metasolutions','song','contention','argu','ment','quarrel','antagonism','contend','dis','sent','litigate','accuse','argue','reprimand','advocate','speak','publicly','bring','suit','complaint','demand','justice','plead','before','court','quarrelsome','adversarial'] },
  Design: { interrogative: 'WHY', operation: 'structure', seedGate: 14, terms: ['design','structure','progress','smell','life','art','why','plan','code','build','agreed','agree','agreeing','understandable','fitting','approve','sensible','equal','visited','correct','factor','conscious','yep','toward','disagree','unpopular','unacceptable','widow','refuse','disagrees','dull','sham','nope','incorrect','noooo','gear','teenage','misogyny','outright','violating','irrelevant','denial','whatsoever','objectively','seeking','inherently','criticism','dialogue','disappointing','disappoints','disappointment','disappointed','upsetting','depressing','upset','failing','disaster','unlucky','doomed','unfortunately','wasted','unhappy','boring','prequel','depressed','miss','ouch','missed','dammit','irritated','annoying','annoyed','irritating','frustrating','annoys','annoy','accuse','reckless','intolerance','childish','idiots','idiotic','managers','idiot','fools','sucked','nonsense','rubbish','confused','confusing','puzzled','confuses','unsure','confusion','doubtful','clueless','clue','doubt','understood','inch','nor','bridge','whether','visiting','understand','vague','curious','curiosity','withdrawals','intrigued','communities','elaborate','incident','describing','oooh','terminal','wondered','wondering','interests','familiar','attempts','proving','odds','clone','explain','arrest','realization','realised','realized','realizing','realise','realize','realizes','reminds','typed','bullies','discovered','previously','spoke','struggles','recognized','figured','protest','aware','learned','noticed','ourselves','shocking','stance','sorry','apologies','regret','apologize','apology','repent','guilt','misunderstanding','sincerely','guilty','regrets','losses','forgive','loss','whoops','mistake','insulting','meant','ignorance','sold','error','english','divorce','possession','greatness','wealth','endowments','enrichment','abundance','affluence','assets','dominion','domain','home','belonging','here','tenure','laying','claims','vantage','command','territory','spheres','influence','enterprise','venture','credit','value','interest','appreciation','treasuring','worth','gratitude','counting','blessings','experience','owning','one','power','assign','rearrange','revise','values','prometheus','foresight','gave','fire','from','heaven','man','entitlement','appropriation','accomplished','best','better','big','complete','critical','crucial','developed','enormous','entire','extreme','far','full','fully','grown','good','grand','great','est','hea','high','large','greatly','important','long','loud','major','mature','mighty','noble','noteworthy','old','overall','perfect','pure','realized','ripe','seasoned','serious','significant','strong','successful','vast','whole','wholesome','vital','lot','lots','master','completeness','development','entirety','eur','ness','growth','importance','largesse','jority','maturity','vastness','wholeness','deal','much','already','completely','entirely','thoroughly','wholly','masse','well','you','being','there','they','had','will','come','something','exist','remain','stay','become','arise','appear','grow','attain','learn','turn','would','could','may','might','can','get','having','their','contain','hold','keep','maintain','retain','possess','presume','assume','remember','take','granted','bring','about','achieve','acquire','capture','claim','earn','find','gain','seize','took','present','possessed','contained','held','abundant','rich','plentiful','attainment','existence','presence','what','exists','question','anybody','somebody','someone','given','especially','sovereign','ous','wealth','possession','great','measure','symbol','sovereignty','having','possessing','possessor','abundance'] },
  Space: { interrogative: 'WHO', operation: 'integrate', seedGate: 20, terms: ['space','form','illusion','hearing','music','freedom','who','integrate','connect','compose','observe','view','attend','study','contemplate','consider','examine','examined','life','investigation','reconnaissance','review','survey','learning','comprehension','compass','changing','points','frames','mind','postulates','hypotheses','outlooks','ken','reference','reframing','universes','discourse','suspending','dis','belief','overview','taking','inventory','overall','objectivity','ification','reading','signs','understanding','other','perspectives','point','being','beheld','objectified','guan','behold','comprehend','divine','evaluate','gaze','upon','concentration','look','per','ceive','see','watch','scry','regard','observation','perception','perspective','prospect','scenery','sight','vista'] },
});

export const TOOL_LEVELS = Object.freeze([
  { level: 0, name: 'Base', kind: 'constant' },
  { level: 1, name: 'Tone', kind: 'classifier' },
  { level: 2, name: 'Color', kind: 'filter' },
  { level: 3, name: 'Bigram', kind: 'matcher' },
  { level: 4, name: 'Trigram', kind: 'analyzer' },
  { level: 5, name: 'Hexagram', kind: 'rule' },
  { level: 6, name: 'Channel', kind: 'connector' },
  { level: 7, name: 'Circuit', kind: 'system' },
]);

export class FactoryError extends Error {
  constructor(code, message, details = {}) { super(message); this.name = 'FactoryError'; this.code = code; this.details = details; }
}

const words = value => String(value ?? '').toLowerCase().match(/[a-z0-9']+/g) || [];
const clone = value => globalThis.structuredClone ? structuredClone(value) : JSON.parse(JSON.stringify(value));
const canonical = value => value === null || typeof value !== 'object' ? JSON.stringify(value)
  : Array.isArray(value) ? `[${value.map(canonical).join(',')}]`
  : `{${Object.keys(value).sort().map(k => `${JSON.stringify(k)}:${canonical(value[k])}`).join(',')}}`;

export function stableId(value) {
  const text = canonical(value); let a = 0xdeadbeef ^ text.length, b = 0x41c6ce57 ^ text.length;
  for (let i=0;i<text.length;i++) { a=Math.imul(a^text.charCodeAt(i),2654435761); b=Math.imul(b^text.charCodeAt(i),1597334677); }
  a=Math.imul(a^(a>>>16),2246822507)^Math.imul(b^(b>>>13),3266489909);
  b=Math.imul(b^(b>>>16),2246822507)^Math.imul(a^(a>>>13),3266489909);
  return `${(b>>>0).toString(16).padStart(8,'0')}${(a>>>0).toString(16).padStart(8,'0')}`;
}

function seeded(seed) { let x=parseInt(stableId(seed).slice(0,8),16)||1; return () => ((x=Math.imul(x^x>>>15,1|x),x^=x+Math.imul(x^x>>>7,61|x),(x^x>>>14)>>>0)/4294967296); }
function gateBits(gate) { return ((gate-1)&63).toString(2).padStart(6,'0'); }
function xnorGate(a,b) { return ((~((a-1)^(b-1)))&63)+1; }
function port(id,direction,type='json',guarantees=[],requires=[]) { return Object.freeze({id,direction,type,schemaVersion:'1',guarantees:Object.freeze([...guarantees]),requires:Object.freeze([...requires])}); }

export class DimensionRouter {
  constructor(seed = DIMENSIONS) { this.spaces=new Map(Object.entries(seed).map(([name,data])=>[name,{...data,counts:new Map(data.terms.map(t=>[t,1]))}])); }
  ingest(dimension,text) { const space=this.spaces.get(dimension); if(!space) throw new FactoryError('UNKNOWN_DIMENSION',dimension); for(const term of words(text)) space.counts.set(term,(space.counts.get(term)||0)+1); return this; }
  route(text) {
    const tokens=words(text); const ranked=[...this.spaces].map(([dimension,s])=>({dimension,score:tokens.reduce((n,t)=>n+(s.counts.get(t)||0),0),seedGate:s.seedGate,interrogative:s.interrogative,operation:s.operation})).sort((a,b)=>b.score-a.score||a.dimension.localeCompare(b.dimension));
    if(!ranked[0]?.score) return Object.freeze({status:'unresolved',reason:'NO_DIMENSION_MATCH',ranked:Object.freeze(ranked)});
    const top=ranked.filter(x=>x.score===ranked[0].score); if(top.length>1) return Object.freeze({status:'ambiguous',reason:'DIMENSION_TIE',candidates:Object.freeze(top)});
    return Object.freeze({status:'resolved',...ranked[0],tokens:Object.freeze(tokens)});
  }
}

const LEVEL_CUES = Object.freeze([
  ['constant','seed','value','base'], ['classify','boolean','tone','yin','yang'], ['filter','select','color','predicate'],
  ['match','compare','pair','bigram'], ['analyze','learn','trigram'], ['rule','grammar','hexagram'],
  ['connect','transmit','channel','event'], ['system','circuit','workflow','pipeline','orchestrate'],
]);

export class PurposePlanner {
  plan(request, route) {
    const explicit=request.level; let level;
    if(Number.isInteger(explicit)&&explicit>=0&&explicit<=7) level=explicit;
    else { const tokens=new Set(words(`${request.purpose||''} ${request.input||''}`)); const scores=LEVEL_CUES.map(c=>c.reduce((n,x)=>n+(tokens.has(x)?1:0),0)); const max=Math.max(...scores); level=max ? scores.lastIndexOf(max) : ({transition:3,transform:4,instantiate:5,structure:5,integrate:7}[route.operation]??5); }
    const gate=Number.isInteger(request.gate)?request.gate:route.seedGate;
    if(gate<1||gate>64) throw new FactoryError('INVALID_GATE','gate must be 1..64');
    const targetGate=Number.isInteger(request.targetGate)?request.targetGate:(((gate+level*7-1)%64)+1);
    const line=Number.isInteger(request.line)?request.line:((level%6)+1);
    return Object.freeze({level,gate,targetGate,line,color:((line-1)%6)+1,tone:((line-1)%6)+1,base:((line-1)%5)+1,dimension:route.dimension,operation:route.operation,purpose:request.purpose||null});
  }
}

function makeRuntime(plan, id) {
  const random=seeded(id), state={seen:new Map(),rules:[],events:[]};
  const address=`${plan.dimension}:G${plan.gate}.L${plan.line}.C${plan.color}.T${plan.tone}.B${plan.base}`;
  const implementations=[
    async input=>({address,value:plan.base,input}),
    async input=>{const value=Number(input); if(!Number.isFinite(value)) throw new FactoryError('EXPECTED_NUMBER','Tone requires a number'); const attractor=gateBits(plan.gate).endsWith('1')?.75:.25; return {value,isYang:value>.5,isYin:value<=.5,tension:Math.abs(value-attractor)};},
    async input=>{const list=Array.isArray(input)?input:[input]; return list.filter(item=>item?.color===plan.color||item?.tone===plan.tone||item?.gate===plan.gate);},
    async input=>{const [a,b]=Array.isArray(input)?input:[input,input]; const aa=words(a),bb=words(b),both=[...new Set(aa.filter(x=>bb.includes(x)))]; return {match:both.length/Math.max(1,new Set([...aa,...bb]).size),shared:both,pattern:[plan.gate&3,(plan.gate>>2)&3]};},
    async input=>{const key=canonical(input),count=(state.seen.get(key)||0)+1; state.seen.set(key,count); return {input,occurrences:count,features:{gate:plan.gate,line:plan.line,color:plan.color,tone:plan.tone,base:plan.base}};},
    async input=>{const record={lhs:plan.dimension,rhs:input,weight:1+state.rules.length/10}; state.rules.push(record); return {rule:record,address};},
    async input=>{const packet={sequence:state.events.length+1,from:plan.gate,to:plan.targetGate,signal:input,address}; state.events.push(packet); return packet;},
    async input=>{const tokens=words(input),transitions=[]; for(let i=0;i<tokens.length-1;i++) transitions.push({from:tokens[i],to:tokens[i+1]}); const walk=[]; if(tokens.length){let cur=tokens[0]; walk.push(cur); for(let i=0;i<15;i++){const options=transitions.filter(t=>t.from===cur); if(!options.length)break; cur=options[Math.floor(random()*options.length)].to; walk.push(cur);}} return {output:walk.join(' '),nodes:[...new Set(tokens)],edges:transitions,address};},
  ];
  return {execute:implementations[plan.level],exportState:()=>({seen:[...state.seen],rules:clone(state.rules),events:clone(state.events)})};
}

function generatedSource(plan,id,name,state={seen:[],rules:[],events:[]}) {
  const address=`${plan.dimension}:G${plan.gate}.L${plan.line}.C${plan.color}.T${plan.tone}.B${plan.base}`;
  const manifest={manifestVersion:'ato.standalone-tool.v1',id,name,address:{mode:'macro',gate:plan.gate,line:plan.line,color:plan.color,tone:plan.tone,base:plan.base},addressKey:address,structure:{level:plan.level,name:TOOL_LEVELS[plan.level].name},functionalLevel:plan.dimension.toLowerCase(),metadata:{family:'standalone-generated-tool',purpose:plan.purpose,decomposable:true,deterministic:true,zeroImport:true}};
  return `// Cynthia standalone Tool-Automaton — no factory, mesh, backend, or imports required.\nexport const manifest = Object.freeze(${JSON.stringify(manifest,null,2)});\nconst plan=Object.freeze(${JSON.stringify(plan)});\nconst address=${JSON.stringify(address)};\nconst state={seen:new Map(${JSON.stringify(state.seen||[])}),rules:${JSON.stringify(state.rules||[])},events:${JSON.stringify(state.events||[])}};\nconst words=value=>String(value??'').toLowerCase().match(/[a-z0-9']+/g)||[];\nconst canonical=value=>value===null||typeof value!=='object'?JSON.stringify(value):Array.isArray(value)?'['+value.map(canonical).join(',')+']':'{'+Object.keys(value).sort().map(k=>JSON.stringify(k)+':'+canonical(value[k])).join(',')+'}';\nfunction hash(value){const text=canonical(value);let a=0xdeadbeef^text.length,b=0x41c6ce57^text.length;for(let i=0;i<text.length;i++){a=Math.imul(a^text.charCodeAt(i),2654435761);b=Math.imul(b^text.charCodeAt(i),1597334677);}a=Math.imul(a^(a>>>16),2246822507)^Math.imul(b^(b>>>13),3266489909);b=Math.imul(b^(b>>>16),2246822507)^Math.imul(a^(a>>>13),3266489909);return ((b>>>0).toString(16).padStart(8,'0'))+((a>>>0).toString(16).padStart(8,'0'));}\nfunction seeded(seed){let x=parseInt(hash(seed).slice(0,8),16)||1;return()=>((x=Math.imul(x^x>>>15,1|x),x^=x+Math.imul(x^x>>>7,61|x),(x^x>>>14)>>>0)/4294967296);}\nconst random=seeded(manifest.id);\nconst runners=[\n async input=>({address,value:plan.base,input}),\n async input=>{const value=Number(input);if(!Number.isFinite(value))throw new TypeError('Tone requires a number');const attractor=((plan.gate-1)&1)?.75:.25;return{value,isYang:value>.5,isYin:value<=.5,tension:Math.abs(value-attractor)};},\n async input=>(Array.isArray(input)?input:[input]).filter(item=>item?.color===plan.color||item?.tone===plan.tone||item?.gate===plan.gate),\n async input=>{const[a,b]=Array.isArray(input)?input:[input,input],aa=words(a),bb=words(b),shared=[...new Set(aa.filter(x=>bb.includes(x)))];return{match:shared.length/Math.max(1,new Set([...aa,...bb]).size),shared,pattern:[plan.gate&3,(plan.gate>>2)&3]};},\n async input=>{const key=canonical(input),occurrences=(state.seen.get(key)||0)+1;state.seen.set(key,occurrences);return{input,occurrences,features:{gate:plan.gate,line:plan.line,color:plan.color,tone:plan.tone,base:plan.base}};},\n async input=>{const rule={lhs:plan.dimension,rhs:input,weight:1+state.rules.length/10};state.rules.push(rule);return{rule,address};},\n async input=>{const packet={sequence:state.events.length+1,from:plan.gate,to:plan.targetGate,signal:input,address};state.events.push(packet);return packet;},\n async input=>{const tokens=words(input),edges=[];for(let i=0;i<tokens.length-1;i++)edges.push({from:tokens[i],to:tokens[i+1]});const walk=[];if(tokens.length){let cur=tokens[0];walk.push(cur);for(let i=0;i<15;i++){const options=edges.filter(t=>t.from===cur);if(!options.length)break;cur=options[Math.floor(random()*options.length)].to;walk.push(cur);}}return{output:walk.join(' '),nodes:[...new Set(tokens)],edges,address};}\n];\nexport async function execute(input,context={}){return runners[plan.level](input,context);}\nexport const call=execute;\nexport function exportState(){return{seen:[...state.seen],rules:structuredClone(state.rules),events:structuredClone(state.events)};}\nexport default Object.freeze({manifest,execute,call,exportState});\n`;
}

export class GeneratedTool {
  constructor({id,name,plan,request,parents=[]}) { this.id=id; this.name=name; this.plan=plan; this.request=Object.freeze(clone(request)); this.parents=Object.freeze([...parents]); this.calls=0; this.lifecycle='ready'; this.createdAtSequence=0; const runtime=makeRuntime(plan,id); this._execute=runtime.execute; this._exportState=runtime.exportState; this.ports=Object.freeze([port('in','input','json'),port('out','output','json',['addressed','generated'])]); }
  get address(){return {mode:'macro',gate:this.plan.gate,line:this.plan.line,color:this.plan.color,tone:this.plan.tone,base:this.plan.base};}
  get addressKey(){return `${this.plan.dimension}:G${this.plan.gate}.L${this.plan.line}.C${this.plan.color}.T${this.plan.tone}.B${this.plan.base}`;}
  async call(input,context={}) { this.lifecycle='active'; this.calls++; try { const output=await this._execute(input,context); this.lifecycle='ready'; return output; } catch(error){this.lifecycle='error';throw error;} }
  async execute(input,context={}){return this.call(input,context);}
  exportModule({ includeState=true }={}) { return generatedSource(this.plan,this.id,this.name,includeState?this._exportState():undefined); }
  get source(){return this.exportModule();}
  manifest(){return Object.freeze({manifestVersion:'ato.generated-tool.v2',id:this.id,name:this.name,address:this.address,addressKey:this.addressKey,structure:{level:this.plan.level,name:TOOL_LEVELS[this.plan.level].name},functionalLevel:this.plan.dimension.toLowerCase(),ports:this.ports,metadata:Object.freeze({family:'generated-tool',purpose:this.plan.purpose,parents:this.parents,decomposable:true,deterministic:true}),lifecycle:this.lifecycle,calls:this.calls});}
  exportRecord(){return Object.freeze({manifest:this.manifest(),request:this.request,plan:this.plan,source:this.exportModule(),state:this._exportState()});}
}

export class IntegratedToolFactory {
  constructor({router=new DimensionRouter(),planner=new PurposePlanner()}={}) { this.router=router; this.planner=planner; this.tools=new Map(); this.revisions=[]; }
  generate(request={}) { if(!request.purpose&&!request.input) throw new FactoryError('MISSING_PURPOSE','purpose or input is required'); const route=request.dimension?Object.freeze({status:'resolved',dimension:request.dimension,...DIMENSIONS[request.dimension]}):this.router.route(`${request.purpose||''} ${request.input||''}`); if(route.status!=='resolved') return route; if(!DIMENSIONS[route.dimension]) throw new FactoryError('UNKNOWN_DIMENSION',route.dimension); const normalizedRoute={...DIMENSIONS[route.dimension],...route}; const plan=this.planner.plan(request,normalizedRoute); const identity={version:'integrated-tool-factory.v1',plan,request:{purpose:request.purpose||null,input:request.input||null},parents:request.parents||[]}; const id=`tool-${stableId(identity)}`; if(this.tools.has(id)) return Object.freeze({status:'existing',tool:this.tools.get(id)}); const name=`${TOOL_LEVELS[plan.level].name}${plan.dimension}G${plan.gate}L${plan.line}`; const tool=new GeneratedTool({id,name,plan,request,parents:request.parents}); this.tools.set(id,tool); return Object.freeze({status:'generated',tool}); }
  mount(tool,mesh) { if(!tool?.manifest||typeof tool.call!=='function') throw new FactoryError('INVALID_TOOL','mount requires a generated tool'); if(!mesh||typeof mesh.add!=='function') throw new FactoryError('INVALID_MESH','mesh.add is required'); try { mesh.add(tool); } catch(error) { if(error?.code==='INVALID_AUTOMATON'&&mesh.automatons instanceof Map){if(mesh.automatons.has(tool.id))throw new FactoryError('DUPLICATE_TOOL',tool.id);mesh.automatons.set(tool.id,tool);} else throw error; } return tool.manifest(); }
  dissolve(id,reason='dissolved'){const tool=this.tools.get(id);if(!tool)return false;tool.lifecycle='dissolved';this.revisions.push({type:'dissolve',id,reason});return true;}
  restore(id){const tool=this.tools.get(id);if(!tool)throw new FactoryError('TOOL_NOT_FOUND',id);tool.lifecycle='ready';this.revisions.push({type:'restore',id});return tool;}
  exportTool(id,options={}){const tool=this.tools.get(id);if(!tool)throw new FactoryError('TOOL_NOT_FOUND',id);return Object.freeze({fileName:`${tool.name}-${tool.id}.mjs`,source:tool.exportModule(options),manifest:tool.manifest()});}
  snapshot(){return Object.freeze({manifestVersion:'integrated-tool-factory.v1',tools:Object.freeze([...this.tools.values()].map(t=>t.exportRecord())),revisions:Object.freeze(clone(this.revisions))});}
}

export { DIMENSIONS };
export { ATONativeBridge } from './ato-native-bridge.mjs';
