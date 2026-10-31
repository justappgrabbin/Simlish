import * as THREE from './vendor/three/three.module.js';
import {BEACONS,startRun,offerChoices,activateBeacon,rechargeAtCamp,restoreRun} from './adventure-core.mjs';

const $=selector=>document.querySelector(selector);
const STORE='you-n-i-verse-five-lights-v1';
const stage=$('#stage');
let run,hadSave=false,renderer,scene,camera,player,cloak,arms=[],lighting=[],glows=[],starfield;
let lastFrame=0,travel=null,pending=null,choiceBusy=false,gameStarted=false,resetArmed=false;
const keys=new Set();
const interactive=[];
const beacons=new Map();

function status(message){$('#toast').textContent=message;}
function overlay(id,visible){$('#'+id).hidden=!visible;}
function anyOverlay(){return !$('#intro').hidden||!$('#decision').hidden||!$('#ending').hidden||!$('#renderError').hidden;}
function save(){
  try{localStorage.setItem(STORE,JSON.stringify(run));return true;}
  catch{status('Your device could not save progress. Keep this session open.');return false;}
}
function load(){
  try{
    const raw=localStorage.getItem(STORE);
    if(raw){run=restoreRun(JSON.parse(raw));hadSave=true;return;}
  }catch{status('Saved game could not be restored; a fresh journey is available.');}
  run=startRun();
}
function newGame(){
  run=startRun();travel=null;pending=null;gameStarted=true;resetArmed=false;$('#reset').textContent='New game';
  player.position.set(0,0,0);lighting.forEach(({beacon,orb,ring})=>{
    orb.material.emissiveIntensity=.65;ring.material.opacity=.58;beacon.userData.lit=false;
  });
  save();updateHud();
  for(const name of ['intro','decision','ending'])overlay(name,false);
  status('Your journey begins. Tap any light to walk toward it.');
}
function updateHud(){
  $('#lights').textContent=run.lit.length+' / 5';
  $('#focus').textContent=run.focus+' / 9';
  $('#score').textContent=String(run.score);
  $('#gate').textContent='GATE '+run.path.gate;
  document.querySelectorAll('#progress span').forEach((part,index)=>part.classList.toggle('lit',index<run.lit.length));
  const list=$('#places');list.replaceChildren();
  BEACONS.forEach((beacon,index)=>{
    const done=run.lit.includes(beacon.id);
    const button=document.createElement('button');button.className='place'+(done?' done':'');
    const color='#'+beacon.color.toString(16).padStart(6,'0');button.style.setProperty('--color',color);
    const glyph=document.createElement('span');glyph.className='glyph';glyph.textContent=done?'✦':'◇';
    const title=document.createElement('span');const strong=document.createElement('strong');strong.textContent=beacon.name;
    const small=document.createElement('small');small.textContent=done?'Light restored':beacon.title;
    title.append(strong,small);
    const action=document.createElement('span');action.className='go';action.textContent=done?'✓':'GO ↗';
    button.append(glyph,title,action);
    button.onclick=()=>goTo(beacon.id);
    list.append(button);
  });
  const journal=$('#chronicle');journal.replaceChildren();
  if(!run.journal.length){journal.textContent='Your first choice is ahead.';}
  else for(const event of run.journal.slice(-5).reverse()){
    const p=document.createElement('div');p.className='log-entry';
    p.textContent=event.type==='rest'?'At camp: +'+(event.focusAfter-event.focusBefore)+' focus, −'+event.scoreCost+' points':
      event.dimension+': '+event.choice+' · '+event.fromGate+' → '+event.toGate;
    journal.append(p);
  }
  lighting.forEach(({beacon,orb,ring},index)=>{
    const done=run.lit.includes(BEACONS[index].id);beacon.userData.lit=done;
    orb.material.emissiveIntensity=done?2.5:.65;ring.material.opacity=done?1:.58;
  });
  $('#stageHint').textContent=run.lit.length===5?'The residence is glowing':(5-run.lit.length)+' lights remaining';
}
function speakNote(frequency,duration,delay=0){
  try{
    const AudioContext=window.AudioContext||window.webkitAudioContext;
    if(!AudioContext)return;
    if(!speakNote.context)speakNote.context=new AudioContext();
    const context=speakNote.context,osc=context.createOscillator(),gain=context.createGain(),time=context.currentTime+delay;
    osc.type='sine';osc.frequency.setValueAtTime(frequency,time);
    gain.gain.setValueAtTime(.0001,time);gain.gain.exponentialRampToValueAtTime(.07,time+.025);
    gain.gain.exponentialRampToValueAtTime(.0001,time+duration);
    osc.connect(gain);gain.connect(context.destination);osc.start(time);osc.stop(time+duration+.03);
  }catch{}
}
function sound(){speakNote(440,.18);speakNote(587,.21,.09);speakNote(880,.35,.21);}
function glowingTexture(){
  const c=document.createElement('canvas');c.width=c.height=128;
  const g=c.getContext('2d'),r=g.createRadialGradient(64,64,2,64,64,63);
  r.addColorStop(0,'#ffffffdd');r.addColorStop(.2,'#ffffff88');r.addColorStop(1,'#ffffff00');
  g.fillStyle=r;g.fillRect(0,0,128,128);return new THREE.CanvasTexture(c);
}
function labelTexture(word){
  const c=document.createElement('canvas');c.width=512;c.height=96;
  const g=c.getContext('2d');g.fillStyle='#170f2ad9';g.beginPath();g.roundRect(16,10,480,75,28);g.fill();
  g.strokeStyle='#ebc9b18c';g.lineWidth=3;g.stroke();g.fillStyle='#fff0dd';
  g.textAlign='center';g.font='bold 37px system-ui';g.fillText(word.toUpperCase(),256,61);
  const tex=new THREE.CanvasTexture(c);tex.colorSpace=THREE.SRGBColorSpace;return tex;
}
const material=(color,emissive=0,emissiveIntensity=0)=>new THREE.MeshStandardMaterial({color,roughness:.55,metalness:.16,emissive,emissiveIntensity});
const mesh=(geometry,mat,x=0,y=0,z=0)=>{
  const obj=new THREE.Mesh(geometry,mat);obj.position.set(x,y,z);return obj;
};
function addRock(parent,x,z,size,color){
  const o=mesh(new THREE.DodecahedronGeometry(size,0),material(color),x,size*.6,z);
  o.rotation.y=x+z;parent.add(o);
}
function createCharacter(){
  const root=new THREE.Group();
  const skin=material(0xc58e82),robe=material(0x753d8f),trim=material(0xf3b76e),hair=material(0x21172c);
  const boots=material(0x26172e);
  const disk=mesh(new THREE.CircleGeometry(.75,24),new THREE.MeshBasicMaterial({color:0x05030d,transparent:true,opacity:.34,depthWrite:false}),0,.04,0);disk.rotation.x=-Math.PI/2;root.add(disk);
  const legs=[];
  for(const x of [-.24,.24]){
    const leg=new THREE.Group();leg.position.set(x,.94,0);
    leg.add(mesh(new THREE.CylinderGeometry(.18,.23,.78,9),boots,0,-.39,0));
    leg.add(mesh(new THREE.SphereGeometry(.23,10,8),boots,0,-.76,.1));
    root.add(leg);legs.push(leg);
  }
  const cape=mesh(new THREE.ConeGeometry(.95,1.95,10),robe,0,1.34,-.12);
  cape.rotation.z=.04;root.add(cape);
  root.add(mesh(new THREE.CylinderGeometry(.48,.72,.95,10),material(0x633a74),0,1.66,0));
  root.add(mesh(new THREE.TorusGeometry(.43,.1,9,20),trim,0,2.1,0).rotateX(Math.PI/2));
  for(const x of [-.63,.63]){
    const arm=new THREE.Group();arm.position.set(x,2.05,0);
    arm.add(mesh(new THREE.CylinderGeometry(.18,.14,.9,8),robe,0,-.39,0));
    arm.add(mesh(new THREE.SphereGeometry(.18,10,8),skin,0,-.88,0));
    root.add(arm);arms.push(arm);
  }
  root.add(mesh(new THREE.SphereGeometry(.4,18,14),skin,0,2.6,.08));
  root.add(mesh(new THREE.SphereGeometry(.43,15,10),hair,0,2.79,-.09));
  root.add(mesh(new THREE.TorusGeometry(.39,.095,9,24),trim,0,2.77,.04).rotateX(Math.PI/2));
  const eyeMat=new THREE.MeshBasicMaterial({color:0x2b193c});
  for(const x of [-.14,.14])root.add(mesh(new THREE.SphereGeometry(.038,8,8),eyeMat,x,2.61,.438));
  const gem=mesh(new THREE.OctahedronGeometry(.17),material(0x71edd7,0x4aeace,1.7),0,2.03,.58);root.add(gem);
  root.userData.legs=legs;root.userData.cape=cape;
  scene.add(root);return root;
}
function addZoneDecor(group,b,index){
  // Small handcrafted 3D landmarks make the locations visually distinct.
  if(index===1){
    for(const [dx,dz] of [[-2,.9],[1.8,.8],[.4,-2]]){
      group.add(mesh(new THREE.CylinderGeometry(.15,.22,1.5,7),material(0x664c4b),dx,.8,dz));
      group.add(mesh(new THREE.IcosahedronGeometry(.85,0),material(0x338a76,0x164e3b,.35),dx,1.85,dz));
    }
  }else if(index===2){
    const pool=mesh(new THREE.CylinderGeometry(2,2,.06,36),new THREE.MeshStandardMaterial({color:0x1d89b0,metalness:.5,roughness:.25,transparent:true,opacity:.8}),0,.07,0);group.add(pool);
    for(let i=0;i<3;i++){const ring=mesh(new THREE.TorusGeometry(.65+i*.4,.022,6,42),new THREE.MeshBasicMaterial({color:0x8ee9f7,transparent:true,opacity:.55}),0,.115,0);ring.rotation.x=-Math.PI/2;group.add(ring);}
  }else if(index===3){
    for(const [x,z,s] of [[-1.5,1,1],[1.6,-1,.7],[-1,-1.4,.6]]){
      const crystal=mesh(new THREE.OctahedronGeometry(s),material(0xab73d9,0x562a89,.45),x,s,z);crystal.rotation.z=.22;group.add(crystal);
    }
  }else if(index===4){
    const sphere=mesh(new THREE.SphereGeometry(1.35,16,12),new THREE.MeshBasicMaterial({color:0xf8d88d,wireframe:true,transparent:true,opacity:.28}),0,2.15,0);group.add(sphere);
    const rings=mesh(new THREE.TorusGeometry(1.7,.05,8,48),material(0xe5ad79),0,2.1,0);rings.rotation.x=.5;group.add(rings);
  }else{
    for(let i=0;i<3;i++){
      const x=Math.cos(i*2.1)*1.75,z=Math.sin(i*2.1)*1.75;
      const arch=mesh(new THREE.TorusGeometry(.62,.09,8,28,Math.PI),material(0xd8a16c),x,1.05,z);
      arch.rotation.z=Math.PI*.04;group.add(arch);
    }
  }
}
function createBeacon(b,index,glowTex){
  const group=new THREE.Group();group.position.set(b.x,0,b.z);
  const baseColor=new THREE.Color(b.color);
  const base=mesh(new THREE.CylinderGeometry(1.38,1.68,.27,18),material(0x372443),0,.1,0);
  const pedestal=mesh(new THREE.CylinderGeometry(.42,.65,1.05,9),material(0x795f87),0,.7,0);
  const orbMat=material(b.color,b.color,.65);
  const orb=mesh(new THREE.IcosahedronGeometry(.59,1),orbMat,0,1.65,0);
  const ring=mesh(new THREE.TorusGeometry(.88,.065,9,42),new THREE.MeshBasicMaterial({color:baseColor,transparent:true,opacity:.58}),0,1.65,0);
  ring.rotation.x=.5;ring.rotation.y=.22;
  const light=new THREE.PointLight(b.color,1.65,5.5);light.position.y=1.75;
  const glow=new THREE.Sprite(new THREE.SpriteMaterial({map:glowTex,color:b.color,transparent:true,opacity:.75,depthWrite:false}));
  glow.scale.set(4.5,4.5,1);glow.position.y=1.7;
  const name=new THREE.Sprite(new THREE.SpriteMaterial({map:labelTexture(b.name),transparent:true,depthWrite:false}));
  name.scale.set(3.6,.675,1);name.position.y=3.15;
  group.add(base,pedestal,glow,orb,ring,light,name);
  addZoneDecor(group,b,index);
  for(let i=0;i<5;i++){const theta=i*Math.PI*2/5;addRock(group,Math.cos(theta)*2.55,Math.sin(theta)*2.55,.2+(i%3)*.07,0x4e3b63);}
  group.traverse(child=>{if(child.isMesh||child.isSprite){child.userData.destination=b.id;interactive.push(child)}});
  scene.add(group);lighting.push({beacon:group,orb,ring,light,glow});beacons.set(b.id,group);
}
function makeWorld(){
  scene=new THREE.Scene();scene.background=new THREE.Color(0x0d0b1a);scene.fog=new THREE.FogExp2(0x100d20,.014);
  scene.add(new THREE.HemisphereLight(0xbaacd8,0x25142a,2));
  const moon=new THREE.DirectionalLight(0xffd2aa,2.25);moon.position.set(-9,20,14);scene.add(moon);
  const floor=mesh(new THREE.CylinderGeometry(11.3,11.5,.65,80),material(0x231c39),0,-.4,0);scene.add(floor);
  const trim=mesh(new THREE.TorusGeometry(10.65,.08,9,90),new THREE.MeshBasicMaterial({color:0xb770a9}),0,-.04,0);
  trim.rotation.x=Math.PI/2;scene.add(trim);
  for(const radius of [3.8,6.4,9.7]){
    const circle=mesh(new THREE.TorusGeometry(radius,.022,5,90),new THREE.MeshBasicMaterial({color:0x9e659a,transparent:true,opacity:.45}),0,.003,0);circle.rotation.x=Math.PI/2;scene.add(circle);
  }
  const glowTex=glowingTexture();
  for(const [index,b] of BEACONS.entries()){
    const length=Math.hypot(b.x,b.z);
    const trail=mesh(new THREE.BoxGeometry(.48,.04,length),new THREE.MeshBasicMaterial({color:0x5b476b,transparent:true,opacity:.62}),b.x/2,.018,b.z/2);
    trail.rotation.y=Math.atan2(b.x,b.z);scene.add(trail);
    for(let i=0;i<4;i++){const factor=(i+1)/5;addRock(scene,b.x*factor+.65,b.z*factor,.15,0x493a5b);}
    createBeacon(b,index,glowTex);
  }
  const camp=new THREE.Group();camp.position.set(0,0,0);
  camp.add(mesh(new THREE.CylinderGeometry(1.1,1.2,.25,28),material(0x544368),0,.11,0));
  const flame=mesh(new THREE.OctahedronGeometry(.48),material(0xfbc79f,0xf7929d,1.9),0,1,0);
  camp.add(mesh(new THREE.CylinderGeometry(.15,.25,.65,9),material(0x94738c),0,.5,0),flame);
  const campGlow=new THREE.PointLight(0xffba9a,1.8,7);campGlow.position.y=1.2;camp.add(campGlow);
  const campBill=new THREE.Sprite(new THREE.SpriteMaterial({map:labelTexture('CAMP'),transparent:true,depthWrite:false}));
  campBill.scale.set(2.7,.5,1);campBill.position.y=2.1;camp.add(campBill);
  camp.traverse(child=>{if(child.isMesh||child.isSprite){child.userData.destination='camp';interactive.push(child)}});
  scene.add(camp);glows.push({flame,campGlow});
  const points=[];let seed=47;for(let i=0;i<450;i++){
    seed=(seed*1664525+1013904223)>>>0;const x=(seed/4294967296-.5)*120;
    seed=(seed*1664525+1013904223)>>>0;const y=7+(seed/4294967296)*54;
    seed=(seed*1664525+1013904223)>>>0;const z=(seed/4294967296-.5)*120;
    points.push(x,y,z);
  }
  const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(points,3));
  starfield=new THREE.Points(geometry,new THREE.PointsMaterial({color:0xffd4e9,size:.14,sizeAttenuation:true}));
  scene.add(starfield);
  player=createCharacter();
}
function resize(){
  if(!renderer||!camera)return;
  const w=Math.max(1,stage.clientWidth),h=Math.max(1,stage.clientHeight),extent=15.8,aspect=w/h;
  camera.left=-extent*aspect;camera.right=extent*aspect;
  camera.top=extent;camera.bottom=-extent;camera.updateProjectionMatrix();
  renderer.setSize(w,h,false);
}
function init3d(){
  try{
    renderer=new THREE.WebGLRenderer({antialias:true,alpha:false,powerPreference:'low-power'});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,1.5));renderer.outputColorSpace=THREE.SRGBColorSpace;
    renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.4;
    stage.append(renderer.domElement);
    camera=new THREE.OrthographicCamera(-10,10,15,-15,.1,200);
    camera.position.set(18,23,28);camera.lookAt(0,0,0);
    makeWorld();resize();
    if('ResizeObserver' in window)new ResizeObserver(resize).observe(stage);
    else window.addEventListener('resize',resize);
    renderer.domElement.addEventListener('pointerup',hitTest);
    renderer.domElement.addEventListener('webglcontextlost',event=>{event.preventDefault();renderFailure('The graphics context was lost. Restart the app to return to your saved journey.');});
    requestAnimationFrame(frame);
  }catch(error){renderFailure(error.message);}
}
function renderFailure(message){$('#renderMessage').textContent=message;overlay('renderError',true);}
const ray=new THREE.Raycaster(),mouse=new THREE.Vector2();
function hitTest(e){
  if(anyOverlay()||!gameStarted)return;
  const rect=renderer.domElement.getBoundingClientRect();
  mouse.set((e.clientX-rect.left)/rect.width*2-1,-(e.clientY-rect.top)/rect.height*2+1);
  ray.setFromCamera(mouse,camera);
  const hit=ray.intersectObjects(interactive,false).find(item=>item.object.userData.destination);
  if(hit){const dest=hit.object.userData.destination;if(dest==='camp')goCamp();else goTo(dest);}
}
function goTo(id){
  if(anyOverlay()||!gameStarted)return;
  const b=BEACONS.find(item=>item.id===id);if(!b)return;
  if(run.lit.includes(id)){status(b.name+' is already glowing. Try an unlit place.');return;}
  travel={x:b.x,z:b.z,id};status('Walking to '+b.name+'…');
}
function goCamp(){
  if(anyOverlay()||!gameStarted)return;
  travel={x:0,z:0,id:'camp'};status('Returning to the campfire…');
}
function arrive(){
  const dest=travel;travel=null;
  if(!dest)return;
  if(dest.id==='camp'){
    try{run=rechargeAtCamp(run);save();updateHud();speakNote(480,.3);status('Rested at camp: +5 focus, up to 9. A small points penalty applies.');}
    catch(error){status(error.message);}
    return;
  }
  if(!run.lit.includes(dest.id))showDecision(dest.id);
}
function frame(now){
  requestAnimationFrame(frame);
  if(!renderer||!scene)return;
  const dt=Math.min((now-(lastFrame||now))/1000,.06);lastFrame=now;
  if(!document.hidden){
    const free=!anyOverlay()&&gameStarted;
    let dx=0,dz=0;
    if(free){
      if(keys.has('left'))dx-=1;if(keys.has('right'))dx+=1;
      if(keys.has('up'))dz-=1;if(keys.has('down'))dz+=1;
    }
    const manual=Math.hypot(dx,dz)>0;
    if(manual)travel=null;
    if(!manual&&free&&travel){
      const vx=travel.x-player.position.x,vz=travel.z-player.position.z,dist=Math.hypot(vx,vz);
      if(dist<1.25){arrive();}
      else{dx=vx/dist;dz=vz/dist;}
    }
    const d=Math.hypot(dx,dz);
    if(d>0){player.position.x=Math.max(-10,Math.min(10,player.position.x+dx/d*dt*4.7));
      player.position.z=Math.max(-10,Math.min(10,player.position.z+dz/d*dt*4.7));
      player.rotation.y=Math.atan2(dx,dz);arms[0].rotation.x=Math.sin(now*.01)*.42;arms[1].rotation.x=-Math.sin(now*.01)*.42;
      player.userData.legs[0].rotation.x=Math.sin(now*.01)*.28;player.userData.legs[1].rotation.x=-Math.sin(now*.01)*.28;
    }else{arms[0].rotation.x=arms[1].rotation.x=0;player.userData.legs.forEach(leg=>leg.rotation.x=0);}
    cloak.rotation.z=Math.sin(now*.0017)*.018;
    for(const {orb,ring,light,glow} of lighting){
      orb.rotation.y+=dt*.65;ring.rotation.z+=dt*.3;
      light.intensity=1.4+Math.sin(now*.002)*.25+(orb.material.emissiveIntensity>1?1.3:0);
      glow.material.opacity=.55+Math.sin(now*.0021)*.12;
    }
    if(glows.length)glows[0].flame.rotation.y+=dt*1.1;
    if(starfield)starfield.rotation.y+=dt*.002;
    renderer.render(scene,camera);
  }
}
function showDecision(id){
  if(anyOverlay())return;
  const b=BEACONS.find(item=>item.id===id);pending=id;
  $('#decisionEyebrow').textContent=b.name.toUpperCase()+' · LIGHT '+(run.lit.length+1)+' OF 5';
  $('#decisionTitle').textContent=b.title;
  $('#challenge').textContent=b.challenge;
  $('#choiceGate').textContent='Current path · Gate '+run.path.gate;
  $('#choiceFocus').textContent='Available focus · '+run.focus;
  $('#decisionStatus').textContent='Your choice changes the next source-backed gate.';
  const choices=$('#choices');choices.replaceChildren();
  for(const choice of offerChoices(run)){
    const button=document.createElement('button');button.disabled=!choice.affordable;
    const heading=document.createElement('strong');heading.textContent=choice.id+'. '+choice.label;
    const meaning=document.createElement('small');meaning.textContent=choice.meaning;
    const effect=document.createElement('span');effect.className='effect';effect.textContent='−'+choice.cost+' focus  ·  +'+choice.points+' points';
    button.append(heading,meaning,effect);
    button.onclick=()=>makeChoice(choice.id);
    choices.append(button);
  }
  if(run.focus===0)$('#decisionStatus').textContent='No focus left. Walk back to camp, then return.';
  overlay('decision',true);choices.querySelector('button:not(:disabled)')?.focus();
}
async function makeChoice(line){
  if(choiceBusy||!pending)return;
  choiceBusy=true;for(const btn of $('#choices').children)btn.disabled=true;
  try{
    const b=BEACONS.find(item=>item.id===pending);
    const result=await activateBeacon(run,pending,line);
    run=result.run;pending=null;overlay('decision',false);save();updateHud();sound();
    status(b.name+' restored! Gate '+result.transition.fromGate+' → '+result.transition.toGate+'. '+(5-run.lit.length)+' lights left.');
    if(run.finishedAt!==null){showEnding();}
  }catch(error){
    $('#decisionStatus').textContent=error.message;
    for(const [i,button] of [...$('#choices').children].entries())button.disabled=!offerChoices(run)[i].affordable;
  }finally{choiceBusy=false;}
}
function showEnding(){
  const seconds=Math.max(0,Math.floor(((run.finishedAt??Date.now())-run.startedAt)/1000));
  const mm=Math.floor(seconds/60),ss=String(seconds%60).padStart(2,'0');
  $('#endScore').textContent=run.score+' points · '+mm+':'+ss+' · '+run.recharges+' camp visits';
  const list=$('#endChronicle');list.replaceChildren();
  for(const item of run.journal.filter(entry=>entry.type==='light')){
    const p=document.createElement('p');p.textContent=item.dimension+' · '+item.choice+' ('+item.fromGate+' → '+item.toGate+')';
    list.append(p);
  }
  overlay('ending',true);$('#playAgain').focus();
}
function bindControls(){
  const mapping={ArrowUp:'up',KeyW:'up',ArrowDown:'down',KeyS:'down',ArrowLeft:'left',KeyA:'left',ArrowRight:'right',KeyD:'right'};
  window.addEventListener('keydown',e=>{
    const direction=mapping[e.code];if(!direction)return;
    if(/^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName||''))return;
    keys.add(direction);e.preventDefault();
  });
  window.addEventListener('keyup',e=>{if(mapping[e.code]){keys.delete(mapping[e.code]);e.preventDefault();}});
  window.addEventListener('blur',()=>keys.clear());
  document.querySelectorAll('[data-move]').forEach(button=>{
    const key=button.dataset.move;
    button.addEventListener('pointerdown',e=>{e.preventDefault();keys.add(key);button.setPointerCapture(e.pointerId);});
    for(const name of ['pointerup','pointercancel','lostpointercapture'])button.addEventListener(name,()=>keys.delete(key));
  });
  $('#camp').onclick=goCamp;
  $('#closeDecision').onclick=()=>{if(choiceBusy)return;pending=null;overlay('decision',false);status('Come back to this light whenever you are ready.');};
  $('#start').onclick=()=>{gameStarted=true;overlay('intro',false);if(run.finishedAt!==null)showEnding();else status('Tap a glowing light to walk there.');stage.focus();};
  $('#playAgain').onclick=newGame;
  $('#reset').onclick=()=>{
    if(!resetArmed){resetArmed=true;$('#reset').textContent='Confirm new game';status('Press Confirm new game again to erase this journey.');return;}
    newGame();
  };
}
load();bindControls();init3d();updateHud();
$('#resumeHint').textContent=hadSave?(run.finishedAt!==null?'Your completed journey is here. Start to see its ending.':'Your saved progress is ready: '+run.lit.length+' of 5 lights restored.'):'Your decisions are saved on this device as you play.';
$('#start').textContent=hadSave?'Continue your journey ✦':'Begin the journey ✦';
overlay('intro',true);
