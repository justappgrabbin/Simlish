// Appearance presets never alter photo pixels, facial geometry or identity.
export const AVATAR_THEMES={cosmic:{background:'#161128',top:'#344f78',bottom:'#202c45',shoes:'#171d29'},forest:{background:'#183c29',top:'#476b42',bottom:'#35422d',shoes:'#382d23'},ocean:{background:'#123c59',top:'#328aaf',bottom:'#203e65',shoes:'#192c39'},desert:{background:'#5b3728',top:'#be9163',bottom:'#69503c',shoes:'#392e26'}};
const linear=v=>v<=.04045?v/12.92:((v+.055)/1.055)**2.4;
export function applyAvatarTheme(model,world){
 const theme=AVATAR_THEMES[world];if(!theme)throw Error('Unsupported appearance theme');
 if(!model||model.userData.appearanceWorld===world)return false;
 let changed=false;
 model.traverse(node=>{const slots=node.userData.appearanceSlots,colors=node.geometry?.attributes.color;if(!slots||!colors)return;for(const slot of ['top','bottom','shoes']){const hex=theme[slot].slice(1),rgb=[0,2,4].map(i=>linear(parseInt(hex.slice(i,i+2),16)/255));for(const index of slots[slot]??[])colors.setXYZ(index,...rgb);}colors.needsUpdate=true;changed=true;});
 if(changed)model.userData.appearanceWorld=world;
 return changed;
}
