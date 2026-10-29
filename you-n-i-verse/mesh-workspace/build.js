const fs=require('fs');const path=require('path');const root=__dirname;
let html=fs.readFileSync(path.join(root,'src','shell.html'),'utf8');
for(const [token,file] of [['/*CSS_INLINE*/','styles.css'],['/*CORE_INLINE*/','engine.js'],['/*UI_INLINE*/','ui.js']]){
let data=fs.readFileSync(path.join(root,'src',file),'utf8');if(file.endsWith('.js'))data=data.replace(/<\/script/gi,'<\\/script');html=html.replace(token,()=>data);
}
fs.writeFileSync(path.join(root,'index.html'),html);
console.log('Built index.html: '+Buffer.byteLength(html)+' bytes (self-contained, no external imports)');
