"""Adapters for the supplied upstream interfaces. No shell interpolation."""
from pathlib import Path
import os, sys, subprocess, shutil

ROOTS = {
 'deca': ('DECA_ROOT', 'run/reconstruct_cmd.py'),
 'instantid': ('INSTANTID_ROOT', 'pipeline_stable_diffusion_xl_instantid_img2img.py'),
 'liveportrait': ('LIVEPORTRAIT_ROOT', 'inference.py'),
 'liveportrait-animal': ('LIVEPORTRAIT_ROOT', 'inference_animals.py'),
 'facefusion': ('FACEFUSION_ROOT', 'facefusion.py'),
 'unum': ('UNUM_ROOT', 'scripts/build_avatar_blender.py'),
}
WORKERS = Path(__file__).parent

def readiness(engine):
 if engine == 'text-image':
  path = Path(os.environ.get('SDXL_MODEL', '/not-configured'))
  return {'ready':path.is_dir() and engine in os.environ.get('MORPH_READY_ENGINES','').split(','), 'reason': 'Set SDXL_MODEL to a locally installed Diffusers model directory.'}
 env, entry = ROOTS[engine]
 root = Path(os.environ.get(env, '/not-configured'))
 configured = (root / entry).is_file()
 # The operator must explicitly confirm upstream dependencies and checkpoints.
 provisioned = engine in os.environ.get('MORPH_READY_ENGINES', '').split(',')
 return {'ready': configured and provisioned, 'reason':f'Configure {env}, install upstream dependencies/checkpoints, then include {engine} in MORPH_READY_ENGINES.'}

def command(engine, job, params):
 out = job / 'output'; out.mkdir(exist_ok=True)
 python = os.environ.get('MODEL_PYTHON_'+engine.upper().replace('-','_'),os.environ.get('MODEL_PYTHON', sys.executable))
 if engine == 'text-image':
  return [python,str(WORKERS/'generate.py'),'--mode','text-image','--output',str(out/'image.png'),'--prompt',params['prompt'],'--seed',str(params.get('seed',0))], WORKERS
 env, entry = ROOTS[engine]; root = Path(os.environ[env]).resolve()
 photo = str(job/'photo.png')
 if engine == 'deca':
  return [python,str(root/entry),'-i',photo,'-s',str(out),'--device',os.environ.get('MORPH_DEVICE','cpu'),'--saveObj','True','--saveKpt','True','--extractTex','True'],root
 if engine == 'instantid':
  return [python,str(WORKERS/'generate.py'),'--mode','instantid','--photo',photo,'--root',str(root),'--output',str(out/'image.png'),'--prompt',params['prompt'],'--seed',str(params.get('seed',0))],root
 if engine.startswith('liveportrait'):
  return [python,str(root/entry),'--source',photo,'--driving',str(job/'driving.mp4'),'--output-dir',str(out)],root
 if engine == 'facefusion':
  return [python,str(root/entry),'headless-run','--source-paths',photo,'--target-path',str(job/'target.png'),'--output-path',str(out/'image.png'),'--processors','face_swapper'],root
 if engine == 'unum':
  return [os.environ.get('BLENDER_BIN','blender'),'--background','--python',str(WORKERS/'unum.py'),'--',str(root/entry),photo,str(out),os.environ['UNUM_FACE_MODEL']],root
 raise ValueError('Unsupported engine')

def execute(engine, job, params):
 args,cwd=command(engine,job,params)
 with (job/'worker.log').open('w') as log:
  result=subprocess.run(args,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,timeout=int(os.environ.get('MORPH_TIMEOUT','1800')))
 if result.returncode:raise RuntimeError(f'{engine} failed; see worker.log (exit {result.returncode}).')
 if engine=='deca' and shutil.which(os.environ.get('BLENDER_BIN','blender')):
  for mesh in (job/'output').rglob('*.obj'):
   with (job/'worker.log').open('a') as log:
    conversion=subprocess.run([os.environ.get('BLENDER_BIN','blender'),'--background','--python',str(WORKERS/'obj_to_glb.py'),'--',str(mesh),str(mesh.with_suffix('.glb'))],stdout=log,stderr=subprocess.STDOUT,timeout=300)
   if conversion.returncode:raise RuntimeError('DECA mesh produced; GLB conversion failed. See worker.log.')
 files=[p for p in (job/'output').rglob('*') if p.is_file()]
 if not files:raise RuntimeError('Engine returned without output files.')
 return files
