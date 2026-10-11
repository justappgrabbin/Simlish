"""Override the supplied Blender builder's hardcoded input/output boundaries."""
import sys, importlib.util, os
from pathlib import Path
args=sys.argv[sys.argv.index('--')+1:];script,photo,out,model=args
spec=importlib.util.spec_from_file_location('unum_builder',script);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
# Keep InsightFace model storage in the configured workspace boundary.
if os.environ.get('UNUM_INSIGHTFACE_ROOT'):
 import insightface.app
 original=insightface.app.FaceAnalysis
 def configured_analysis(*args,**kwargs):
  kwargs['root']=os.environ['UNUM_INSIGHTFACE_ROOT'];return original(*args,**kwargs)
 insightface.app.FaceAnalysis=configured_analysis
sys.path.insert(0,str(Path(__file__).parent))
from body import add_body
module.add_shoulders=add_body
def export_full_body(path):
 import bpy
 bpy.ops.export_scene.gltf(filepath=path,export_format='GLB',export_apply=False,export_skins=True,export_morph=True,export_animations=True,export_yup=False,export_cameras=False,export_lights=False)
module.export_glb=export_full_body
module.FRONT_IMG=photo;module.OUTPUT=out;module.INPUT=str(Path(photo).parent);module.DATA=str(Path(model).parent);module.MODEL_PATH=model;module.OUT_GLB=str(Path(out)/'avatar.glb');module.main()
