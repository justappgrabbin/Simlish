"""Override the supplied Blender builder's hardcoded input/output boundaries."""
import sys, importlib.util
from pathlib import Path
args=sys.argv[sys.argv.index('--')+1:];script,photo,out,model=args
spec=importlib.util.spec_from_file_location('unum_builder',script);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.FRONT_IMG=photo;module.OUTPUT=out;module.INPUT=str(Path(photo).parent);module.DATA=str(Path(model).parent);module.MODEL_PATH=model;module.OUT_GLB=str(Path(out)/'avatar.glb');module.main()
