"""Convert DECA output geometry/materials to a portable embedded GLB in Blender."""
import sys,bpy
from pathlib import Path
source,out=sys.argv[sys.argv.index('--')+1:]
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
bpy.ops.wm.obj_import(filepath=source)
bpy.ops.export_scene.gltf(filepath=out,export_format='GLB')
