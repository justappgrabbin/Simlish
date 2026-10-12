"""Generic inferred humanoid, retaining the photo-derived head and facial morphs."""
import bpy, math
from mathutils import Vector

def add_body(points, skin):
 h=(Vector(points[10])-Vector(points[152])).length;c=Vector(points[152]);c.y-=.28*h;c.z-=.08*h
 def p(x,y):return c+Vector((x*h,-y*h,0))
 joints={'hips':(p(0,2.3),p(0,1.5),None),'chest':(p(0,1.5),p(0,0),'hips'),'head':(p(0,0),p(0,-1),'chest')}
 for side,s in [('Left',1),('Right',-1)]:
  for name,a,b,parent in [('UpperArm',(.7,.15),(1.4,.4),'chest'),('Forearm',(1.4,.4),(2,.7),side+'UpperArm'),('Hand',(2,.7),(2.3,.8),side+'Forearm'),('Thigh',(.35,2.3),(.35,3.7),'hips'),('Shin',(.35,3.7),(.35,5),side+'Thigh'),('Foot',(.35,5),(.35,5.3),side+'Shin')]:joints[side+name]=(p(s*a[0],a[1]),p(s*b[0],b[1]),parent)
 originals=list(bpy.context.scene.objects);arm=bpy.data.armatures.new('HumanoidRig');rig=bpy.data.objects.new('HumanoidRig',arm);bpy.context.collection.objects.link(rig);bpy.context.view_layer.objects.active=rig;rig.select_set(True)
 bpy.ops.object.mode_set(mode='EDIT')
 for name,(a,b,parent) in joints.items():
  bone=arm.edit_bones.new(name);bone.head=a;bone.tail=b
  if parent:bone.parent=arm.edit_bones[parent]
 bpy.ops.object.mode_set(mode='OBJECT')
 def bind(obj,name):
  bpy.context.view_layer.objects.active=obj; bpy.ops.object.select_all(action='DESELECT');obj.select_set(True);bpy.ops.object.transform_apply(location=True,rotation=True,scale=True)
  group=obj.vertex_groups.new(name=name);group.add(list(range(len(obj.data.vertices))),1,'REPLACE');mod=obj.modifiers.new('Humanoid skin','ARMATURE');mod.object=rig
 for obj in originals:
  if obj.type=='MESH':bind(obj,'head')
 mat=bpy.data.materials.new('Inferred clothing');mat.diffuse_color=(.09,.15,.25,1)
 def segment(name,a,b,w,d,bone):
  bpy.ops.mesh.primitive_uv_sphere_add(segments=20,ring_count=12,location=(a+b)/2);obj=bpy.context.object;obj.name=name;obj.rotation_mode='QUATERNION';obj.rotation_quaternion=(b-a).to_track_quat('Y','Z');obj.scale=(w*h,(b-a).length*.57,d*h);bpy.ops.object.transform_apply(location=False,rotation=False,scale=True);obj.data.materials.append(mat)
  for face in obj.data.polygons:face.use_smooth=True
  bind(obj,bone);obj.shape_key_add(name='Basis')
  for key,scale in [('BodyBroad',1.25),('BodySlender',.8)]:
   shape=obj.shape_key_add(name=key)
   for i,v in enumerate(obj.data.vertices):shape.data[i].co.x=(v.co.x-((a+b)/2).x)*scale+((a+b)/2).x
 segment('Torso',p(0,.1),p(0,1.6),.7,.34,'chest');segment('Pelvis',p(0,1.6),p(0,2.5),.52,.3,'hips')
 for side in ['Left','Right']:
  for part,w,d in [('UpperArm',.19,.19),('Forearm',.15,.15),('Hand',.14,.09),('Thigh',.27,.26),('Shin',.19,.19),('Foot',.2,.35)]:
   name=side+part;a,b,_=joints[name];segment(name,a,b,w,d,name)
 for side in ['Left','Right']:
  for part,w in [('UpperArm',.22),('Forearm',.16),('Hand',.13),('Thigh',.26),('Shin',.19),('Foot',.16)]:
   name=side+part;a,b,_=joints[name];segment(side+part+'Joint',a-Vector((0,w*h,0)),a+Vector((0,w*h,0)),w,w,name)
 for frame in [1,13,25,37,49]:
  phase=(frame-1)/48*math.pi*2
  for side,s in [('Left',1),('Right',-1)]:
   for part,amount in [('Thigh',.3),('Shin',.15),('UpperArm',-.2)]:
    bone=rig.pose.bones[side+part];bone.rotation_mode='XYZ';bone.rotation_euler.x=math.sin(phase)*s*amount;bone.keyframe_insert(data_path='rotation_euler',frame=frame)
 bpy.context.scene.frame_start=1;bpy.context.scene.frame_end=49;bpy.context.scene.frame_set(1);rig['body_provenance']='Generic inferred body; not reconstructed body proportions or clothing.'
 return rig
