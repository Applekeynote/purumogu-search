"""Illustrative product/mechanism source renders. Final editing is done in Resolve.
Run with Blender 5.1 --background --python this_file -- --preview or --render.
The model follows RescueRobot's 7cm sphere/taper proportions; it is not a physics trial.
"""
import bpy, math, sys, os
from pathlib import Path
from mathutils import Vector

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'media/concept_film_sources/robot-stage'
OUT.mkdir(parents=True,exist_ok=True)
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
scene=bpy.context.scene
scene.render.engine='CYCLES'
scene.cycles.samples=24
scene.cycles.use_denoising=True
scene.cycles.device='GPU'
try:
    prefs=bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type='OPTIX';prefs.get_devices()
    for device in prefs.devices:device.use=device.type!='CPU'
    if not any(d.use for d in prefs.devices):scene.cycles.device='CPU'
except Exception:scene.cycles.device='CPU'
scene.render.resolution_x=1920;scene.render.resolution_y=1080
scene.render.resolution_percentage=100;scene.render.fps=30
scene.render.image_settings.file_format='PNG'
scene.render.image_settings.color_mode='RGBA'
scene.render.film_transparent=False
scene.world.color=(.018,.018,.018)
scene.view_settings.view_transform='AgX'
scene.render.image_settings.color_depth='8'
if os.environ.get('PURUMOGU_EEVEE')=='1':
    scene.render.engine='BLENDER_EEVEE'
    scene.eevee.taa_render_samples=32

def mat(name,color,metal,rough,emission=0):
    m=bpy.data.materials.new(name);m.diffuse_color=(*color,1);m.use_nodes=True
    n=next(n for n in m.node_tree.nodes if n.type=='BSDF_PRINCIPLED');n.inputs['Base Color'].default_value=(*color,1)
    n.inputs['Metallic'].default_value=metal;n.inputs['Roughness'].default_value=rough
    if emission:n.inputs['Emission Color'].default_value=(*color,1);n.inputs['Emission Strength'].default_value=emission
    return m
shell=mat('Brushed titanium',(.36,.40,.39),.88,.25)
dark=mat('Obsidian sensor face',(.012,.025,.029),.68,.19)
mint=mat('Docking port — mint',(.04,.85,.58),.45,.22,2)
floor=mat('Dark studio floor',(.008,.013,.018),.30,.32)

def empty(name,parent=None,location=(0,0,0)):
    o=bpy.data.objects.new(name,None);scene.collection.objects.link(o);o.parent=parent;o.location=location;return o
def smooth(o):
    for p in o.data.polygons:p.use_smooth=True
def sphere(name,parent,loc,scale,material):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64,ring_count=32)
    o=bpy.context.object;o.name=name;o.parent=parent;o.location=loc;o.scale=scale;o.data.materials.append(material);smooth(o);return o
def torus(name,parent,loc,major,minor,rotation=(0,0,0),material=dark):
    bpy.ops.mesh.primitive_torus_add(major_radius=major,minor_radius=minor,major_segments=96,minor_segments=12)
    o=bpy.context.object;o.name=name;o.parent=parent;o.location=loc;o.rotation_euler=rotation;o.data.materials.append(material);smooth(o);return o
def tapered(parent,sign):
    pivot=empty('Articulated cone pivot',parent,(sign*.75,0,0))
    verts=[];faces=[];rings=[(0,.88),(.5,.92),(2.1,.37),(2.65,.16)];sides=64
    for x,r in rings:
        for i in range(sides):
            a=2*math.pi*i/sides
            verts.append((sign*x,math.cos(a)*r,max(-.74*r,min(.74*r,math.sin(a)*r))))
    for k in range(3):
        for i in range(sides):
            a=k*sides+i;b=k*sides+(i+1)%sides
            face=(a,b,b+sides,a+sides);faces.append(face if sign>0 else face[::-1])
    faces.extend([tuple(range(sides-1,-1,-1)),tuple(range(3*sides,4*sides))])
    mesh=bpy.data.meshes.new('FlattenedRoundedTaper');mesh.from_pydata(verts,[],faces);mesh.update()
    o=bpy.data.objects.new('Tapered titanium cone',mesh);scene.collection.objects.link(o);o.parent=pivot;o.data.materials.append(shell);smooth(o)
    b=o.modifiers.new('Soft manufactured edges','BEVEL');b.width=.045;b.segments=3
    o.modifiers.new('Weighted surface normals','WEIGHTED_NORMAL')
    sphere('Spherical seating ring',pivot,(0,0,0),(.45,.90,.90),shell)
    points=[(sign*.32,-.60,.70),(sign*.32,.60,.70),(sign*2.24,.15,.25),(sign*2.24,-.15,.25)]
    panel=bpy.data.meshes.new('Inset face');panel.from_pydata(points,[],[(0,1,2,3)]);panel.update()
    face=bpy.data.objects.new('Dark sensor panel',panel);scene.collection.objects.link(face);face.parent=pivot;face.data.materials.append(dark)
    solid=face.modifiers.new('Inset depth','SOLIDIFY');solid.thickness=.018
    bevel=face.modifiers.new('Panel corner radius','BEVEL');bevel.width=.06;bevel.segments=3
    sphere('Tip magnetic contact',pivot,(sign*2.65,0,0),(.10,.22,.22),mint)
    return pivot

def robot(name):
    root=empty(name)
    sphere('Central spherical joint',root,(0,0,0),(.95,.95,.95),dark)
    torus('Ball equator seam',root,(0,0,0),.956,.018,(math.pi/2,0,0),shell)
    for y,z,scale in [(1,0,(.17,.04,.17)),(-1,0,(.17,.04,.17)),(0,1,(.17,.17,.04)),(0,-1,(.17,.17,.04))]:
        sphere('Sphere magnetic contact',root,(0,y*.96,z*.96),scale,mint)
    cones=[tapered(root,s) for s in [1,-1]]
    return root,cones

r0,c0=robot('NODE 01');r1,c1=robot('NODE 02');r2,c2=robot('NODE 03')
for root in [r1,r2]:
    root.scale=(0,0,0);root.keyframe_insert('scale',frame=1)
    root.scale=(0,0,0);root.keyframe_insert('scale',frame=239)
    root.scale=(1,1,1);root.keyframe_insert('scale',frame=240)
for f,rot in [(1,-.22),(120,.12),(239,.42),(240,0),(720,0)]:
    r0.rotation_euler=(0,0,rot);r0.keyframe_insert('rotation_euler',frame=f)
for f in [1,60,120,180,239,240,300,360,420,480,540,600,660,720]:
    for i,c in enumerate(c0):
        c.rotation_euler=(math.sin(f*.025+i)*.45,math.sin(f*.018+i)*.24,math.sin(f*.014+i)*.17) if f<239 else (0,0,0)
        c.keyframe_insert('rotation_euler',frame=f)
r1.location=(10.5,0,0);r1.keyframe_insert('location',frame=240)
r1.location=(6.8,0,0);r1.keyframe_insert('location',frame=360)
r1.keyframe_insert('location',frame=530)
r1.location=(9.7,0,0);r1.keyframe_insert('location',frame=650)
r2.location=(0,6,0);r2.keyframe_insert('location',frame=240)
r2.keyframe_insert('location',frame=400)
r2.location=(0,1.93,0);r2.keyframe_insert('location',frame=500)
r2.keyframe_insert('location',frame=545)
r2.location=(-2,5,0);r2.keyframe_insert('location',frame=690)

bpy.ops.mesh.primitive_plane_add(size=200,location=(0,0,-1.6));bpy.context.object.data.materials.append(floor)
def area(name,loc,energy,color,size,target=(0,0,0)):
    data=bpy.data.lights.new(name,'AREA');data.energy=energy;data.color=color;data.shape='DISK';data.size=size
    o=bpy.data.objects.new(name,data);scene.collection.objects.link(o);o.location=loc;o.rotation_euler=(Vector(target)-o.location).to_track_quat('-Z','Y').to_euler()
area('Key softbox',(1,-8,12),2100,(.82,.93,1),9)
area('Mint edge',(0,6,5),1500,(.15,1,.72),7)
area('White strip',(-8,2,7),1900,(1,.98,.92),5)
area('Right fill',(12,-3,2),700,(.33,.55,1),5)
camData=bpy.data.cameras.new('Product cinema camera');cam=bpy.data.objects.new('Product cinema camera',camData);scene.collection.objects.link(cam);scene.camera=cam
camData.lens=53;camData.clip_start=.1;camData.clip_end=300
for f,pos,target in [(1,(8,-15,9),(0,0,0)),(239,(6,-15,7),(0,0,0)),(240,(13,-23,13),(2.4,.5,0)),(720,(13,-23,13),(2.4,.5,0))]:
    cam.location=pos;cam.rotation_euler=(Vector(target)-cam.location).to_track_quat('-Z','Y').to_euler();cam.keyframe_insert('location',frame=f);cam.keyframe_insert('rotation_euler',frame=f)
scene.frame_start=1;scene.frame_end=720
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'robot-stage.blend'))
if '--render' in sys.argv or os.environ.get('PURUMOGU_RENDER_ANIM')=='1':
    scene.render.filepath=str(OUT/'frame-');bpy.ops.render.render(animation=True)
else:
    scene.frame_set(150);scene.render.filepath=str(OUT/'preview.png');bpy.ops.render.render(write_still=True)
