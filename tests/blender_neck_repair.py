"""Run with Blender --background --factory-startup --python this_file."""
from pathlib import Path
import sys, math
import bpy, bmesh
from mathutils import Vector
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mmd_transplant.neck_blender import repair_existing


def fixture(head_count=20, gap=.08):
    arm_data=bpy.data.armatures.new('synthetic_arm')
    arm=bpy.data.objects.new('synthetic_arm',arm_data)
    bpy.context.collection.objects.link(arm)
    bpy.context.view_layer.objects.active=arm;arm.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    for name,co in [('neck',(0,0,-.4)),('head',(0,0,0)),('eye_l',(.3,0,.3)),('eye_r',(-.3,0,.3))]:
        bone=arm_data.edit_bones.new(name);bone.head=co;bone.tail=(co[0],co[1],co[2]+.1)
    bpy.ops.object.mode_set(mode='OBJECT')
    vertices=[];faces=[];sides=[]
    for count,radius,low,high,side in [(16,.15,-1,0,'body'),(head_count,.24,gap,1,'head')]:
        start=len(vertices)
        for height in (low,high):
            vertices.extend((math.cos(i*2*math.pi/count)*radius,math.sin(i*2*math.pi/count)*radius,height) for i in range(count))
        for i in range(count):
            j=(i+1)%count
            faces.append((start+i,start+j,start+count+j,start+count+i));sides.append(side)
    data=bpy.data.meshes.new('synthetic_mesh');data.from_pydata(vertices,[],faces)
    mesh=bpy.data.objects.new('synthetic_mesh',data);bpy.context.collection.objects.link(mesh)
    for name in ['身_skin','头_skin']:
        data.materials.append(bpy.data.materials.new(name))
    for polygon,side in zip(data.polygons,sides):polygon.material_index=int(side=='head')
    uv=data.uv_layers.new(name='UVMap')
    for loop in data.loops:uv.data[loop.index].uv=(loop.vertex_index/len(vertices),.4)
    for name,indices in [('neck',range(32)),('head',range(32,len(vertices)))]:
        mesh.vertex_groups.new(name=name).add(list(indices),1,'REPLACE')
    mesh.vertex_groups.new(name='aux').add(list(range(len(vertices))),.5,'REPLACE')
    mesh.modifiers.new('arm','ARMATURE').object=arm
    basis=mesh.shape_key_add(name='Basis');expression=mesh.shape_key_add(name='expression')
    for i in range(32,len(vertices)):expression.data[i].co.x+=.03
    data.normals_split_custom_set([(0,1,0)]*len(data.loops))
    return mesh,arm


def verify(count):
    mesh,arm=fixture(count)
    old=mesh.data
    coordinates=[v.co.copy() for v in old.vertices]
    offsets=[k.co-old.shape_keys.key_blocks[0].data[i].co for i,k in enumerate(old.shape_keys.key_blocks[1].data)]
    old_uv=[tuple(u.uv) for u in old.uv_layers.active.data]
    old_normals=[tuple(loop.normal) for loop in old.loops]
    twin=mesh.copy();bpy.context.collection.objects.link(twin)
    report=repair_existing(mesh)
    assert report['status']=='bridged',report
    assert twin.data==old and len(old.vertices)==len(coordinates)
    assert len(mesh.data.vertices)==len(coordinates)+16+count
    assert report['bridge_faces']==16+count
    for i,co in enumerate(coordinates):assert (mesh.data.vertices[i].co-co).length<1e-6
    assert [tuple(u.uv) for u in mesh.data.uv_layers.active.data][:len(old_uv)]==old_uv
    for a,b in zip(old_normals,mesh.data.loops):assert (b.normal-Vector(a)).length<1e-3,(a,tuple(b.normal))
    for vertex in mesh.data.vertices[len(coordinates):]:
        index=min(range(len(coordinates)),key=lambda i:(coordinates[i]-vertex.co).length)
        assert (coordinates[index]-vertex.co).length<1e-6
        assert {mesh.vertex_groups[g.group].name for g in vertex.groups} <= {'neck','head'}
        delta=mesh.data.shape_keys.key_blocks[1].data[vertex.index].co-mesh.data.shape_keys.key_blocks[0].data[vertex.index].co
        assert (delta-offsets[index]).length<1e-6
    assert repair_existing(mesh)['status']=='already_bridged'
    mesh.data.shape_keys.key_blocks[1].value=.7
    arm.pose.bones['head'].rotation_mode='XYZ';arm.pose.bones['head'].rotation_euler.y=.2
    bpy.context.view_layer.update()
    evaluated=mesh.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh()
    for i in range(len(coordinates),len(mesh.data.vertices)):
        source=min(range(len(coordinates)),key=lambda j:(coordinates[j]-mesh.data.vertices[i].co).length)
        assert (evaluated.vertices[i].co-evaluated.vertices[source].co).length<1e-5
    mesh.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh_clear()
    bm=bmesh.new();bm.from_mesh(mesh.data)
    face=next(f for f in bm.faces if f.material_index==report['material'])
    bmesh.ops.delete(bm,geom=[face],context='FACES_ONLY');bm.to_mesh(mesh.data);bm.free()
    assert repair_existing(mesh)['status']=='rebridged'
    assert repair_existing(mesh)['status']=='already_bridged'


verify(20);verify(16)
mesh,arm=fixture(gap=2)
old=mesh.data
try:repair_existing(mesh)
except ValueError:pass
else:raise AssertionError('Unsafe gap accepted')
assert mesh.data==old
mesh,arm=fixture();mesh.shape_key_add(name='mmd_sdef_cache');old=mesh.data
try:repair_existing(mesh)
except ValueError:pass
else:raise AssertionError('Bound SDEF accepted')
assert mesh.data==old
import mmd_transplant
mmd_transplant.register()
mesh,arm=fixture()
bpy.context.view_layer.objects.active=mesh
assert bpy.ops.mmd_transplant.fit_existing_neck()=={'FINISHED'}
assert '补面连接' in bpy.context.scene.mmd_transplant.status
assert bpy.ops.mmd_transplant.fit_existing_neck()=={'FINISHED'}
assert '无需重复桥接' in bpy.context.scene.mmd_transplant.status
assert mmd_transplant.bl_info['version']==(0,3,3)
print('NECK_GEOMETRY_VERIFIED: unequal/equal rims, weights, expressions, UVs, normals, shared data, repeat, rebridge, unsafe gap, SDEF')
