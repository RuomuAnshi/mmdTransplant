import math
import importlib.util
import pathlib
import sys
import types
import unittest

if 'mmd_transplant' not in sys.modules:
    package=types.ModuleType('mmd_transplant')
    package.__path__=[str(pathlib.Path(__file__).resolve().parents[1]/'mmd_transplant')]
    sys.modules['mmd_transplant']=package
from mmd_transplant.neck import fit_plan,bridge_plan
from mmd_transplant.core import fit_neck_pmx,influences


class Weight:
    BDEF1=0
    BDEF4=2
    def __init__(self):
        self.type=0
        self.bones=[0]
        self.weights=[]


class Offset:
    pass


class Morph:
    def __init__(self,offsets,kind=1):
        self.offsets=offsets
        self.kind=kind
    def type_index(self):
        return self.kind


def fixture(body_count=20):
    points=[];faces=[];parts={}
    for material,count,radius,low,high in [(0,20,0.4,0.1,1.0),(1,body_count,0.25,-1,0.05)]:
        start=len(points)
        for y in (low,high):
            points += [(math.cos(i*2*math.pi/count)*radius,y,math.sin(i*2*math.pi/count)*radius) for i in range(count)]
        part=[]
        for i in range(count):
            j=(i+1)%count
            part.extend([(start+i,start+j,start+count+j),(start+i,start+count+j,start+count+i)])
        parts[material]=part
        faces+=part
    return points,faces,parts


class NeckTests(unittest.TestCase):
    def test_unequal_bridge_uses_all_boundary_edges(self):
        from collections import Counter
        points,faces,parts=fixture(16)
        plan=bridge_plan(points,{0:parts[0]},{1:parts[1]},(0,0,0),1)
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan['triangles']),36)
        counts=Counter(tuple(sorted((a,b))) for face in plan['triangles'] for a,b in zip(face,face[1:]+face[:1]))
        n,m=20,16
        boundary={tuple(sorted((i,(i+1)%n))) for i in range(n)}|{tuple(sorted((n+i,n+(i+1)%m))) for i in range(m)}
        self.assertEqual({edge for edge,count in counts.items() if count==1},boundary)
        self.assertTrue(all(count in (1,2) for count in counts.values()))
        self.assertTrue(all(len(set(face))==3 for face in plan['triangles']))
        # Tilted rings can overlap in their global height bounds and still be separated.
        tilted=[(x,y+z*.2,z) for x,y,z in points]
        self.assertIsNotNone(bridge_plan(tilted,{0:parts[0]},{1:parts[1]},(0,0,0),1))
        crossing=[(x,y-.2 if i<40 else y,z) for i,(x,y,z) in enumerate(points)]
        self.assertIsNone(bridge_plan(crossing,{0:parts[0]},{1:parts[1]},(0,0,0),1))

    def test_bridge_preserves_inputs_and_copies_boundary_deformation(self):
        points,faces,parts=fixture(16)
        vertices=[]
        for i,co in enumerate(points):
            w=Weight();w.bones=[0 if i<40 else 1]
            vertices.append(types.SimpleNamespace(co=co,normal=(0,1,0),uv=(i*.001,.2),additional_uvs=[],weight=w))
        head_offset=Offset();head_offset.index=0;head_offset.offset=(.03,0,0)
        body_offset=Offset();body_offset.index=56;body_offset.offset=(0,.02,0)
        tint=Offset();tint.index=1
        model=types.SimpleNamespace(vertices=vertices,faces=list(faces),materials=[types.SimpleNamespace(name='head',vertex_count=len(parts[0])*3),types.SimpleNamespace(name='body',vertex_count=len(parts[1])*3)],morphs=[Morph([head_offset,body_offset]),Morph([tint],8)])
        original=[(v.co,tuple(influences(v))) for v in vertices]
        plan=bridge_plan(points,{0:parts[0]},{1:parts[1]},(0,0,0),1)
        pmx=types.SimpleNamespace(BoneWeight=Weight,VertexMorphOffset=Offset)
        report=fit_neck_pmx(pmx,model,{0},{1},(0,0,0),1)
        self.assertEqual(report['status'],'bridged')
        self.assertEqual(report['bridge_faces'],36)
        self.assertIn('skin',model.materials[-1].name)
        self.assertEqual(model.materials[-1].name_e,'neck_skin_bridge')
        self.assertIn('[MMDTransplant:SKIN]',model.materials[-1].comment)
        self.assertEqual(report['skin_bridge']['uv_space'],'PMX_TOP_LEFT')
        self.assertEqual(report['skin_bridge']['head_uvs'],[vertices[i].uv for i in plan['source_indices'][:20]])
        start=len(points)
        self.assertEqual(original,[(v.co,tuple(influences(v))) for v in vertices[:start]])
        offsets={o.index:o.offset for o in model.morphs[0].offsets}
        for i,source in enumerate(plan['source_indices']):
            self.assertEqual(model.vertices[start+i].co,model.vertices[source].co)
            self.assertEqual(tuple(influences(model.vertices[start+i])),tuple(influences(model.vertices[source])))
            self.assertEqual(offsets.get(start+i,(0,0,0)),offsets.get(source,(0,0,0)))
        self.assertEqual(sum(m.vertex_count for m in model.materials),len(model.faces)*3)
        self.assertEqual([o.index for o in model.morphs[1].offsets],[1,2])
        once=(len(model.vertices),len(model.faces))
        fit_neck_pmx(pmx,model,{0,2},{1},(0,0,0),1)
        self.assertEqual(once,(len(model.vertices),len(model.faces)))

    def test_derived_strip_texture_keeps_source_uv_and_position_morphs(self):
        from mmd_transplant.core import set_bridge_texture
        vertices=[types.SimpleNamespace(uv=(.2,.3)) for _ in range(8)]
        original=vertices[0].uv
        def offset(index):return types.SimpleNamespace(index=index,offset=(.1,.2,0,0))
        model=types.SimpleNamespace(vertices=vertices,textures=[],materials=[types.SimpleNamespace(texture=-1)],
              morphs=[Morph([offset(0),offset(6)],3),Morph([offset(6)],1)])
        meta={'skin_bridge':{'material':0,'head_vertices':[6],'body_vertices':[7]}}
        set_bridge_texture(types.SimpleNamespace(Texture=lambda:types.SimpleNamespace()),model,meta,'derived.png')
        self.assertEqual(model.vertices[0].uv,original)
        self.assertEqual(model.vertices[6].uv,(.5,.5/128))
        self.assertEqual(model.vertices[7].uv,(.5,1-.5/128))
        self.assertEqual([o.index for o in model.morphs[0].offsets],[0])
        self.assertEqual([o.index for o in model.morphs[1].offsets],[6])
        self.assertEqual(model.materials[0].texture,0)

    def test_detect_match_and_leave_detached_hair_alone(self):
        points,faces,parts=fixture()
        points += [(0.1,0.13,0),(0.12,0.13,0),(0.11,0.16,0)]
        hair=(len(points)-3,len(points)-2,len(points)-1)
        plan=fit_plan(points,{0:parts[0],2:[hair]},{1:parts[1]},(0,0,0),1)
        self.assertEqual(plan['rim_vertices'],20)
        self.assertFalse(set(hair)&set(plan['changes']))
        self.assertTrue(all(c['reference']>=40 for c in plan['changes'].values()))

    def test_reject_unequal_loops_and_unrelated_eye_holes(self):
        points,faces,parts=fixture(16)
        self.assertIsNone(fit_plan(points,{0:parts[0]},{1:parts[1]},(0,0,0),1))
        points,faces,parts=fixture()
        shifted=[(x+4,y,z) for x,y,z in points]
        self.assertIsNone(fit_plan(shifted,{0:parts[0]},{1:parts[1]},(0,0,0),1))

    def test_coordinates_weights_morphs_and_idempotence(self):
        points,faces,parts=fixture()
        vertices=[]
        for i,co in enumerate(points):
            w=Weight();w.bones=[0 if i<40 else 1]
            vertices.append(types.SimpleNamespace(co=co,normal=(0,1,0),weight=w))
        a=Offset();a.index=0;a.offset=(0.2,0.1,0)
        model=types.SimpleNamespace(vertices=vertices,faces=faces,materials=[types.SimpleNamespace(name='head',vertex_count=len(parts[0])*3),types.SimpleNamespace(name='body',vertex_count=len(parts[1])*3)],morphs=[Morph([a])])
        pmx=types.SimpleNamespace(BoneWeight=Weight,VertexMorphOffset=Offset)
        body_before=[v.co for v in vertices[40:]]
        r=fit_neck_pmx(pmx,model,{0},{1},(0,0,0),1)
        self.assertEqual(r['status'],'fitted')
        self.assertEqual(body_before,[v.co for v in vertices[40:]])
        for v in vertices[:20]:
            ref=min(vertices[40:],key=lambda b:math.dist(v.co,b.co))
            self.assertLess(math.dist(v.co,ref.co),1e-5)
            self.assertEqual(list(influences(v)),[(1,1)])
        self.assertFalse(any(o.index==0 for o in model.morphs[0].offsets))
        once=[v.co for v in vertices]
        fit_neck_pmx(pmx,model,{0},{1},(0,0,0),1)
        self.assertEqual(once,[v.co for v in vertices])


if __name__=='__main__':
    unittest.main()
