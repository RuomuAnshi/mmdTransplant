"""Regression tests against synthetic PMX-like data, without Blender."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS, ModuleType
import unittest
from copy import deepcopy

package=ModuleType('mmd_transplant')
package.__path__=[str(Path(__file__).resolve().parents[1]/'mmd_transplant')]
sys.modules['mmd_transplant']=package
spec = importlib.util.spec_from_file_location('mmd_transplant.core', Path(__file__).resolve().parents[1] / 'mmd_transplant/core.py')
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)


class Model:
    def __init__(self):
        self.name = self.name_e = 'fixture'
        self.comment = self.comment_e = ''
        self.filepath = '/test/source.pmx'
        for attr in ('vertices', 'faces', 'textures', 'materials', 'bones', 'morphs', 'display', 'rigids', 'joints'):
            setattr(self, attr, [])


def fixture():
    m = Model()
    for name, parent, co in [('root', -1, (0,0,0)), ('首',0,(0,1,0)), ('頭',1,(0,2,0)), ('左目',2,(0.3,2.2,0)), ('右目',2,(-0.3,2.2,0))]:
        m.bones.append(NS(name=name,name_e=name,parent=parent,location=co,displayConnection=(0,0.1,0),additionalTransform=None,isIK=False,hasAdditionalRotate=False,hasAdditionalLocation=False))
    for bone in (0,0,0,2,3,4):
        m.vertices.append(NS(co=(bone*0.1,2 if bone else 0,0),weight=NS(type=0,bones=[bone],weights=[]),additional_uvs=[]))
    m.faces = [(0,1,2),(3,4,5)]
    m.materials = [NS(name='body',vertex_count=3,texture=-1,sphere_texture=-1,toon_texture=0,is_shared_toon_texture=True),NS(name='head',vertex_count=3,texture=-1,sphere_texture=-1,toon_texture=0,is_shared_toon_texture=True)]
    return m


def eye_fixture():
    """Synthetic eye surfaces; rotation pivots can move independently."""
    model = fixture()
    for bone, center in ((3, 0.5), (4, -0.5)):
        for x in (-0.1, -0.05, 0, 0.05, 0.1):
            for y in (-0.1, -0.05, 0, 0.05, 0.1):
                model.vertices.append(NS(co=(center+x, 2.2+y, 0),
                    weight=NS(type=0,bones=[bone],weights=[]), additional_uvs=[]))
    return model


class Morph:
    def __init__(self, name, kind, offsets):
        self.name = name
        self.kind = kind
        self.offsets = offsets
    def type_index(self):
        return self.kind


class CoreTests(unittest.TestCase):
    def test_auto_scale_uses_eye_geometry_instead_of_misplaced_pivots(self):
        head, body = eye_fixture(), eye_fixture()
        body.bones[3].location = (0.12, 2.2, 0)
        body.bones[4].location = (-0.12, 2.2, 0)
        before = deepcopy([v.co for v in head.vertices])
        plan = core.analyze(head, body, core.Options(scale=1.2))
        self.assertAlmostEqual(plan['scale'], 1.2)
        self.assertEqual(plan['scale_estimate']['source'], 'eye_geometry')
        self.assertAlmostEqual(plan['scale_estimate']['bone_ratio'], 0.4)
        self.assertTrue(plan['warnings'])
        self.assertEqual([v.co for v in head.vertices], before)

    def test_eye_geometry_preserves_uniform_scale_and_ignores_uv_duplicates(self):
        head, body = eye_fixture(), eye_fixture()
        for vertex in head.vertices:
            vertex.co = tuple(x*2 for x in vertex.co)
        for bone in head.bones:
            bone.location = tuple(x*2 for x in bone.location)
        # Duplicate one eye's edge as UV splits, then add a stray weighted point.
        body.vertices.extend(deepcopy(body.vertices[6:11])*20)
        body.vertices.append(NS(co=(1000, 1000, 1000),
            weight=NS(type=0,bones=[3],weights=[]), additional_uvs=[]))
        plan = core.analyze(head, body, core.Options())
        self.assertAlmostEqual(plan['scale'], 0.5)
        self.assertEqual(plan['scale_estimate']['source'], 'eye_geometry')

    def test_eye_estimation_fallbacks_do_not_mix_measurement_types(self):
        head, body = eye_fixture(), fixture()
        estimate = core.estimate_scale(head, body)
        self.assertEqual(estimate['source'], 'unchanged')
        self.assertEqual(estimate['ratio'], 1)
        head, body = fixture(), fixture()
        self.assertEqual(core.estimate_scale(head, body)['source'], 'eye_bones')
        for bone in body.bones[3:]:
            bone.name = bone.name_e = 'unrecognized'
        self.assertEqual(core.estimate_scale(head, body)['source'], 'unchanged')

    def test_rejects_unilateral_or_implausible_eye_geometry(self):
        model = eye_fixture()
        for vertex in model.vertices[31:]:
            vertex.co = (vertex.co[0], vertex.co[1]+2, vertex.co[2])
        self.assertIsNone(core.eye_measurement(model)['geometry_span'])
        model = eye_fixture()
        for vertex in model.vertices[6:]:
            vertex.co = (vertex.co[0]*20, vertex.co[1], vertex.co[2])
        # Here the eyes remain bilateral but the surfaces collapse vertically
        # relative to their separation, so they cannot provide a reliable size.
        self.assertIsNone(core.eye_measurement(model)['geometry_span'])

    def test_preserves_body_and_maps_donor(self):
        head, body = fixture(), fixture()
        result, report = core.transplant(NS(Model=Model),head,body)
        self.assertEqual(len(result.faces),2)
        self.assertEqual(result.vertices[0].co,body.vertices[0].co)
        self.assertEqual(len(result.bones),5)
        self.assertEqual(result.bones[2].parent,1)
        self.assertEqual(result.bones[3].parent,2)
        self.assertEqual(report['donor_faces'],1)
        self.assertEqual(head.vertices[3].weight.bones,[2])

    def test_group_forward_refs_and_all_material_morph_scope(self):
        head, body = fixture(), fixture()
        head.morphs = [Morph('combo',0,[NS(morph=1,factor=1.0)]),Morph('blink',1,[NS(index=4,offset=(0,0.1,0))]),Morph('tint',8,[NS(index=-1)])]
        result,_ = core.transplant(NS(Model=Model),head,body)
        tint = next(m for m in result.morphs if m.name=='tint')
        self.assertEqual([x.index for x in tint.offsets],[1])
        combo = next(m for m in result.morphs if m.name=='combo')
        self.assertEqual(result.morphs[combo.offsets[0].morph].name,'blink')

    def test_scale_translation_applies_to_morphs_and_sdef(self):
        head, body = fixture(), fixture()
        head.vertices[3].weight = NS(type=3,bones=[2,1],weights=NS(weight=0.8,c=(0,2,0),r0=(0,2,0),r1=(0,2,0)))
        head.morphs=[Morph('move',1,[NS(index=3,offset=(0,0.1,0))])]
        result, report=core.transplant(NS(Model=Model),head,body,core.Options(auto_scale=False,scale=2,offset=(1,0,0)))
        v=result.vertices[report['donor_vertex_map'][3]]
        self.assertEqual(v.weight.weights.c,(1,2,0))
        self.assertEqual(result.morphs[0].offsets[0].offset,(0,0.2,0))

    def test_invalid_head_and_cycles_fail_clearly(self):
        with self.assertRaises(core.TransplantError):
            core.analyze(fixture(),fixture(),core.Options(head_bone='missing'))
        model=fixture()
        model.bones[0].parent=2
        with self.assertRaises(core.TransplantError):
            core.validate(model)

    def test_material_override_and_empty_group_pruning(self):
        head,body=fixture(),fixture()
        head.morphs=[Morph('empty',0,[NS(morph=1,factor=1)]),Morph('body-only',1,[NS(index=0,offset=(1,0,0))])]
        result,_=core.transplant(NS(Model=Model),head,body)
        self.assertFalse(result.morphs)
        plan=core.analyze(head,body,core.Options(head_materials={0:'INCLUDE'}))
        self.assertTrue(all(plan['head_faces']))


if __name__ == '__main__':
    unittest.main()
