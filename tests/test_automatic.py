"""One-click regression cases built entirely from synthetic PMX data."""
from copy import deepcopy
import math
from types import SimpleNamespace as NS
import unittest

from test_core import Model, Morph, fixture, core
from mmd_transplant.automatic import assess_result, transplant_automatic


def tube_model(overlap=.06):
    model = fixture()
    model.vertices = []
    model.faces = []
    model.materials = []
    for name, count, radius, low, high, bone in (
        ('body', 16, .15, 1.4, 2.03, 1),
        ('head', 20, .24, 2.03-overlap, 2.6, 2),
    ):
        start = len(model.vertices)
        for y in (low, high):
            for i in range(count):
                angle = i*2*math.pi/count
                model.vertices.append(NS(co=(math.cos(angle)*radius, y, math.sin(angle)*radius),
                    normal=(math.cos(angle), 0, math.sin(angle)), uv=(i/count, .5), additional_uvs=[],
                    weight=NS(type=0, bones=[bone], weights=[])))
        faces = []
        for i in range(count):
            j = (i+1)%count
            faces.extend(((start+i,start+j,start+count+j), (start+i,start+count+j,start+count+i)))
        model.faces.extend(faces)
        model.materials.append(NS(name=name, vertex_count=len(faces)*3, texture=-1,
            sphere_texture=-1, toon_texture=0, is_shared_toon_texture=True))
    model.morphs = [Morph('synthetic_expression', 1, [NS(index=32, offset=(.01,0,0))])]
    return model


class AutomaticTests(unittest.TestCase):
    def test_small_crossing_is_rebuilt_with_donor_rig_and_morphs_preserved(self):
        pmx = NS(Model=Model)
        head, body = tube_model(), tube_model()
        before = deepcopy([v.co for v in head.vertices])
        ordinary, raw = core.transplant(pmx, head, body)
        self.assertEqual(raw['neck_fit']['status'], 'skipped')
        result, report = transplant_automatic(pmx, head, body)
        self.assertEqual(report['neck_fit']['status'], 'bridged')
        self.assertTrue(report['automatic_alignment']['applied'])
        shift = report['automatic_alignment']['vertical_shift']
        for source, index in report['donor_vertex_map'].items():
            old = ordinary.vertices[raw['donor_vertex_map'][source]].co
            self.assertAlmostEqual(result.vertices[index].co[1]-old[1], shift)
        for source, index in report['body_vertex_map'].items():
            self.assertEqual(result.vertices[index].co, body.vertices[source].co)
        eye = report['head_bone_map'][3]
        self.assertAlmostEqual(result.bones[eye].location[1]-ordinary.bones[raw['head_bone_map'][3]].location[1], shift)
        self.assertEqual(result.bones[2].location, body.bones[2].location)
        self.assertEqual(result.morphs[0].offsets[0].offset, (.01,0,0))
        self.assertEqual([v.co for v in head.vertices], before)
        core.validate(result)

    def test_valid_join_is_not_repositioned_and_large_overlap_is_not_guessed(self):
        pmx = NS(Model=Model)
        head, body = tube_model(-.05), tube_model(-.05)
        result, report = transplant_automatic(pmx, head, body)
        self.assertEqual(report['neck_fit']['status'], 'bridged')
        self.assertFalse(report['automatic_alignment']['applied'])
        head, body = tube_model(.2), tube_model(.2)
        _, report = transplant_automatic(pmx, head, body)
        self.assertEqual(report['neck_fit']['status'], 'skipped')
        self.assertFalse(report['automatic_alignment']['applied'])

    def test_incomplete_neck_or_textures_are_reported_as_needing_review(self):
        _, report = transplant_automatic(NS(Model=Model), fixture(), fixture())
        quality = assess_result(report, {'status': 'skipped'})
        self.assertEqual(quality['status'], 'review')
        _, report = transplant_automatic(NS(Model=Model), tube_model(-.05), tube_model(-.05))
        self.assertEqual(assess_result(report, {'status': 'matched'})['status'], 'ready')
        quality = assess_result(report, {'status': 'skipped'}, ['synthetic_missing_texture'])
        self.assertEqual(quality['status'], 'review')
        self.assertEqual(len(quality['issues']), 2)


if __name__ == '__main__':
    unittest.main()
