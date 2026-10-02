"""Synthetic neck-plane clipping tests with no external assets or Blender."""
from collections import Counter, defaultdict
from copy import deepcopy
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import sys
from types import ModuleType


spec = importlib.util.spec_from_file_location(
    "neck_clip", Path(__file__).resolve().parents[1] / "mmd_transplant/neck_clip.py")
clip = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clip)
PMX = NS(BoneWeight=lambda: NS(), VertexMorphOffset=lambda: NS(), UVMorphOffset=lambda: NS())


def integration_fixture(donor):
    model = NS(name="fixture", name_e="fixture", comment="", comment_e="", filepath="fixture.pmx",
               vertices=[], faces=[], materials=[], textures=[], bones=[], morphs=[], display=[], rigids=[], joints=[])
    for name, parent, position in (("root", -1, (0, -1, 0)), ("首", 0, (0, 0, 0)),
                                   ("頭", 1, (0, 1, 0)), ("左目", 2, (0.5, 1.5, 0)),
                                   ("右目", 2, (-0.5, 1.5, 0))):
        model.bones.append(NS(name=name, name_e=name, parent=parent, location=position,
                              displayConnection=(0, 0.1, 0), additionalTransform=None, isIK=False,
                              hasAdditionalRotate=False, hasAdditionalLocation=False))
    if donor:
        model.bones.append(NS(name="aux", name_e="aux", parent=0, location=(0, 0, 0),
                              displayConnection=(0, 0.1, 0), additionalTransform=None, isIK=False,
                              hasAdditionalRotate=False, hasAdditionalLocation=False))
    count = 8 if donor else 10
    low, high, radius = (0.6, 1.3, 0.3) if donor else (-0.5, 0.8, 0.28)
    for upper in (False, True):
        for i in range(count):
            angle = i * math.tau / count
            weight = (NS(type=1, bones=[2, 5], weights=[0.9 if upper else 0.1]) if donor
                      else NS(type=0, bones=[1], weights=[]))
            model.vertices.append(NS(co=(math.cos(angle)*radius,
                (high if upper else low)+math.sin(angle)*(0.15 if donor else 0.08), math.sin(angle)*radius),
                normal=(math.cos(angle), 0, math.sin(angle)), uv=(i/count, int(upper)),
                additional_uvs=[], edge_scale=1, weight=weight))
    for i in range(count):
        j = (i+1) % count
        model.faces.extend(((i, count+j, j), (i, count+i, count+j)))
    attributes = dict(texture=-1, sphere_texture=-1, toon_texture=0, is_shared_toon_texture=True)
    model.materials.append(NS(name="tube" if donor else "surface", vertex_count=len(model.faces)*3, **attributes))
    if donor:
        apex = len(model.vertices)
        model.vertices.append(NS(co=(0, 1.8, 0), normal=(0, 1, 0), uv=(0.5, 1),
            additional_uvs=[], edge_scale=1, weight=NS(type=0, bones=[2], weights=[])))
        model.faces.extend((count+i, count+(i+1)%count, apex) for i in range(count))
    else:
        apex = len(model.vertices)
        model.vertices.extend(NS(co=co, normal=(0, 1, 0), uv=(0, 0), additional_uvs=[],
            edge_scale=1, weight=NS(type=0, bones=[2], weights=[]))
            for co in ((0, 1.5, 0), (0.1, 1.5, 0), (0, 1.6, 0.1)))
        model.faces.append((apex, apex+1, apex+2))
    model.materials.append(NS(name="upper", vertex_count=(count if donor else 1)*3, **attributes))
    motion = Morph(1, [NS(index=i, offset=(0, 0.01+i*0.001, 0)) for i in range(count*2)])
    motion.name = "motion"
    model.morphs.append(motion)
    return model


class Morph:
    def __init__(self, kind, offsets):
        self.kind = kind
        self.offsets = offsets

    def type_index(self):
        return self.kind


def fixture():
    vertices = [NS(co=co, normal=(0, 1, 0), uv=(0, 0), additional_uvs=[], edge_scale=1,
                   weight=NS(type=0, bones=[0], weights=[]))
                for co in ((4, 0, 0), (5, 0, 0), (4, 0, 1))]
    for high in (False, True):
        for i in range(8):
            angle = i * math.tau / 8
            weight = (NS(type=3, bones=[1, 2], weights=NS(weight=0.3)) if high
                      else NS(type=1, bones=[0, 1], weights=[0.6]))
            vertices.append(NS(
                co=(math.cos(angle), (1 if high else -1) + 0.2 * math.sin(angle), math.sin(angle)),
                normal=(math.cos(angle), 0, math.sin(angle)), uv=(i / 8, int(high)),
                additional_uvs=[(i / 8, int(high), 0, 1)] + ([(1, 0.5, 0, 1)] if high else []),
                edge_scale=3 if high else 1, weight=weight))
    faces = [(0, 1, 2)]
    for i in range(8):
        a, b, c, d = 3 + i, 3 + (i + 1) % 8, 11 + (i + 1) % 8, 11 + i
        faces.extend(((a, c, b), (a, d, c)))
    faces.append((2, 1, 0))
    morphs = [Morph(1, [NS(index=i, offset=(i * 0.01, 0.02, -0.03))
                       for i in range(3, 19, 2)])]
    for kind in range(3, 8):
        morphs.append(Morph(kind, [NS(index=i, offset=(i * 0.02, 0.1, -0.2, 0.3))
                                  for i in range(4, 19, 3)]))
    morphs.extend((Morph(0, [NS(morph=0, factor=0.5)]), Morph(8, [NS(index=1, amount=0.5)])))
    return NS(vertices=vertices, faces=faces,
              materials=[NS(vertex_count=3), NS(vertex_count=48), NS(vertex_count=3)], morphs=morphs)


def weight_values(vertex):
    return {bone: value for bone, value in clip._influences(vertex.weight) if bone >= 0 and value > 0}


class ClipTests(unittest.TestCase):
    def test_unsafe_complete_tube_does_not_fall_back_to_jaw_bridge(self):
        if "mmd_transplant" not in sys.modules:
            package = ModuleType("mmd_transplant")
            package.__path__ = [str(Path(__file__).resolve().parents[1]/"mmd_transplant")]
            sys.modules["mmd_transplant"] = package
        from mmd_transplant import core
        head, body = integration_fixture(True), integration_fixture(False)
        for vertex in head.vertices[:8]:
            vertex.co = (vertex.co[0],vertex.co[1]-0.3,vertex.co[2])
        def empty_model():
            return NS(**{attr: [] for attr in ("vertices", "faces", "materials", "textures", "bones",
                                               "morphs", "display", "rigids", "joints")})
        result, report = core.transplant(NS(**vars(PMX), Model=empty_model),head,body)
        self.assertEqual(report['neck_tube_materials'],[0])
        self.assertEqual(report['neck_fit']['status'],'skipped')
        self.assertFalse(any(m.name == '头_颈部连接' for m in result.materials))
        core.validate(result)

    def test_transplant_recovers_complete_tube_and_bridges_overlap_automatically(self):
        if "mmd_transplant" not in sys.modules:
            package = ModuleType("mmd_transplant")
            package.__path__ = [str(Path(__file__).resolve().parents[1]/"mmd_transplant")]
            sys.modules["mmd_transplant"] = package
        from mmd_transplant import core
        head, body = integration_fixture(True), integration_fixture(False)
        original_head, original_body = deepcopy(head.vertices), deepcopy(body.vertices)
        def empty_model():
            return NS(**{attr: [] for attr in ("vertices", "faces", "materials", "textures", "bones",
                                               "morphs", "display", "rigids", "joints")})
        pmx = NS(**vars(PMX), Model=empty_model)
        result, report = core.transplant(pmx, head, body)
        self.assertEqual(report["neck_tube_materials"], [0])
        self.assertEqual(report["neck_fit"]["status"], "bridged")
        self.assertTrue(report["neck_fit"]["trimmed_neck"])
        self.assertGreater(report["neck_fit"]["interpolated_vertices"], 0)
        self.assertEqual(report["head_bone_map"][5], report["body_bone_map"][1])
        self.assertEqual(head.vertices, original_head)
        self.assertEqual(body.vertices, original_body)
        core.validate(result)
        band_count = report["neck_fit"]["bridge_vertices"]
        band_start = len(result.vertices)-band_count
        n = report["neck_fit"]["head_rim_vertices"]
        band = result.vertices[band_start:]
        self.assertGreater(min(v.co[1] for v in band[:n]), max(v.co[1] for v in band[n:]))
        offsets = {o.index: o.offset for o in result.morphs[0].offsets}
        for i, vertex in enumerate(band, band_start):
            source = next(j for j, other in enumerate(result.vertices[:band_start])
                          if other.co == vertex.co and weight_values(other) == weight_values(vertex))
            self.assertEqual(offsets.get(i, (0, 0, 0)), offsets.get(source, (0, 0, 0)))

    def test_tilted_tube_has_closed_plane_boundary_and_preserves_other_materials(self):
        for keep_above in (True, False):
            with self.subTest(keep_above=keep_above):
                model = fixture()
                before = deepcopy(model)
                report = clip.clip_material(PMX, model, 1, 0, keep_above)
                self.assertEqual(model.faces[0], before.faces[0])
                self.assertEqual(model.faces[-1], before.faces[-1])
                self.assertEqual(model.materials[0].vertex_count, 3)
                self.assertEqual(model.materials[2].vertex_count, 3)
                self.assertEqual(model.vertices[:19], before.vertices)
                self.assertEqual(len(report["created_indices"]), 16)
                target = model.faces[1:-1]
                self.assertTrue(all(clip._nondegenerate([model.vertices[i].co for i in face]) for face in target))
                self.assertTrue(all(model.vertices[i].co[1] * (1 if keep_above else -1) >= -1e-9
                                    for face in target for i in face))
                edges = Counter(tuple(sorted((a, b))) for face in target
                                for a, b in zip(face, face[1:] + face[:1]))
                plane_edges = [edge for edge, count in edges.items() if count == 1
                               and all(abs(model.vertices[i].co[1]) < 1e-9 for i in edge)]
                adjacency = defaultdict(set)
                for a, b in plane_edges:
                    adjacency[a].add(b)
                    adjacency[b].add(a)
                self.assertEqual(len(plane_edges), 16)
                self.assertTrue(all(len(neighbors) == 2 for neighbors in adjacency.values()))
                self.assertEqual(set(adjacency), set(report["created_indices"]))
                self.assertTrue(all(count in (1, 2) for count in edges.values()))
                seen, stack = set(), [next(iter(adjacency))]
                while stack:
                    current = stack.pop()
                    if current not in seen:
                        seen.add(current)
                        stack.extend(adjacency[current] - seen)
                self.assertEqual(seen, set(adjacency))
                for triangle in target:
                    support = set()
                    for index in triangle:
                        support.update(report["sources"][index]["edge"]
                                       if index in report["sources"] else (index,))
                    original = next(face for face in before.faces[1:-1] if support <= set(face))
                    def cross(face, vertices):
                        a, b, c = [vertices[i].co for i in face]
                        u = [b[i] - a[i] for i in range(3)]
                        v = [c[i] - a[i] for i in range(3)]
                        return (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
                    self.assertGreater(sum(a*b for a, b in zip(cross(triangle, model.vertices),
                                                              cross(original, before.vertices))), 0)

    def test_intersections_interpolate_weights_attributes_and_morph_offsets(self):
        model = fixture()
        before = deepcopy(model)
        report = clip.clip_material(PMX, model, 1, 0, True)
        for index, source in report["sources"].items():
            a, b = source["edge"]
            t = source["t"]
            vertex = model.vertices[index]
            expected = {}
            for v, factor in ((before.vertices[a], 1 - t), (before.vertices[b], t)):
                for bone, value in weight_values(v).items():
                    expected[bone] = expected.get(bone, 0) + value * factor
            for bone, value in expected.items():
                self.assertAlmostEqual(weight_values(vertex)[bone], value)
            self.assertAlmostEqual(sum(weight_values(vertex).values()), 1)
            self.assertEqual(vertex.weight.type, 2)
            self.assertAlmostEqual(sum(n * n for n in vertex.normal), 1)
            for actual, x, y in zip(vertex.uv, before.vertices[a].uv, before.vertices[b].uv):
                self.assertAlmostEqual(actual, x+(y-x)*t)
            self.assertAlmostEqual(vertex.edge_scale,
                                   before.vertices[a].edge_scale * (1 - t) + before.vertices[b].edge_scale * t)
            self.assertEqual(len(vertex.additional_uvs), 2)
            for channel in range(2):
                source_a = before.vertices[a].additional_uvs
                source_b = before.vertices[b].additional_uvs
                values_a = source_a[channel] if channel < len(source_a) else (0, 0, 0, 0)
                values_b = source_b[channel] if channel < len(source_b) else (0, 0, 0, 0)
                for actual, x, y in zip(vertex.additional_uvs[channel], values_a, values_b):
                    self.assertAlmostEqual(actual, x+(y-x)*t)
            for morph, original in zip(model.morphs, before.morphs):
                kind = morph.type_index()
                if kind != 1 and not 3 <= kind <= 7:
                    self.assertEqual(morph.offsets, original.offsets)
                    continue
                length = 3 if kind == 1 else 4
                offsets = {o.index: o.offset for o in original.offsets}
                actual = {o.index: o.offset for o in morph.offsets}.get(index, (0,) * length)
                for value, x, y in zip(actual, offsets.get(a, (0,)*length), offsets.get(b, (0,)*length)):
                    self.assertAlmostEqual(value, x+(y-x)*t)
        self.assertEqual(model.vertices[:19], before.vertices)

    def test_repeated_clip_has_no_new_vertices_faces_or_morph_offsets(self):
        model = fixture()
        clip.clip_material(PMX, model, 1, 0, True)
        before = (len(model.vertices), deepcopy(model.faces), [len(m.offsets) for m in model.morphs])
        report = clip.clip_material(PMX, model, 1, 0, True)
        self.assertEqual(report["created_indices"], [])
        self.assertEqual((len(model.vertices), model.faces, [len(m.offsets) for m in model.morphs]), before)

    def test_failure_is_transactional(self):
        model = fixture()
        model.morphs[0].offsets[0].offset = (1, 2)
        before = deepcopy(model)
        with self.assertRaises(ValueError):
            clip.clip_material(PMX, model, 1, 0, True)
        self.assertEqual(model.vertices, before.vertices)
        self.assertEqual(model.faces, before.faces)
        self.assertEqual(model.materials, before.materials)
        self.assertEqual([m.offsets for m in model.morphs], [m.offsets for m in before.morphs])


if __name__ == "__main__":
    unittest.main()
