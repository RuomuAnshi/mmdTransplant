"""Synthetic collection-membership regression without Blender or assets."""
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest


class Objects:
    def __init__(self, collection):
        self.collection = collection
        self.items = {}

    def __iter__(self):
        return iter(self.items.values())

    def __contains__(self, name):
        return name in self.items

    def link(self, obj):
        if obj.name in self.items:
            raise RuntimeError('Duplicate membership')
        self.items[obj.name] = obj
        obj.users_collection.append(self.collection)

    def unlink(self, obj):
        del self.items[obj.name]
        obj.users_collection.remove(self.collection)


class Collection:
    def __init__(self, name):
        self.name = name
        self.objects = Objects(self)
        self.children = []

    @property
    def all_objects(self):
        return {o.name:o for c in [self]+self.children for o in c.objects}


def load_helper(fake_bpy):
    spec = importlib.util.spec_from_file_location('physics_helper', Path(__file__).resolve().parents[1]/'mmd_transplant/physics_blender.py')
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get('bpy')
    sys.modules['bpy'] = fake_bpy
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            del sys.modules['bpy']
        else:
            sys.modules['bpy'] = previous
    return module


def obj(name, body=False, joint=False):
    return NS(name=name,users_collection=[],rigid_body=NS(mass=.3,kinematic=False) if body else None,
              rigid_body_constraint=NS(disable_collisions=True) if joint else None)


class PhysicsTests(unittest.TestCase):
    def test_preview_reorganization_retains_physics_registration(self):
        source,preview,bodies,joints=(Collection(n) for n in ('source','preview','bodies','joints'))
        rigid,constraint,mesh=obj('rigid',body=True),obj('joint',joint=True),obj('mesh')
        for o in (rigid,constraint,mesh):source.objects.link(o)
        bodies.objects.link(rigid);joints.objects.link(constraint)
        scene=NS(rigidbody_world=NS(collection=bodies,constraints=joints))
        helper=load_helper(NS())
        result=helper.move_to_preview([rigid,constraint,mesh],preview,scene)
        self.assertEqual(result,{'bodies':1,'joints':1,'bodies_linked':0,'joints_linked':0})
        self.assertIn('rigid',bodies.all_objects)
        self.assertIn('joint',joints.all_objects)
        self.assertEqual(len(list(source.objects)),0)
        self.assertEqual(len(list(preview.objects)),3)
        helper.move_to_preview([rigid,constraint,mesh],preview,scene)
        self.assertEqual(rigid.rigid_body.mass,.3)
        self.assertTrue(constraint.rigid_body_constraint.disable_collisions)

    def test_repair_missing_membership_is_idempotent_and_handles_nested_world(self):
        preview,bodies,joints=(Collection(n) for n in ('preview','bodies','joints'))
        rigid,constraint=obj('rigid',body=True),obj('joint',joint=True)
        preview.objects.link(rigid);preview.objects.link(constraint)
        scene=NS(rigidbody_world=NS(collection=bodies,constraints=joints))
        helper=load_helper(NS())
        result=helper.repair_membership(preview.objects,scene)
        self.assertEqual((result['bodies_linked'],result['joints_linked']),(1,1))
        result=helper.repair_membership(preview.objects,scene)
        self.assertEqual((result['bodies_linked'],result['joints_linked']),(0,0))
        bodies.objects.unlink(rigid);bodies.children.append(preview)
        result=helper.repair_membership(preview.objects,scene)
        self.assertEqual(result['bodies_linked'],0)
        self.assertFalse(rigid.rigid_body.kinematic)

    def test_empty_objects_do_not_create_a_world(self):
        scene=NS(rigidbody_world=None)
        result=load_helper(NS()).repair_membership([obj('mesh')],scene)
        self.assertEqual(result['bodies'],0)
        self.assertIsNone(scene.rigidbody_world)


if __name__ == '__main__':
    unittest.main()
