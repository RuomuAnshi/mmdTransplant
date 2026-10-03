"""Keep MMD objects registered with Blender's rigid-body world."""
import bpy


def move_to_preview(objects, collection, scene):
    """Relocate scene organization while retaining simulation collections."""
    objects = list(objects)
    world = scene.rigidbody_world
    protected = {c for c in (world.collection, world.constraints) if c} if world else set()
    for obj in objects:
        for previous in list(obj.users_collection):
            if previous not in protected and previous != collection:
                previous.objects.unlink(obj)
        if obj.name not in collection.objects:
            collection.objects.link(obj)
    return repair_membership(objects, scene)


def repair_membership(objects, scene):
    """Restore missing membership without changing rigid or joint settings."""
    objects = list(objects)
    bodies = [o for o in objects if o.rigid_body is not None]
    joints = [o for o in objects if o.rigid_body_constraint is not None]
    if not bodies and not joints:
        return {'bodies': 0, 'joints': 0, 'bodies_linked': 0, 'joints_linked': 0}
    if scene.rigidbody_world is None:
        with bpy.context.temp_override(scene=scene):
            bpy.ops.rigidbody.world_add()
    world = scene.rigidbody_world
    added = []
    for prop, name, targets in (('collection', 'MMD 换头物理刚体', bodies),
                                 ('constraints', 'MMD 换头物理关节', joints)):
        collection = getattr(world, prop)
        if collection is None:
            collection = bpy.data.collections.new(name)
            scene.collection.children.link(collection)
            setattr(world, prop, collection)
        count = 0
        for obj in targets:
            if obj.name not in collection.all_objects:
                collection.objects.link(obj)
                count += 1
        added.append(count)
    return {'bodies': len(bodies), 'joints': len(joints),
            'bodies_linked': added[0], 'joints_linked': added[1]}
