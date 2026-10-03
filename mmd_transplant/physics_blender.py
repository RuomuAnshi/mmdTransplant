"""Keep MMD objects registered with Blender's rigid-body world."""
import bpy


def _has_root(obj):
    seen = set()
    while obj is not None and id(obj) not in seen:
        seen.add(id(obj))
        if getattr(obj, 'mmd_type', None) == 'ROOT':
            return True
        obj = getattr(obj, 'parent', None)
    return False


def _abandoned_body(obj):
    if getattr(obj, 'mmd_type', None) != 'RIGID_BODY' or _has_root(obj):
        return False
    relation = obj.constraints.get('mmd_tools_rigid_parent')
    return relation is not None and relation.target is None


def _abandoned_joint(obj):
    if getattr(obj, 'mmd_type', None) not in {'JOINT', 'TEMPORARY'} or _has_root(obj):
        return False
    constraint = obj.rigid_body_constraint
    if constraint is None:
        return False
    ends = (constraint.object1, constraint.object2)
    return all(o is None for o in ends) or any(o is not None and _abandoned_body(o) for o in ends)


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
    bodies = [o for o in objects if o.rigid_body is not None and not _abandoned_body(o)]
    joints = [o for o in objects if o.rigid_body_constraint is not None and not _abandoned_joint(o)]
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


def quarantine_orphans(scene):
    """Disconnect abandoned MMD physics, keeping its objects for inspection.

    Only rootless MMD bodies whose bone-parent target has disappeared and
    rootless MMD constraints referencing those bodies (or no bodies) qualify.
    Standalone Blender physics and intact MMD models are never quarantined.
    """
    world = scene.rigidbody_world
    if world is None:
        return {'bodies': 0, 'joints': 0}
    bodies = [o for o in list(world.collection.all_objects) if _abandoned_body(o)] if world.collection else []
    joints = [o for o in list(world.constraints.all_objects) if _abandoned_joint(o)] if world.constraints else []
    if not bodies and not joints:
        return {'bodies': 0, 'joints': 0}
    quarantine = bpy.data.collections.new('MMD 换头残留物理隔离')
    scene.collection.children.link(quarantine)
    quarantine.hide_viewport = True
    quarantine.hide_render = True
    def descendants(collection):
        if collection is None:
            return []
        return [collection] + [c for child in collection.children for c in descendants(child)]
    simulation = descendants(world.collection) + descendants(world.constraints)
    for obj in bodies + joints:
        quarantine.objects.link(obj)
        for collection in list(obj.users_collection):
            if collection in simulation:
                collection.objects.unlink(obj)
    return {'bodies': len(bodies), 'joints': len(joints)}
