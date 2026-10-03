"""Apply the same rim fitting to an already edited Blender mesh."""
import bpy
from mathutils import Vector
from collections import Counter,defaultdict
from .neck import fit_plan, bridge_plan, find_rim, distance
from .skin_blender import material_name, is_bridge


def _neck_context(mesh):
    arm = mesh.find_armature()
    if arm is None:
        raise ValueError("网格没有绑定骨架。")
    points = [tuple(v.co) for v in mesh.data.vertices]
    head_faces, body_faces = {}, {}
    for polygon in mesh.data.polygons:
        material = mesh.material_slots[polygon.material_index].material
        if material is None:
            continue
        name = material_name(material)
        faces = head_faces if name.startswith('头_') else body_faces if name.startswith('身_') else None
        if faces is not None:
            faces.setdefault(polygon.material_index, []).append(tuple(polygon.vertices))
    bones = {p.mmd_bone.name_j:p.bone for p in arm.pose.bones if hasattr(p,'mmd_bone') and p.mmd_bone.name_j}
    bones.update({b.name:b for b in arm.data.bones if b.name not in bones})
    def control(aliases):
        return next((bones[name] for name in aliases if name in bones), None)
    head = control(('頭','head','头','頭部'))
    left = control(('左目','eye_l','left eye'))
    right = control(('右目','eye_r','right eye'))
    if not all((head,left,right)):
        raise ValueError("找不到头部和左右眼控制骨骼，无法可靠估计颈部范围。")
    transform = mesh.matrix_world.inverted() @ arm.matrix_world
    anchor = transform @ head.head_local
    width = ((transform @ left.head_local)-(transform @ right.head_local)).length
    if width <= 1e-6:
        raise ValueError('眼部定位距离无效，无法可靠估计颈部范围。')
    return arm, points, head_faces, body_faces, anchor, width


def fit_existing(mesh):
    arm, points, head_faces, body_faces, anchor, width = _neck_context(mesh)
    band_ids={i for material,faces in head_faces.items()
              if is_bridge(mesh.material_slots[material].material) for face in faces for i in face}
    if band_ids:
        # The band uses independent UV vertices. Verify the original seam
        # counterparts rather than treating its top loop as an unfitted donor.
        outside={i for faces in list(head_faces.values())+list(body_faces.values()) for face in faces for i in face}-band_ids
        lookup=defaultdict(list)
        for index in outside:lookup[tuple(round(x,6) for x in points[index])].append(index)
        keys=list(mesh.data.shape_keys.key_blocks) if mesh.data.shape_keys else []
        geometric={i:tuple(round(x,6) for x in points[i]) for i in band_ids|outside}
        band_points={geometric[i] for i in band_ids}
        band_edges=Counter(tuple(sorted((geometric[a],geometric[b]))) for material,faces in head_faces.items()
                           if is_bridge(mesh.material_slots[material].material)
                           for face in faces for a,b in zip(face,face[1:]+face[:1]))
        source_edges={tuple(sorted((geometric[a],geometric[b])))
                      for faces in list(head_faces.values())+list(body_faces.values()) for face in faces
                      for a,b in zip(face,face[1:]+face[:1])
                      if a in outside and b in outside and geometric[a] in band_points and geometric[b] in band_points}
        intact=all(count in (1,2) and (count!=1 or edge in source_edges) for edge,count in band_edges.items())
        def weights(index):
            return {g.group:g.weight for g in mesh.data.vertices[index].groups
                    if mesh.vertex_groups[g.group].name in arm.data.bones}
        def coincides(a,b):
            wa,wb=weights(a),weights(b)
            return (all(abs(wa.get(g,0)-wb.get(g,0))<1e-5 for g in wa.keys()|wb.keys())
                    and all((key.data[a].co-key.data[b].co).length<width*1e-4 for key in keys))
        if intact and all(any(coincides(index,other) for other in lookup[tuple(round(x,6) for x in points[index])]) for index in band_ids):
            return {'status':'already_bridged','affected_vertices':0,'rim_vertices':len(band_ids)}
        raise ValueError("现有颈部连接面边界已被编辑；请先保存手动修改，再重新生成预览或手动修复边界。")
    plan = fit_plan(points, head_faces, body_faces, tuple(anchor), width, up=2)
    if plan is None:
        raise ValueError("颈圈点数不同或边界不可靠。不同点数的自动桥接需要重新生成预览；请先保存当前手动编辑。")
    result = {'status':'fitted','rim_vertices':plan['rim_vertices'],'affected_vertices':len(plan['changes']),
              'max_displacement':plan['max_delta'],'head_material':mesh.data.materials[plan['head_rim']['material']].name,
              'body_material':mesh.data.materials[plan['body_rim']['material']].name}
    if plan['max_delta'] <= width*1e-4:
        result['status']='already_fitted'
        result['affected_vertices']=0
        return result
    changes = plan['changes']
    # Refuse a bound SDEF driver; rebinding without respecting its cache is unsafe.
    if mesh.data.shape_keys and any(k.name.startswith('mmd_sdef') for k in mesh.data.shape_keys.key_blocks):
        raise ValueError("此网格含 SDEF 辅助形态键；请通过生成预览的 PMX 流程自动贴合，避免修改已绑定的 SDEF 缓存。")
    groups = {g.index:g for g in mesh.vertex_groups if g.name in arm.data.bones}
    original_weights = {i:{g.group:g.weight for g in mesh.data.vertices[i].groups if g.group in groups}
                        for i in set(changes)|{c['reference'] for c in changes.values()}}
    if any(not original_weights[i] for i in original_weights):
        raise ValueError("颈部附近顶点缺少有效骨骼权重，请先修复权重。")
    # Save affected coordinates first: Blender shape keys and mesh coordinates
    # may share Basis storage, so no reads may depend on partially updated data.
    keys = list(mesh.data.shape_keys.key_blocks) if mesh.data.shape_keys else []
    snapshot = {key.name:{i:key.data[i].co.copy() for i in original_weights} for key in keys}
    basis = snapshot[keys[0].name] if keys else {}
    old_normals = [loop.normal.copy() for loop in mesh.data.loops] if mesh.data.has_custom_normals else None
    for index, change in changes.items():
        delta = Vector(change['delta']);t=change['weight_factor'];reference=change['reference']
        mesh.data.vertices[index].co = Vector(points[index])+delta
        for key in keys:
            own = snapshot[key.name][index]
            own_offset = own-basis[index]
            body_offset = snapshot[key.name][reference]-basis[reference]
            key.data[index].co = basis[index]+delta+own_offset*(1-t)+body_offset*t
        weights = {g:w*(1-t) for g,w in original_weights[index].items()}
        for group,value in original_weights[reference].items():
            weights[group] = weights.get(group,0)+value*t
        # Match the PMX path's maximum of four bone influences.
        ordered=sorted(((g,w) for g,w in weights.items() if w>1e-8),key=lambda x:-x[1])[:4]
        total=sum(w for g,w in ordered)
        for group in original_weights[index]:
            groups[group].remove([index])
        for group,value in ordered:
            groups[group].add([index],value/total,'REPLACE')
    mesh.data.update()
    if old_normals is not None:
        # Recompute only fitted-region custom normals; preserve anime shading elsewhere.
        mesh.data.normals_split_custom_set([((0,0,0) if loop.vertex_index in changes else old_normals[loop.index])
                                            for loop in mesh.data.loops])
        mesh.data.update()
    bpy.context.view_layer.update()
    return result


def repair_existing(mesh):
    """Bridge a credible neck gap in a copy, retaining edited source data.

    New boundary vertices copy their source weights and every shape key.
    Broken generated strips can be replaced; unrelated holes are untouched.
    """
    import bmesh
    arm, points, head_faces, body_faces, anchor, width = _neck_context(mesh)
    bands = {i for i in head_faces if is_bridge(mesh.material_slots[i].material)}
    if bands:
        try:
            intact = fit_existing(mesh)
            if intact['status'] == 'already_bridged':
                return intact
        except ValueError:
            pass
    original_head = {i:faces for i,faces in head_faces.items() if i not in bands}
    head = find_rim(points, original_head, tuple(anchor), width, up=2)
    body = find_rim(points, body_faces, tuple(anchor), width, up=2)
    if head is None or body is None:
        missing = '头侧' if head is None else '身体侧'
        raise ValueError(f'未找到完整可信的{missing}颈部边界；边界可能破损、材质来源标记丢失或超出搜索范围，需要手动整理边界。')
    plan = bridge_plan(points, original_head, body_faces, tuple(anchor), width, up=2,
                       head_rim=head, body_rim=body, allow_equal=True)
    if plan is None:
        if not bands and len(head['points']) == len(body['points']):
            return fit_existing(mesh)
        raise ValueError('颈部边界交叉、重叠或距离过大，不能安全补面；请先调整头部位置或重新一键生成。')
    if mesh.data.shape_keys and any(k.name.startswith('mmd_sdef') for k in mesh.data.shape_keys.key_blocks):
        raise ValueError('网格含已绑定的 SDEF 缓存，不能安全改变顶点数量；请先解除 SDEF 绑定或重新生成。')
    bone_groups = {g.index for g in mesh.vertex_groups if g.name in arm.data.bones}
    sources = plan['source_indices']
    if any(not any(g.group in bone_groups and g.weight>1e-8 for g in mesh.data.vertices[i].groups) for i in sources):
        raise ValueError('颈部边界缺少有效骨骼权重，请先修复权重。')
    candidate = mesh.copy()
    candidate.data = mesh.data.copy()
    material = mesh.material_slots[body['material']].material.copy()
    material.name = '头_颈部连接_skin'
    material['mmd_transplant_surface'] = 'SKIN_BRIDGE'
    if 'zd_original_name' in material:
        material['zd_original_name'] = material.name
    if hasattr(material, 'mmd_material'):
        material.mmd_material.name_j = material.name
        material.mmd_material.name_e = 'neck_skin_bridge'
    bm = bmesh.new()
    committed = False
    try:
        bm.from_mesh(candidate.data)
        bm.verts.ensure_lookup_table();bm.faces.ensure_lookup_table()
        originals = [bm.verts[i] for i in sources]
        normals = {loop:tuple(mesh.data.loops[index].normal)
                   for face, polygon in zip(bm.faces, mesh.data.polygons)
                   for loop, index in zip(face.loops, polygon.loop_indices)} if mesh.data.has_custom_normals else None
        if bands:
            bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.material_index in bands], context='FACES')
        slot = len(candidate.data.materials)
        candidate.data.materials.append(material)
        deform = bm.verts.layers.deform.verify()
        new = []
        for source in originals:
            vertex = bm.verts.new(source.co)
            # Copy shape layers explicitly. copy_from also copies internal
            # shape-key indices and can invalidate a newly created BMVert.
            for kind in ('shape','float','int','string','float_vector','color','float_color'):
                access = getattr(bm.verts.layers,kind,None)
                if access is not None:
                    for layer in access.values():
                        vertex[layer] = source[layer]
            for group, weight in source[deform].items():
                if group in bone_groups:
                    vertex[deform][group] = weight
            new.append(vertex)
        n = len(plan['head_rim']['points'])
        # The strip initially uses the body's atlas, while all original UVs
        # remain unchanged. Optional skin matching can give it its own atlas.
        uv_refs = {}
        for layer in bm.loops.layers.uv.values():
            refs = {}
            for face in bm.faces:
                if face.material_index == body['material']:
                    for loop in face.loops:
                        refs.setdefault(loop.vert, loop[layer].uv.copy())
            uv_refs[layer] = refs
        for triangle in plan['triangles']:
            face = bm.faces.new([new[i] for i in triangle])
            face.material_index = slot
            face.smooth = True
            for loop, index in zip(face.loops, triangle):
                reference = index if index >= n else n+min(range(len(new)-n), key=lambda j:distance(points[sources[index]],points[sources[n+j]]))
                for layer, refs in uv_refs.items():
                    if originals[reference] in refs:
                        loop[layer].uv = refs[originals[reference]]
        custom = [normals.get(loop,(0,0,0)) for face in bm.faces for loop in face.loops] if normals is not None else None
        bm.to_mesh(candidate.data)
        candidate.data.update()
        if custom is not None:
            candidate.data.normals_split_custom_set(custom)
        # Commit only after geometry creation has succeeded.
        mesh.data = candidate.data
        committed = True
        bpy.context.view_layer.update()
        return {'status':'rebridged' if bands else 'bridged','bridge_faces':len(plan['triangles']),
                'bridge_vertices':len(sources),'rim_vertices':len(sources),'material':slot}
    finally:
        bm.free()
        data = candidate.data
        bpy.data.objects.remove(candidate,do_unlink=True)
        if not committed and data.users == 0:
            bpy.data.meshes.remove(data)
        if not committed and material.users == 0:
            bpy.data.materials.remove(material)
