"""Apply the same rim fitting to an already edited Blender mesh."""
import bpy
from mathutils import Vector
from collections import Counter,defaultdict
from .neck import fit_plan


def fit_existing(mesh):
    arm = mesh.find_armature()
    if arm is None:
        raise ValueError("网格没有绑定骨架。")
    points = [tuple(v.co) for v in mesh.data.vertices]
    head_faces, body_faces = {}, {}
    for polygon in mesh.data.polygons:
        material = mesh.data.materials[polygon.material_index]
        if material is None:
            continue
        name = material.name
        faces = head_faces if name.startswith('头_') else body_faces if name.startswith('身_') else None
        if faces is not None:
            faces.setdefault(polygon.material_index, []).append(tuple(polygon.vertices))
    bones = {p.mmd_bone.name_j:p.bone for p in arm.pose.bones if hasattr(p,'mmd_bone') and p.mmd_bone.name_j}
    bones.update({b.name:b for b in arm.data.bones if b.name not in bones})
    if not all(name in bones for name in ('頭','左目','右目')):
        raise ValueError("找不到 頭 / 左目 / 右目 骨骼，无法可靠估计颈部范围。")
    transform = mesh.matrix_world.inverted() @ arm.matrix_world
    anchor = transform @ bones['頭'].head_local
    width = ((transform @ bones['左目'].head_local)-(transform @ bones['右目'].head_local)).length
    band_ids={i for material,faces in head_faces.items()
              if mesh.data.materials[material].name.startswith('头_颈部连接') for face in faces for i in face}
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
                           if mesh.data.materials[material].name.startswith('头_颈部连接')
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
