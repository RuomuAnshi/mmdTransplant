"""PMX transplant engine. No bpy dependency; inputs are never mutated."""
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
import shutil
from statistics import median
from .neck import fit_plan, bridge_plan, find_rim


class TransplantError(ValueError):
    pass


@dataclass
class Options:
    head_bone: str = ""
    body_head_bone: str = ""
    threshold: float = 0.5
    auto_scale: bool = True
    scale: float = 1.0
    offset: tuple = (0.0, 0.0, 0.0)  # PMX coordinates: X right, Y up, Z back
    head_materials: dict = field(default_factory=dict)
    body_materials: dict = field(default_factory=dict)
    extra_head_bones: tuple = ()
    physics: bool = True
    fit_neck: bool = True


def bone_index(model, override="", aliases=("頭", "head", "头", "頭部")):
    for alias in ((override,) if override else aliases):
        for i, bone in enumerate(model.bones):
            if alias.casefold() in (bone.name.casefold(), bone.name_e.casefold()):
                return i
    raise TransplantError(f"模型 {model.name} 找不到头骨骼，请填写实际骨骼名。")


def descendants(model, roots):
    result = set(roots)
    children = {}
    for i, bone in enumerate(model.bones):
        children.setdefault(bone.parent, []).append(i)
    stack = list(roots)
    while stack:
        for i in children.get(stack.pop(), ()):
            if i not in result:
                result.add(i)
                stack.append(i)
    return result


def influences(vertex):
    w = vertex.weight
    if w.type == 0:
        weights = (1.0,)
    elif w.type in (1, 3):
        value = w.weights.weight if w.type == 3 else w.weights[0]
        weights = (value, 1.0 - value)
    elif w.type == 2:
        weights = w.weights
    else:
        raise TransplantError(f"不支持权重类型 {w.type}；当前支持 PMX 2.0 BDEF/SDEF。")
    return zip(w.bones, weights)


def eye_measurement(model):
    """Measure actual weighted eye geometry, independently of control pivots.

    Eye pivots can sit anywhere an author needs them for rotation. Unique
    positions and medians avoid UV duplicates and dense highlights skewing
    the measured centers. Only accept a plausible bilateral pair.
    """
    try:
        indices = [bone_index(model, aliases=aliases) for aliases in
                   (("左目", "eye_l", "left eye"), ("右目", "eye_r", "right eye"))]
    except TransplantError:
        return {"bone_span": 0.0, "geometry_span": None}
    pivots = [model.bones[i].location for i in indices]
    bone_span = math.dist(*pivots)
    groups = [descendants(model, {i}) for i in indices]
    samples = [set(), set()]
    for vertex in model.vertices:
        weights = list(influences(vertex))
        point = tuple(float(x) for x in vertex.co)
        if not all(math.isfinite(x) for x in point):
            continue
        for group, points in zip(groups, samples):
            if sum(weight for bone, weight in weights if bone in group) > 0.5:
                points.add(point)
    result = {"bone_span": bone_span, "geometry_span": None}
    if min(map(len, samples)) < 8:
        return result
    centers = [tuple(median(p[k] for p in points) for k in range(3)) for points in samples]
    span = math.dist(*centers)
    if (span <= 1e-6 or abs(centers[0][0] - centers[1][0]) < span * 0.9
            or any(abs(centers[0][k] - centers[1][k]) > span * 0.25 for k in (1, 2))):
        return result
    # Samples must describe similarly sized eyes, not stray weighted geometry.
    sizes = []
    for points in samples:
        dimensions = []
        for k in range(3):
            values = sorted(p[k] for p in points)
            trim = max(1, len(values) // 10)
            dimensions.append(values[-trim-1] - values[trim])
        if (not span * 0.02 < dimensions[0] < span * 0.9
                or not span * 0.02 < dimensions[1] < span or dimensions[2] > span * 1.5):
            return result
        sizes.append(dimensions)
    if any(not 0.4 < a / b < 2.5 for a, b in zip(sizes[0][:2], sizes[1][:2])):
        return result
    result["geometry_span"] = span
    return result


def estimate_scale(head, body):
    """Compare like measurements; never mix a pivot with a geometry center."""
    measures = [eye_measurement(m) for m in (head, body)]
    geometry = [m["geometry_span"] for m in measures]
    bones = [m["bone_span"] for m in measures]
    bone_ratio = bones[1] / bones[0] if min(bones) > 1e-6 else None
    if all(value is not None for value in geometry):
        ratio = geometry[1] / geometry[0]
        return {"source": "eye_geometry", "ratio": ratio, "bone_ratio": bone_ratio}
    if bone_ratio is not None:
        # If only one side has measured eyes, its pivots still need to agree
        # with that geometry before trusting a two-pivot fallback.
        if any(value is not None and not 0.75 < value / bone < 1.333333
               for value, bone in zip(geometry, bones)):
            return {"source": "unchanged", "ratio": 1.0, "bone_ratio": bone_ratio}
        return {"source": "eye_bones", "ratio": bone_ratio, "bone_ratio": bone_ratio}
    return {"source": "unchanged", "ratio": 1.0, "bone_ratio": None}


def neck_controls(model, head_index):
    """Find the neck and coincident helper controls without naming assumptions."""
    try:
        neck = bone_index(model, aliases=('首', 'neck', '脖子'))
    except TransplantError:
        return set()
    reference = model.bones[neck]
    tolerance = max(1e-6, math.dist(reference.location,model.bones[head_index].location)*1e-4)
    return {neck} | {i for i,bone in enumerate(model.bones)
                     if bone.parent == reference.parent and math.dist(bone.location,reference.location) < tolerance}


def complete_neck_tubes(model, head_index, selected, stats):
    """Keep a skin tube intact when weight thresholding cuts its middle.

    Require a compact neck/head-weighted surface, a closed lower rim, and
    shared geometry with already selected head surfaces. Clothing and hair
    do not qualify merely because they lie near the neck.
    """
    measurement = eye_measurement(model)
    width = measurement['geometry_span'] or measurement['bone_span']
    if width <= 1e-6:
        return []
    controls = neck_controls(model,head_index) | {head_index}
    if len(controls) < 2:
        return []
    anchor = model.bones[head_index].location
    positions = [v.co for v in model.vertices]
    head_points = {tuple(round(float(x), 5) for x in positions[i])
                   for face, keep in zip(model.faces, selected) if keep for i in face}
    cursor = 0
    changed = []
    for stat in stats:
        end = cursor + stat['faces']
        fraction = stat['head_faces'] / max(1, stat['faces'])
        if stat['mode'] == 'AUTO' and 0.05 < fraction < 0.95:
            faces = model.faces[cursor:end]
            ids = {i for face in faces for i in face}
            compact = (all(math.hypot(positions[i][0]-anchor[0], positions[i][2]-anchor[2]) < width for i in ids)
                       and all(anchor[1]-width*1.2 < positions[i][1] < anchor[1]+width*.7 for i in ids))
            weighted = compact and all(sum(w for b, w in influences(model.vertices[i]) if b in controls) > .95 for i in ids)
            rim = find_rim(positions, {stat['index']:faces}, anchor, width, prefer_low=True) if weighted else None
            if rim and rim['center'][1] < anchor[1] and max(p[1] for p in rim['points'])-min(p[1] for p in rim['points']) < width*.4:
                bottom = {tuple(round(x, 5) for x in p) for p in rim['points']}
                # Only the upper join should connect to head skin, never the
                # body's side of the cut. Exclude already selected tube points.
                other = {tuple(round(float(x),5) for x in positions[i]) for j,face in enumerate(model.faces)
                         if not cursor <= j < end and selected[j] for i in face}
                joins = (head_points & other & {tuple(round(float(x),5) for x in positions[i]) for i in ids}) - bottom
                if len(joins) >= 8:
                    selected[cursor:end] = [True] * stat['faces']
                    stat['head_faces'] = stat['faces']
                    changed.append(stat['index'])
        cursor = end
    return changed


def select_head(model, head, threshold, overrides=None, extra=()):
    roots = {head}
    for name in extra:
        roots.add(bone_index(model, name))
    bones = descendants(model, roots)
    scores = [sum(w for b, w in influences(v) if b in bones) for v in model.vertices]
    selected = []
    stats = []
    cursor = 0
    overrides = overrides or {}
    for i, material in enumerate(model.materials):
        count = material.vertex_count // 3
        part = model.faces[cursor:cursor + count]
        mode = overrides.get(i, "AUTO")
        flags = [mode == "INCLUDE" or (mode == "AUTO" and sum(scores[v] for v in f) / 3 >= threshold) for f in part]
        selected.extend(flags)
        stats.append({"index": i, "name": material.name, "faces": count, "head_faces": sum(flags), "mode": mode})
        cursor += count
    if cursor != len(model.faces):
        raise TransplantError("材质面数与总面数不一致，源 PMX 可能损坏。")
    if not any(selected):
        raise TransplantError("没有识别到头部面，请检查头骨骼或将头部材质设为包含。")
    return bones, selected, stats


def analyze(head, body, options):
    hi = bone_index(head, options.head_bone)
    bi = bone_index(body, options.body_head_bone)
    hb, hf, hs = select_head(head, hi, options.threshold, options.head_materials, options.extra_head_bones)
    bb, bf, bs = select_head(body, bi, options.threshold, options.body_materials)
    neck_tubes = complete_neck_tubes(head, hi, hf, hs) if options.fit_neck else []
    transition_adjustments=[]
    if options.fit_neck:
        # A handful of blended neck triangles must not punch a hole into an
        # otherwise complete face material, or cut the continuous body neck.
        for source, model, index, flags, stats, donor in (("head",head,hi,hf,hs,True),("body",body,bi,bf,bs,False)):
            try:
                left=model.bones[bone_index(model,aliases=("左目","eye_l","left eye"))].location
                right=model.bones[bone_index(model,aliases=("右目","eye_r","right eye"))].location
                neck_radius=math.dist(left,right)
            except TransplantError:
                continue
            cursor=0
            for stat in stats:
                count=stat['faces']; fraction=stat['head_faces']/max(1,count)
                eligible=stat['mode']=='AUTO' and ((donor and 0.99<=fraction<1) or (not donor and 0<fraction<=0.001))
                fringe=[i for i in range(cursor,cursor+count) if flags[i]!=donor] if eligible else []
                if fringe and all(math.dist(tuple(sum(model.vertices[v].co[j] for v in model.faces[i])/3 for j in range(3)),model.bones[index].location)<neck_radius for i in fringe):
                    for i in fringe:
                        flags[i]=donor
                    stat['head_faces']=count if donor else 0
                    transition_adjustments.append({'source':source,'material':stat['name'],'faces':len(fringe)})
                cursor+=count
    if all(bf):
        raise TransplantError("身体所有面都被识别为头部，请检查骨骼或材质覆盖设置。")
    scale = options.scale
    warnings = []
    scale_estimate = {"source": "manual", "ratio": 1.0, "bone_ratio": None}
    if options.auto_scale:
        scale_estimate = estimate_scale(head, body)
        scale *= scale_estimate["ratio"]
        if scale_estimate["source"] == "unchanged":
            warnings.append("缺少可靠的眼部几何或眼骨骼比例，自动比例保持 1；请检查并手动调节。")
        elif scale_estimate["source"] == "eye_geometry" and scale_estimate["bone_ratio"] is not None:
            disagreement = scale_estimate["ratio"] / scale_estimate["bone_ratio"]
            if not 0.8 < disagreement < 1.25:
                warnings.append("眼骨骼布局与眼部几何不一致，已按实际眼部几何估算比例；请检查头身效果。")
    if not math.isfinite(scale) or scale <= 0 or not all(math.isfinite(v) for v in options.offset):
        raise TransplantError("比例必须大于零，偏移必须是有限数值。")
    for label, m, bones in (("头部", head, hb), ("身体", body, bb)):
        escaped = [b.name for i, b in enumerate(m.bones) if i not in bones and any(word in b.name.lower() for word in ("髪", "hair", "头发"))]
        if escaped:
            warnings.append(f"{label}有未挂在头骨骼下的疑似头发骨骼：{', '.join(escaped[:6])}；检查材质范围。")
    return {"head_index": hi, "body_head_index": bi, "head_bones": hb, "body_head_bones": bb,
            "head_faces": hf, "body_head_faces": bf, "head_materials": hs, "body_materials": bs,
            "scale": scale, "scale_estimate": scale_estimate, "warnings": warnings,
            "neck_tube_materials": neck_tubes, "neck_transition_adjustments":transition_adjustments}


def _ancestor(model, index, mapping, fallback):
    seen = set()
    while index is not None and index >= 0 and index not in seen:
        if index in mapping:
            return mapping[index]
        seen.add(index)
        index = model.bones[index].parent
    return fallback


def _remap_bone(b, mapping, parent_fallback):
    b.parent = mapping.get(b.parent, parent_fallback)
    if isinstance(b.displayConnection, int):
        b.displayConnection = mapping.get(b.displayConnection, -1)
    if b.additionalTransform:
        index, influence = b.additionalTransform
        if index in mapping:
            b.additionalTransform = (mapping[index], influence)
        else:
            b.additionalTransform = None
            b.hasAdditionalRotate = b.hasAdditionalLocation = False
    if b.isIK:
        if b.target not in mapping or any(x.target not in mapping for x in b.ik_links):
            raise TransplantError(f"骨骼 {b.name} 的 IK 依赖被切除，请调整范围。")
        b.target = mapping[b.target]
        for link in b.ik_links:
            link.target = mapping[link.target]


def transplant(pmx, head, body, options=None):
    options = options or Options()
    plan = analyze(head, body, options)
    hi, bi = plan["head_index"], plan["body_head_index"]
    hb, bb = plan["head_bones"], plan["body_head_bones"]
    scale = plan["scale"]
    source_anchor = head.bones[hi].location
    target_anchor = body.bones[bi].location
    def position(v):
        return tuple((v[i] - source_anchor[i]) * scale + target_anchor[i] + options.offset[i] for i in range(3))
    def delta(v):
        return tuple(x * scale for x in v)

    out = pmx.Model()
    out.name = f"{head.name} × {body.name}"
    out.name_e = f"{head.name_e} on {body.name_e}"
    out.comment = f"MMD Transplant 0.3.1\n头部来源：{head.filepath}\n身体来源：{body.filepath}\n\n{head.comment}\n\n{body.comment}"
    out.comment_e = head.comment_e + "\n" + body.comment_e
    keep_body = set(range(len(body.bones))) - (bb - {bi})
    bm = {old: new for new, old in enumerate(sorted(keep_body))}
    hm = {hi: bm[bi]}
    # Preserve the body head/neck and all body locomotion bone indices in relative order.
    names = {b.name: bm[i] for i, b in enumerate(body.bones) if i in bm}
    names_e = {b.name_e.casefold(): bm[i] for i, b in enumerate(body.bones) if i in bm and b.name_e}
    donor_indices = sorted(hb - {hi})
    for i in donor_indices:
        hm[i] = len(bm) + len(hm) - 1
    # External donor dependencies map by name; never bring along a second body rig.
    for i, b in enumerate(head.bones):
        if i not in hm:
            if b.name in names:
                hm[i] = names[b.name]
            elif b.name_e and b.name_e.casefold() in names_e:
                hm[i] = names_e[b.name_e.casefold()]
    # Some exporters duplicate the neck control solely for skin weighting.
    # Its coincident bind position/parent identifies the same body control.
    for i in neck_controls(head,hi):
        if i not in hm:
            source_neck = bone_index(head,aliases=('首','neck','脖子'))
            if source_neck in hm:
                hm[i] = hm[source_neck]
    used_names = set(names)
    for i in sorted(keep_body):
        b = deepcopy(body.bones[i])
        _remap_bone(b, bm, _ancestor(body, body.bones[i].parent, bm, -1))
        out.bones.append(b)
    # The body head control stays fixed. Offset affects donor child bones/geometry only.
    for i in donor_indices:
        b = deepcopy(head.bones[i])
        b.location = position(b.location)
        if not isinstance(b.displayConnection, int):
            b.displayConnection = delta(b.displayConnection)
        _remap_bone(b, hm, bm[bi])
        if b.name in used_names:
            original = b.name
            suffix = 1
            while b.name in used_names:
                b.name = f"{original}_head{suffix}"
                suffix += 1
        used_names.add(b.name)
        out.bones.append(b)

    maps = []
    for model, selected, bone_map, donor in ((body, [not x for x in plan["body_head_faces"]], bm, False), (head, plan["head_faces"], hm, True)):
        vertex_map, material_map, texture_map = {}, {}, {}
        cursor = 0
        for mi, mat in enumerate(model.materials):
            end = cursor + mat.vertex_count // 3
            faces = [f for f, keep in zip(model.faces[cursor:end], selected[cursor:end]) if keep]
            cursor = end
            if not faces:
                continue
            material_map[mi] = len(out.materials)
            material = deepcopy(mat)
            material.name = ("头_" if donor else "身_") + material.name
            material.vertex_count = len(faces) * 3
            if donor and hasattr(material, "edge_size"):
                material.edge_size *= scale
            for attr in ("texture", "sphere_texture", "toon_texture"):
                if attr == "toon_texture" and mat.is_shared_toon_texture:
                    continue
                old = getattr(mat, attr)
                if old < 0:
                    continue
                if old not in texture_map:
                    texture_map[old] = len(out.textures)
                    out.textures.append(deepcopy(model.textures[old]))
                setattr(material, attr, texture_map[old])
            out.materials.append(material)
            for face in faces:
                new_face = []
                for vi in face:
                    if vi not in vertex_map:
                        v = deepcopy(model.vertices[vi])
                        v.weight.bones = [bone_map.get(b, _ancestor(model, b, bone_map, bm[bi])) if b >= 0 else -1 for b in v.weight.bones]
                        if donor:
                            v.co = position(v.co)
                            if v.weight.type == 3:
                                for attr in ("c", "r0", "r1"):
                                    setattr(v.weight.weights, attr, position(getattr(v.weight.weights, attr)))
                        vertex_map[vi] = len(out.vertices)
                        out.vertices.append(v)
                    new_face.append(vertex_map[vi])
                out.faces.append(tuple(new_face))
        maps.append((vertex_map, material_map, bone_map))

    # Copy only surviving morph offsets. Identical JP morph names share one slider.
    morph_maps = [{}, {}]
    by_name = {}
    pending = []
    for source, model in enumerate((body, head)):
        vm, mm, bone_map = maps[source]
        allowed_bones = set(bm) if source == 0 else hb
        for old, morph in enumerate(model.morphs):
            kind = morph.type_index()
            if kind == 0:
                pending.append((source, old, morph))
                continue
            filtered = deepcopy(morph)
            filtered.offsets = []
            for original in morph.offsets:
                mapping = vm if kind == 1 or 3 <= kind <= 7 else bone_map if kind == 2 else mm if kind == 8 else None
                if mapping is None:
                    raise TransplantError(f"不支持表情类型 {kind}。")
                if kind == 2 and original.index not in allowed_bones:
                    continue
                targets = list(mm) if kind == 8 and original.index == -1 else [original.index]
                for index in targets:
                    if index not in mapping:
                        continue
                    offset = deepcopy(original)
                    offset.index = mapping[index]
                    if source == 1:
                        if kind == 1:
                            offset.offset = delta(offset.offset)
                        elif kind == 2:
                            offset.location_offset = delta(offset.location_offset)
                    filtered.offsets.append(offset)
            if filtered.offsets:
                key = (morph.name, kind)
                if key in by_name:
                    new = by_name[key]
                    out.morphs[new].offsets.extend(filtered.offsets)
                else:
                    new = len(out.morphs)
                    by_name[key] = new
                    out.morphs.append(filtered)
                morph_maps[source][old] = new
    # Groups may refer to groups declared later. Reserve then filter to a fixed point.
    for source, old, morph in pending:
        key = (morph.name, 0)
        if key not in by_name:
            by_name[key] = len(out.morphs)
            m = deepcopy(morph)
            m.offsets = []
            out.morphs.append(m)
        morph_maps[source][old] = by_name[key]
    for source, old, morph in pending:
        new = morph_maps[source][old]
        for offset in morph.offsets:
            if offset.morph in morph_maps[source]:
                o = deepcopy(offset)
                o.morph = morph_maps[source][o.morph]
                out.morphs[new].offsets.append(o)
    alive = {i for i, m in enumerate(out.morphs) if m.type_index() != 0 and m.offsets}
    while True:
        expanded = alive | {i for i, m in enumerate(out.morphs) if m.type_index() == 0 and any(o.morph in alive for o in m.offsets)}
        if expanded == alive:
            break
        alive = expanded
    compact = {old: new for new, old in enumerate(sorted(alive))}
    out.morphs = [m for i, m in enumerate(out.morphs) if i in alive]
    for m in out.morphs:
        if m.type_index() == 0:
            m.offsets = [o for o in m.offsets if o.morph in compact]
            for o in m.offsets:
                o.morph = compact[o.morph]
    morph_maps = [{old: compact[new] for old, new in mapping.items() if new in compact} for mapping in morph_maps]

    out.display = []
    for source, model in enumerate((body, head)):
        allowed = keep_body if source == 0 else hb
        bone_map = maps[source][2]
        for display in model.display:
            d = deepcopy(display)
            d.data = [(kind, (bone_map if kind == 0 else morph_maps[source])[index]) for kind, index in display.data
                      if (kind == 0 and index in allowed and index in bone_map) or (kind == 1 and index in morph_maps[source])]
            if not d.data:
                continue
            same = next((x for x in out.display if x.name == d.name and x.isSpecial == d.isSpecial), None)
            if same:
                same.data = list(dict.fromkeys(same.data + d.data))
            else:
                out.display.append(d)

    rigid_maps = [{}, {}]
    body_rigid_by_bone = {}
    # Keep body physics irrespective of donor-physics toggle.
    for source, model in enumerate((body, head)):
        bone_map = maps[source][2]
        for i, rigid in enumerate(model.rigids):
            keep = (rigid.bone not in bb) if source == 0 else (options.physics and rigid.bone in hb)
            if not keep:
                continue
            r = deepcopy(rigid)
            r.bone = bone_map.get(r.bone, None)
            if source == 1:
                r.location, r.size = position(r.location), delta(r.size)
                r.name = "头_" + r.name
            rigid_maps[source][i] = len(out.rigids)
            if source == 0 and r.bone is not None:
                body_rigid_by_bone.setdefault(r.bone, len(out.rigids))
            out.rigids.append(r)
        if source == 1 and options.physics:
            for i, r in enumerate(model.rigids):
                if i not in rigid_maps[1] and r.bone not in hb and hm.get(r.bone) in body_rigid_by_bone:
                    rigid_maps[1][i] = body_rigid_by_bone[hm[r.bone]]
        for joint in model.joints:
            rm = rigid_maps[source]
            if joint.src_rigid not in rm or joint.dest_rigid not in rm:
                continue
            if source == 1 and not (model.rigids[joint.src_rigid].bone in hb or model.rigids[joint.dest_rigid].bone in hb):
                continue
            j = deepcopy(joint)
            j.src_rigid, j.dest_rigid = rm[j.src_rigid], rm[j.dest_rigid]
            if source == 1:
                j.location = position(j.location)
                j.minimum_location, j.maximum_location = delta(j.minimum_location), delta(j.maximum_location)
                j.name = "头_" + j.name
            out.joints.append(j)
    neck_report = {"status": "off"}
    if options.fit_neck:
        try:
            left = out.bones[bone_index(out, aliases=("左目", "eye_l", "left eye"))].location
            right = out.bones[bone_index(out, aliases=("右目", "eye_r", "right eye"))].location
            width = eye_measurement(out)['geometry_span'] or math.dist(left, right)
            repair_materials = {maps[1][1][i] for i in plan['neck_tube_materials'] if i in maps[1][1]}
            neck_report = fit_neck_pmx(pmx, out, set(maps[1][1].values()), set(maps[0][1].values()), target_anchor, width, repair_materials)
        except TransplantError:
            neck_report = {"status": "skipped", "reason": "找不到眼骨骼，无法可靠估计颈部搜索范围"}
        if neck_report["status"] == "skipped":
            plan["warnings"].append("颈部未自动贴合：" + neck_report["reason"])
    validate(out)
    report = {k: v for k, v in plan.items() if k not in ("head_faces", "body_head_faces", "head_bones", "body_head_bones")}
    report.update(neck_fit=neck_report, vertices=len(out.vertices), faces=len(out.faces), bones=len(out.bones), morphs=len(out.morphs),
                  rigids=len(out.rigids), joints=len(out.joints), donor_faces=sum(plan["head_faces"]),
                  removed_body_faces=sum(plan["body_head_faces"]), body_bone_map=bm, head_bone_map=hm,
                  donor_vertex_map=maps[1][0], body_vertex_map=maps[0][0])
    return out, report


def fit_neck_pmx(pmx, model, head_materials, body_materials, anchor, width, repair_materials=()):
    head_faces, body_faces = {}, {}
    cursor = 0
    for i, material in enumerate(model.materials):
        faces = model.faces[cursor:cursor + material.vertex_count//3]
        cursor += material.vertex_count//3
        if i in head_materials:
            head_faces[i] = faces
        elif i in body_materials:
            body_faces[i] = faces
    if repair_materials:
        repaired = repair_neck_overlap(pmx, model, head_faces, body_faces, anchor, width, repair_materials)
        if repaired is not None:
            return repaired
        return {"status":"skipped", "reason":"已保留完整颈部皮肤，但无法确认安全修剪和连接范围；请检查材质覆盖或手动处理"}
    plan = fit_plan([v.co for v in model.vertices], head_faces, body_faces, anchor, width)
    if plan is None:
        bridge=bridge_plan([v.co for v in model.vertices],head_faces,body_faces,anchor,width)
        if bridge is not None:
            return bridge_neck_pmx(pmx,model,bridge)
        return {"status": "skipped", "reason": "未找到可信颈圈，或颈圈交叉/距离过远；可检查材质覆盖或手动处理"}
    changes = plan['changes']
    if plan['max_delta'] > width*1e-4:
        for index, change in changes.items():
            v, reference = model.vertices[index], model.vertices[change['reference']]
            v.co = tuple(v.co[i]+change['delta'][i] for i in range(3))
            t = change['weight_factor']
            weights = {}
            for bone, weight in influences(v):
                if bone >= 0:
                    weights[bone] = weights.get(bone,0) + weight*(1-t)
            for bone, weight in influences(reference):
                if bone >= 0:
                    weights[bone] = weights.get(bone,0) + weight*t
            ordered = sorted(((b,w) for b,w in weights.items() if w>1e-8),key=lambda x:-x[1])[:4]
            total = sum(w for b,w in ordered)
            if not total:
                raise TransplantError("颈部边界没有有效权重。")
            weight = pmx.BoneWeight()
            if len(ordered)==1:
                weight.type=weight.BDEF1
                weight.bones=[ordered[0][0]]
            else:
                weight.type=weight.BDEF4
                weight.bones=[b for b,w in ordered]+[-1]*(4-len(ordered))
                weight.weights=[w/total for b,w in ordered]+[0]*(4-len(ordered))
            v.weight=weight
            if hasattr(v,'normal') and hasattr(reference,'normal'):
                normal=tuple(v.normal[i]*(1-t)+reference.normal[i]*t for i in range(3))
                length=math.sqrt(sum(x*x for x in normal))
                if length>1e-8:
                    v.normal=tuple(x/length for x in normal)
        # Positional fitting translates every shape's basis. Morph offsets near
        # the seam also blend to the body's offsets, so expressions cannot split it.
        for morph in model.morphs:
            if morph.type_index()!=1:
                continue
            offsets={o.index:o for o in morph.offsets}
            original={i:tuple(o.offset) for i,o in offsets.items()}
            for index, change in changes.items():
                a=original.get(index,(0,0,0)); b=original.get(change['reference'],(0,0,0));t=change['weight_factor']
                value=tuple(a[i]*(1-t)+b[i]*t for i in range(3))
                if any(abs(v)>1e-10 for v in value):
                    if index not in offsets:
                        o=pmx.VertexMorphOffset();o.index=index
                        offsets[index]=o
                    offsets[index].offset=value
                else:
                    offsets.pop(index,None)
            morph.offsets=list(offsets.values())
    return {"status":"fitted","rim_vertices":plan['rim_vertices'],"affected_vertices":len(changes),
            "max_displacement":plan['max_delta'],"head_material":model.materials[plan['head_rim']['material']].name,
            "body_material":model.materials[plan['body_rim']['material']].name}


def repair_neck_overlap(pmx, model, head_faces, body_faces, anchor, width, repair_materials):
    """Replace a threshold-cut tube join with two safe cuts and a skin strip."""
    from .neck_clip import clip_material
    points = [v.co for v in model.vertices]
    head = find_rim(points, {i:head_faces[i] for i in repair_materials if i in head_faces}, anchor, width, prefer_low=True)
    body = find_rim(points, body_faces, anchor, width)
    if not head or not body:
        return None
    if min(p[1] for p in head['points']) > max(p[1] for p in body['points']) + width*1e-5:
        plan = bridge_plan(points, head_faces, body_faces, anchor, width,
                           head_rim=head, body_rim=body, allow_equal=True)
        return bridge_neck_pmx(pmx,model,plan) if plan else None
    head_plane = max(p[1] for p in head['points']) + width*.02
    body_plane = min(p[1] for p in body['points']) - width*.02
    if not 0 < head_plane-body_plane < width*.5:
        return None
    if any(math.hypot(points[i][0]-anchor[0],points[i][2]-anchor[2]) > width*1.2
           for face in body_faces[body['material']] for i in face if points[i][1] > body_plane):
        return None
    # Never trim into the jaw's attachment to the tube, even if the chin
    # itself protrudes below this plane on another material.
    tube_ids = {i for face in head_faces[head['material']] for i in face}
    tube_keys = {tuple(round(float(x),5) for x in points[i]) for i in tube_ids}
    joins = [points[i][1] for material,faces in head_faces.items() if material != head['material']
             for face in faces for i in face if tuple(round(float(x),5) for x in points[i]) in tube_keys]
    if not joins or head_plane >= min(joins)-width*.01:
        return None
    candidate = deepcopy(model)
    cuts = [clip_material(pmx,candidate,head['material'],head_plane,True),
            clip_material(pmx,candidate,body['material'],body_plane,False)]
    updated_head,updated_body = {},{}
    cursor = 0
    for i,material in enumerate(candidate.materials):
        end = cursor+material.vertex_count//3
        if i in head_faces:updated_head[i]=candidate.faces[cursor:end]
        if i in body_faces:updated_body[i]=candidate.faces[cursor:end]
        cursor=end
    points = [v.co for v in candidate.vertices]
    head = find_rim(points,{head['material']:updated_head[head['material']]},anchor,width,prefer_low=True)
    body = find_rim(points,{body['material']:updated_body[body['material']]},anchor,width)
    plan = bridge_plan(points,updated_head,updated_body,anchor,width,
                       head_rim=head,body_rim=body,allow_equal=True) if head and body else None
    if plan is None:
        return None
    result = bridge_neck_pmx(pmx,candidate,plan)
    validate(candidate)
    for attr in ('vertices','faces','materials','morphs'):
        setattr(model,attr,getattr(candidate,attr))
    result['trimmed_neck'] = True
    result['interpolated_vertices'] = sum(len(cut['created_indices']) for cut in cuts)
    return result


def bridge_neck_pmx(pmx, model, plan):
    """Add an unequal-rim skin strip with its own UVs and source deformations."""
    head,body=plan['head_rim'],plan['body_rim']
    sources=plan['source_indices'];n=len(head['points'])
    start=len(model.vertices);material_index=len(model.materials)
    material=deepcopy(model.materials[body['material']])
    material.name='头_颈部连接'
    material.vertex_count=len(plan['triangles'])*3
    # This strip is skin geometry, not a new physics chain. Copy weights/SDEF
    # and position morphs from each exact boundary source to keep both joins shut.
    for i,source in enumerate(sources):
        vertex=deepcopy(model.vertices[source])
        if i<n:
            # Use the body's skin atlas on the whole strip. Project the donor
            # rim to the nearest body edge to sample a continuous skin UV.
            best=None
            for j,a in enumerate(body['points']):
                b=body['points'][(j+1)%len(body['points'])]
                delta=tuple(b[k]-a[k] for k in range(3))
                length=sum(x*x for x in delta)
                t=max(0,min(1,sum((vertex.co[k]-a[k])*delta[k] for k in range(3))/length)) if length else 0
                distance=sum((vertex.co[k]-a[k]-delta[k]*t)**2 for k in range(3))
                if best is None or distance<best[0]:best=(distance,j,t)
            _,j,t=best
            a=model.vertices[body['members'][j][0]]
            b=model.vertices[body['members'][(j+1)%len(body['points'])][0]]
            vertex.uv=tuple(a.uv[k]*(1-t)+b.uv[k]*t for k in range(2))
            vertex.additional_uvs=deepcopy(a.additional_uvs)
        model.vertices.append(vertex)
    for morph in model.morphs:
        kind=morph.type_index()
        if kind==1 or 3<=kind<=7:
            originals={o.index:o for o in morph.offsets}
            for i,source in enumerate(sources):
                # Donor UVs use a different atlas; only position offsets carry
                # across that boundary. Body UV morphs use the same atlas.
                if source in originals and (kind==1 or i>=n):
                    offset=deepcopy(originals[source]);offset.index=start+i;morph.offsets.append(offset)
        elif kind==8:
            for offset in list(morph.offsets):
                if offset.index==body['material']:
                    copied=deepcopy(offset);copied.index=material_index;morph.offsets.append(copied)
    model.materials.append(material)
    model.faces.extend(tuple(start+i for i in triangle) for triangle in plan['triangles'])
    return {'status':'bridged','head_rim_vertices':n,'body_rim_vertices':len(body['points']),
            'bridge_vertices':len(sources),'bridge_faces':len(plan['triangles']),
            'head_material':model.materials[head['material']].name,
            'body_material':model.materials[body['material']].name,'bridge_material':material.name}


def validate(model):
    def check(index, count, label, optional=False):
        if optional and index in (None, -1):
            return
        if not isinstance(index, int) or not 0 <= index < count:
            raise TransplantError(f"{label}引用越界：{index} / {count}")
    if sum(m.vertex_count for m in model.materials) != len(model.faces) * 3:
        raise TransplantError("输出材质面数不一致。")
    if not model.vertices or not model.faces:
        raise TransplantError("输出网格为空。")
    for face in model.faces:
        for index in face:
            check(index, len(model.vertices), "面顶点")
    for vertex in model.vertices:
        if not all(math.isfinite(x) for x in vertex.co):
            raise TransplantError("顶点包含非有限坐标。")
        for index, weight in influences(vertex):
            check(index, len(model.bones), "顶点权重", optional=weight <= 1e-6)
    for i, b in enumerate(model.bones):
        check(b.parent, len(model.bones), "骨骼父级", True)
        if isinstance(b.displayConnection, int):
            check(b.displayConnection, len(model.bones), "骨骼尾端", True)
        if b.additionalTransform:
            check(b.additionalTransform[0], len(model.bones), "追加变换")
        if b.isIK:
            check(b.target, len(model.bones), "IK目标")
            for link in b.ik_links:
                check(link.target, len(model.bones), "IK链")
        seen = {i}
        parent = b.parent
        while parent not in (None, -1):
            if parent in seen:
                raise TransplantError(f"骨骼父级循环：{b.name}")
            seen.add(parent)
            parent = model.bones[parent].parent
    for mat in model.materials:
        for attr in ("texture", "sphere_texture", "toon_texture"):
            if attr != "toon_texture" or not mat.is_shared_toon_texture:
                check(getattr(mat, attr), len(model.textures), "材质贴图", True)
    for morph in model.morphs:
        kind = morph.type_index()
        for o in morph.offsets:
            count = len(model.morphs) if kind == 0 else len(model.bones) if kind == 2 else len(model.materials) if kind == 8 else len(model.vertices)
            check(o.morph if kind == 0 else o.index, count, "表情", kind == 8)
    for r in model.rigids:
        check(r.bone, len(model.bones), "刚体骨骼", True)
    for j in model.joints:
        check(j.src_rigid, len(model.rigids), "关节A")
        check(j.dest_rigid, len(model.rigids), "关节B")
    for d in model.display:
        for kind, index in d.data:
            check(index, len(model.bones) if kind == 0 else len(model.morphs), "显示枠")


def bundle_textures(model, folder):
    folder = Path(folder)
    missing = []
    for texture in model.textures:
        source = Path(texture.path)
        if not source.is_file():
            missing.append(str(source))
            continue
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
        target = folder / "textures" / (digest + source.suffix.lower())
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        texture.path = str(target.resolve())
    return missing
