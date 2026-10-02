"""Clip a PMX material at a neck plane without renumbering existing vertices."""
from copy import deepcopy
import math


def _influences(weight):
    if weight.type == 0:
        values = (1.0,)
    elif weight.type in (1, 3):
        first = weight.weights.weight if weight.type == 3 else weight.weights[0]
        values = (first, 1.0 - first)
    elif weight.type == 2:
        values = weight.weights
    else:
        raise ValueError("Unsupported PMX weight type for neck clipping")
    return zip(weight.bones, values)


def _lerp(a, b, t):
    return tuple(x * (1.0 - t) + y * t for x, y in zip(a, b))


def _vertex(pmx, a, b, t, plane):
    vertex = deepcopy(a)
    point = _lerp(a.co, b.co, t)
    vertex.co = (point[0], plane, point[2])
    normal = _lerp(a.normal, b.normal, t)
    length = math.sqrt(sum(x * x for x in normal))
    vertex.normal = tuple(x / length for x in normal) if length > 1e-12 else tuple(a.normal)
    vertex.uv = _lerp(a.uv, b.uv, t)
    vertex.additional_uvs = [
        _lerp(a.additional_uvs[i] if i < len(a.additional_uvs) else (0, 0, 0, 0),
              b.additional_uvs[i] if i < len(b.additional_uvs) else (0, 0, 0, 0), t)
        for i in range(max(len(a.additional_uvs), len(b.additional_uvs)))
    ]
    vertex.edge_scale = a.edge_scale * (1.0 - t) + b.edge_scale * t
    weights = {}
    for source, factor in ((a, 1.0 - t), (b, t)):
        for bone, value in _influences(source.weight):
            if not math.isfinite(value):
                raise ValueError("Nonfinite PMX weight in neck clipping")
            if bone >= 0 and value > 0:
                weights[bone] = weights.get(bone, 0.0) + value * factor
    ordered = sorted(((bone, value) for bone, value in weights.items() if value > 1e-12),
                     key=lambda item: (-item[1], item[0]))[:4]
    total = sum(value for _, value in ordered)
    if total <= 1e-12:
        raise ValueError("Neck clipping intersection has no valid bone weights")
    weight = pmx.BoneWeight()
    if len(ordered) == 1:
        weight.type = 0
        weight.bones = [ordered[0][0]]
        weight.weights = []
    else:
        weight.type = 2
        weight.bones = [bone for bone, _ in ordered] + [-1] * (4 - len(ordered))
        weight.weights = [value / total for _, value in ordered] + [0.0] * (4 - len(ordered))
    vertex.weight = weight
    return vertex


def _nondegenerate(points):
    a, b, c = points
    u = tuple(b[i] - a[i] for i in range(3))
    v = tuple(c[i] - a[i] for i in range(3))
    cross = (u[1] * v[2] - u[2] * v[1],
             u[2] * v[0] - u[0] * v[2],
             u[0] * v[1] - u[1] * v[0])
    return sum(x * x for x in cross) > 1e-28


def clip_material(pmx, model, material_index, plane, keep_above):
    """Replace one material's triangles with their intersection with a halfspace.

    Existing vertices and indices stay intact. Shared source edges reuse their
    intersections, including interpolation of vertex and UV morph offsets.
    All changes are prepared before modifying the supplied model.
    """
    if not math.isfinite(plane):
        raise ValueError("Neck clipping plane must be finite")
    if not 0 <= material_index < len(model.materials):
        raise ValueError("Neck clipping material is out of range")
    counts = [material.vertex_count // 3 for material in model.materials]
    if any(material.vertex_count % 3 for material in model.materials) or sum(counts) != len(model.faces):
        raise ValueError("PMX material triangle counts are inconsistent")
    start = sum(counts[:material_index])
    stop = start + counts[material_index]
    originals = model.vertices
    pending = []
    intersections = {}
    sources = {}
    faces = []
    epsilon = 1e-9 * max(1.0, abs(plane))

    def coordinate(index):
        return (originals[index] if index < len(originals) else pending[index - len(originals)]).co

    def intersection(a, b):
        if abs(originals[a].co[1] - plane) <= epsilon:
            return a
        if abs(originals[b].co[1] - plane) <= epsilon:
            return b
        a, b = sorted((a, b))
        edge = (a, b)
        if edge not in intersections:
            va, vb = originals[a], originals[b]
            t = (plane - va.co[1]) / (vb.co[1] - va.co[1])
            index = len(originals) + len(pending)
            pending.append(_vertex(pmx, va, vb, t, plane))
            intersections[edge] = index
            sources[index] = {"edge": edge, "t": t}
        return intersections[edge]

    for face in model.faces[start:stop]:
        inside = {i: ((originals[i].co[1] - plane) * (1 if keep_above else -1) >= -epsilon)
                  for i in face}
        polygon = []
        for a, b in zip(face[-1:] + face[:-1], face):
            if inside[a] != inside[b]:
                polygon.append(intersection(a, b))
            if inside[b]:
                polygon.append(b)
        polygon = [index for i, index in enumerate(polygon) if index != polygon[i - 1]]
        for i in range(1, len(polygon) - 1):
            triangle = (polygon[0], polygon[i], polygon[i + 1])
            if _nondegenerate([coordinate(index) for index in triangle]):
                faces.append(triangle)

    morph_additions = []
    for morph in model.morphs:
        kind = morph.type_index()
        if kind != 1 and not 3 <= kind <= 7:
            continue
        length = 3 if kind == 1 else 4
        offsets = {}
        for offset in morph.offsets:
            if len(offset.offset) != length:
                raise ValueError("PMX morph offset dimension is inconsistent")
            before = offsets.get(offset.index, (0.0,) * length)
            offsets[offset.index] = tuple(a + b for a, b in zip(before, offset.offset))
        added = []
        for index, source in sources.items():
            a, b = source["edge"]
            value = _lerp(offsets.get(a, (0.0,) * length),
                          offsets.get(b, (0.0,) * length), source["t"])
            if any(abs(x) > 1e-12 for x in value):
                offset = pmx.VertexMorphOffset() if kind == 1 else pmx.UVMorphOffset()
                offset.index = index
                offset.offset = value
                added.append(offset)
        morph_additions.append((morph, added))

    model.vertices.extend(pending)
    model.faces = model.faces[:start] + faces + model.faces[stop:]
    model.materials[material_index].vertex_count = len(faces) * 3
    for morph, added in morph_additions:
        morph.offsets.extend(added)
    return {"created_indices": list(sources), "sources": sources,
            "original_faces": stop - start, "clipped_faces": len(faces),
            "plane": plane, "keep_above": bool(keep_above)}
