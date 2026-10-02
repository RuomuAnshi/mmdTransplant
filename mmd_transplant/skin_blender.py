"""Derived neck textures and optional shader compatibility; source data is read-only."""
from array import array
from collections import defaultdict
from pathlib import Path
import struct
import sys
import uuid
import zlib

import bpy
from .skin_color import estimate_base_color, gradient_rgba, linear_to_srgb, sample_bilinear_rgba, srgb_to_linear
from .core import set_bridge_texture


def material_name(material):
    return (material.get('zd_original_name') or
            getattr(getattr(material, 'mmd_material', None), 'name_j', '') or material.name)


def is_bridge(material):
    return material is not None and (material_name(material).startswith('头_颈部连接')
                                   or material.get('mmd_transplant_surface') == 'SKIN_BRIDGE')


class ImageSampler:
    def __init__(self):
        self.owned = []
        self.buffers = {}

    def close(self):
        for image in self.owned:
            bpy.data.images.remove(image)

    def load(self, path):
        image = bpy.data.images.load(str(path), check_existing=False)
        self.owned.append(image)
        return image

    def sample(self, image, uv):
        key = image.as_pointer()
        if key not in self.buffers:
            width, height = image.size
            if not width or not height:
                raise ValueError('贴图没有可读取的像素')
            space = image.colorspace_settings.name
            if space not in {'sRGB', 'Linear', 'Linear Rec.709', 'Non-Color'}:
                raise ValueError('暂不支持该贴图色彩空间')
            pixels = array('f', [0]) * len(image.pixels)
            image.pixels.foreach_get(pixels)
            self.buffers[key] = (pixels, width, height, image.channels, space)
        pixels, width, height, channels, space = self.buffers[key]
        rgba = sample_bilinear_rgba(pixels, width, height, uv, channels)
        # Byte sRGB image buffers retain encoded values in Blender; do not
        # assume Image.pixels is always scene-linear.
        rgb = srgb_to_linear(rgba) if space == 'sRGB' else rgba[:3]
        return tuple(rgb) + (rgba[3],)


def _write_gradient(path, head, body, strength):
    width, height = 16, 128
    pixels = gradient_rgba(head, body, width, height, strength)
    rows = []
    for y in reversed(range(height)):
        row = bytearray(b'\0')
        for x in range(width):
            offset = (y * width + x) * 4
            row.extend(round(max(0, min(1, c)) * 255) for c in linear_to_srgb(pixels[offset:offset+3]))
            row.append(255)
        rows.append(bytes(row))
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind+data) & 0xffffffff)
    data = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 6, 0, 0, 0))
    data += chunk(b'sRGB', b'\0') + chunk(b'IDAT', zlib.compress(b''.join(rows))) + chunk(b'IEND', b'')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _endpoints(head_samples, body_samples, head_diffuse, body_diffuse, bridge_diffuse=None):
    estimates = [estimate_base_color(samples) for samples in (head_samples, body_samples)]
    if any(e['status'] != 'ok' for e in estimates):
        raise ValueError('颈圈颜色采样不可靠：透明、过暗或颜色差异过大')
    bridge_diffuse = bridge_diffuse or body_diffuse
    if min(bridge_diffuse[:3]) < .05:
        raise ValueError('身体材质颜色过暗，无法安全匹配')
    head = tuple(estimates[0]['rgb_linear'][i] * head_diffuse[i] / bridge_diffuse[i] for i in range(3))
    body = tuple(estimates[1]['rgb_linear'][i] * body_diffuse[i] / bridge_diffuse[i] for i in range(3))
    if not all(0 <= c <= 1 for c in head+body):
        raise ValueError('匹配颜色超出贴图范围，请手动检查材质颜色')
    return head, body, min(e['confidence'] for e in estimates)


def prepare_bridge_skin(pmx, model, report, folder, strength=1.0):
    """Bake one new bridge atlas before texture bundling / PMX serialization."""
    meta = report['neck_fit'].get('skin_bridge')
    if strength == 0:
        return {'status': 'off'}
    if not meta:
        return {'status': 'skipped', 'reason': '没有新增颈部连接带，肤色过渡未应用'}
    sampler = ImageSampler()
    try:
        samples, diffuse = [], []
        for side in ('head', 'body'):
            material = model.materials[meta[side+'_material']]
            color = tuple(material.diffuse)
            if len(color) > 3 and color[3] < .9:
                raise ValueError('边界皮肤材质含透明度，肤色过渡未应用')
            diffuse.append(color)
            uvs = meta[side+'_uvs']
            if material.texture >= 0:
                image = sampler.load(model.textures[material.texture].path)
                sample = [sampler.sample(image, (uv[0], 1-uv[1])) for uv in uvs]
            else:
                sample = [(1, 1, 1, 1)] * len(uvs)
            samples.append(sample)
        head, body, confidence = _endpoints(*samples, *diffuse)
        path = Path(folder) / 'textures' / ('neck_skin_'+uuid.uuid4().hex[:12]+'.png')
        _write_gradient(path, head, body, strength)
        set_bridge_texture(pmx, model, report['neck_fit'], path.resolve())
        return {'status': 'matched', 'confidence': confidence, 'strength': strength,
                'scope': 'bridge_base_texture'}
    except (ValueError, RuntimeError, OSError) as exc:
        return {'status': 'skipped', 'reason': str(exc)}
    finally:
        sampler.close()


def mark_skin_materials(objects, roles=None, scene=None):
    """Mark generated skin and update already converted Zelda materials when available."""
    changed = 0
    prepared = {}
    for obj in objects:
        if obj.type != 'MESH':
            continue
        for index, slot in enumerate(obj.material_slots):
            material = slot.material
            if material is None:
                continue
            if not is_bridge(material) and not (roles and roles.get(index) == 'SKIN'):
                continue
            pointer = material.as_pointer()
            if pointer in prepared:
                slot.link = 'OBJECT'
                slot.material = prepared[pointer]
                continue
            if material.users > 1:
                material = material.copy()
                slot.link = 'OBJECT'
                slot.material = material
            prepared[pointer] = material
            material['mmd_transplant_surface'] = 'SKIN_BRIDGE'
            # Keep shader tools' name-based AUTO recognition on an unconverted
            # material. A converted material retains its original identity.
            if 'skin' not in material.name.casefold():
                material.name += '_skin'
            if 'zd_owned' in material:
                modules = [m for name, m in list(sys.modules.items()) if name.endswith('.zelda_daylight.shading')]
                if modules and scene is not None and hasattr(scene, 'zelda_daylight'):
                    try:
                        modules[0].update_material(material, 'SKIN', scene.zelda_daylight)
                    except (KeyError, AttributeError, RuntimeError, TypeError):
                        material['mmd_transplant_shader_warning'] = '现有 Zelda 节点已编辑，皮肤阴影请手动检查'
            material['zd_role'] = 'SKIN'
            changed += 1
    return changed


def _base_image(material):
    if not material.use_nodes:
        return None, None, tuple(material.diffuse_color)
    groups = [n for n in material.node_tree.nodes if n.type == 'GROUP'
              and all(n.inputs.get(name) is not None for name in ('Base Tex', 'Diffuse Color', 'Base Tex Fac'))]
    if not groups:
        raise ValueError('仅支持 MMD Tools 基础贴图节点，无法安全处理当前自定义材质')
    group = groups[0]
    if group.inputs['Diffuse Color'].is_linked or group.inputs['Base Tex Fac'].is_linked:
        raise ValueError('材质颜色已由自定义节点控制，无法可靠采样')
    if abs(group.inputs['Base Tex Fac'].default_value-1) > 1e-5:
        raise ValueError('基础贴图混合比例不是 1，请手动检查')
    socket = group.inputs['Base Tex']
    image_node = socket.links[0].from_node if socket.is_linked else None
    if image_node is not None and image_node.type != 'TEX_IMAGE':
        raise ValueError('基础贴图已被自定义节点处理，请手动检查')
    color = tuple(group.inputs['Diffuse Color'].default_value)
    return image_node, group, color


def _base_uv(mesh, node):
    layer = next((uv for uv in mesh.data.uv_layers if uv.active_render), None)
    if layer is None:
        raise ValueError('没有渲染 UV 层')
    vector = node.inputs['Vector'] if node else None
    if vector is not None and not vector.is_linked:
        raise ValueError('基础贴图没有明确的 UV 输入，无法安全替换')
    if vector and vector.is_linked:
        link = vector.links[0]
        source = link.from_node
        if source.type == 'UVMAP':
            layer = mesh.data.uv_layers.get(source.uv_map) if source.uv_map else layer
        elif source.type == 'TEX_COORD' and link.from_socket.name == 'UV':
            pass
        elif source.type == 'GROUP' and source.node_tree and source.node_tree.name.startswith('MMDTexUV') and link.from_socket.name == 'Base UV':
            pass
        else:
            raise ValueError('基础贴图使用自定义 UV 映射，无法安全替换')
    if layer is None:
        raise ValueError('贴图引用的 UV 层不存在')
    return layer


def _guard_uv_motion(mesh, bridge_ids, layer):
    groups = {g.index for g in mesh.vertex_groups if g.name.startswith('UV_')}
    if any(g.group in groups and g.weight > 1e-8 for index in bridge_ids for g in mesh.data.vertices[index].groups):
        raise ValueError('连接带包含 UV 表情，请重新生成预览以保留源表情并清理连接带偏移')
    for modifier in mesh.modifiers:
        if modifier.type != 'UV_WARP' or modifier.uv_layer not in ('', layer.name):
            continue
        group = mesh.vertex_groups.get(modifier.vertex_group)
        if not modifier.vertex_group or group and any(g.group == group.index and g.weight > 1e-8 for index in bridge_ids for g in mesh.data.vertices[index].groups):
            raise ValueError('连接带受 UV 变形控制，请先保存编辑并重新生成预览')
    root = mesh
    while root.parent is not None:
        root = root.parent
    if hasattr(root, 'mmd_root'):
        for morph in root.mmd_root.uv_morphs:
            if morph.data_type == 'DATA' and any(d.index in bridge_ids for d in morph.data):
                raise ValueError('连接带包含旧式 UV 表情，请重新生成预览')


def match_existing_bridge(mesh, folder, strength=1.0):
    """Prepare all new materials/UVs first; preserve geometry and source images."""
    if strength == 0:
        return {'status': 'off', 'matched_materials': 0}
    if not mesh.data.uv_layers.active:
        raise ValueError('网格没有 UV，无法采样皮肤贴图')
    bridges = {p.material_index for p in mesh.data.polygons if is_bridge(mesh.material_slots[p.material_index].material)}
    if not bridges:
        raise ValueError('没有找到插件生成的颈部连接带；请保留连接材质或重新生成预览')
    uv_layers = {_base_uv(mesh, _base_image(mesh.material_slots[i].material)[0]).name for i in bridges}
    if len(uv_layers) != 1:
        raise ValueError('连接带使用多个不同 UV 层，无法可靠匹配')
    uv = mesh.data.uv_layers[next(iter(uv_layers))]
    bridge_ids = {v for p in mesh.data.polygons if p.material_index in bridges for v in p.vertices}
    _guard_uv_motion(mesh, bridge_ids, uv)
    lookup = defaultdict(list)
    key = lambda co: tuple(round(float(c), 6) for c in co)
    for polygon in mesh.data.polygons:
        if polygon.material_index in bridges:
            continue
        material = mesh.material_slots[polygon.material_index].material
        name = material_name(material) if material else ''
        side = 'head' if name.startswith('头_') else 'body' if name.startswith('身_') else None
        if side:
            for loop_index in polygon.loop_indices:
                vertex = mesh.data.loops[loop_index].vertex_index
                lookup[key(mesh.data.vertices[vertex].co)].append((side, material, tuple(uv.data[loop_index].uv)))
    sampler = ImageSampler()
    prepared = []
    try:
        for index in sorted(bridges):
            original = mesh.material_slots[index].material
            node, group, bridge_diffuse = _base_image(original)
            if node is None or node.image is None:
                raise ValueError('连接带没有可安全替换的基础贴图')
            samples = {'head': [], 'body': []}
            diffuse = {}
            assignments = {}
            unique = set()
            for polygon in mesh.data.polygons:
                if polygon.material_index != index:
                    continue
                for loop_index in polygon.loop_indices:
                    vertex = mesh.data.loops[loop_index].vertex_index
                    point = key(mesh.data.vertices[vertex].co)
                    refs = lookup.get(point, [])
                    sides = {r[0] for r in refs}
                    if len(sides) != 1:
                        raise ValueError('连接边界已被编辑，无法可靠确认头侧和身体侧')
                    side = next(iter(sides))
                    assignments[loop_index] = (.5, 1-.5/128 if side == 'head' else .5/128)
                    if (point, side) in unique:
                        continue
                    unique.add((point, side))
                    _, source, source_uv = refs[0]
                    source_node, _, color = _base_image(source)
                    if _base_uv(mesh, source_node).name != uv.name:
                        raise ValueError('两端皮肤使用不同 UV 层，无法可靠匹配')
                    diffuse.setdefault(side, color)
                    if any(abs(a-b) > 1e-5 for a, b in zip(diffuse[side], color)):
                        raise ValueError('边界跨越多个不同皮肤材质，无法可靠匹配')
                    samples[side].append(sampler.sample(source_node.image, source_uv)
                                         if source_node and source_node.image else (1, 1, 1, 1))
            head, body, confidence = _endpoints(samples['head'], samples['body'], diffuse['head'], diffuse['body'], bridge_diffuse)
            path = Path(folder) / 'textures' / ('neck_skin_'+uuid.uuid4().hex[:12]+'.png')
            _write_gradient(path, head, body, strength)
            material = original.copy()
            new_node, _, _ = _base_image(material)
            image = bpy.data.images.load(str(path.resolve()), check_existing=False)
            new_node.image = image
            new_node.extension = 'EXTEND'
            if hasattr(material, 'mmd_material'):
                material.mmd_material.texture_rel_path = str(path.resolve())
            prepared.append((index, material, assignments, confidence))
        if mesh.data.users > 1:
            mesh.data = mesh.data.copy()
            uv = mesh.data.uv_layers[uv.name]
        for index, material, assignments, confidence in prepared:
            # Object-linked slots avoid altering another object sharing the mesh.
            mesh.material_slots[index].link = 'OBJECT'
            mesh.material_slots[index].material = material
            for loop_index, value in assignments.items():
                uv.data[loop_index].uv = value
        return {'status': 'matched', 'matched_materials': len(prepared),
                'confidence': min(item[3] for item in prepared), 'scope': 'bridge_base_texture'}
    finally:
        sampler.close()
