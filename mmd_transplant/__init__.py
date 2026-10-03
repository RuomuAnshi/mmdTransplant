bl_info = {
    "name": "MMD 自动换头 / MMD Transplant",
    "author": "RuomuAnshi",
    "version": (0, 3, 3),
    "blender": (4, 2, 0),
    "location": "3D View > Sidebar > MMD 换头",
    "description": "PMX 自动换头：保留身体骨架、头部表情、头发物理，并支持材质覆盖",
    "category": "Object",
}

import importlib
import json
from pathlib import Path
import sys
import traceback
import uuid

import bpy
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, FloatProperty, FloatVectorProperty, PointerProperty, StringProperty
from bpy_extras.io_utils import ImportHelper
from .core import Options, TransplantError, analyze, bundle_textures, transplant, validate
from .automatic import assess_result, transplant_automatic


def pmx_module():
    # Blender extensions use bl_ext.<repository>.mmd_tools, legacy add-ons use mmd_tools.
    packages = [key for key in bpy.context.preferences.addons.keys() if key == "mmd_tools" or key.endswith(".mmd_tools")]
    packages += [key for key in sys.modules if key.endswith(".core.pmx") and "mmd_tools" in key]
    for package in packages:
        try:
            return importlib.import_module(package if package.endswith(".core.pmx") else package + ".core.pmx")
        except ImportError:
            continue
    raise TransplantError("请先安装并启用 mmd_tools（首选项 → 插件 → MMD Tools）。")


def load_models(settings):
    pmx = pmx_module()
    paths = [Path(bpy.path.abspath(x)).expanduser().resolve() for x in (settings.head_path, settings.body_path)]
    if not all(p.is_file() and p.suffix.lower() == ".pmx" for p in paths):
        raise TransplantError("请选择两份存在的 PMX 文件。当前版本支持 PMX 2.0。")
    return pmx, pmx.load(str(paths[0])), pmx.load(str(paths[1]))


def options_from(s):
    if not s.manual_mode:
        return Options()
    return Options(head_bone=s.head_bone, body_head_bone=s.body_head_bone, threshold=s.threshold,
                   auto_scale=s.auto_scale, scale=s.scale, offset=tuple(s.offset),
                   head_materials={item.index: item.mode for item in s.head_materials},
                   body_materials={item.index: item.mode for item in s.body_materials},
                   extra_head_bones=tuple(x.strip() for x in s.extra_bones.split(",") if x.strip()), physics=s.physics,
                   fit_neck=s.fit_neck)


def invalidate(s, context):
    s.head_materials.clear()
    s.body_materials.clear()
    s.status = "来源已更改，点击 2 一键生成。"


class MMDT_Material(bpy.types.PropertyGroup):
    index: bpy.props.IntProperty()
    name: StringProperty()
    fraction: FloatProperty()
    mode: EnumProperty(items=[("AUTO", "自动", "按头部骨骼权重识别"), ("INCLUDE", "是头部", "头源：包含整份材质；身体源：移除整份材质"), ("EXCLUDE", "非头部", "头源：忽略整份材质；身体源：保留整份材质")], default="AUTO")


class MMDT_Settings(bpy.types.PropertyGroup):
    head_path: StringProperty(name="头部来源", subtype="FILE_PATH", update=invalidate)
    body_path: StringProperty(name="身体来源", subtype="FILE_PATH", update=invalidate)
    output_dir: StringProperty(name="输出文件夹", subtype="DIR_PATH", default="//mmd_transplant_output/")
    head_bone: StringProperty(name="头源的头骨骼", description="留空自动识别 頭 / head / 头")
    body_head_bone: StringProperty(name="身源的头骨骼", description="留空自动识别 頭 / head / 头")
    extra_bones: StringProperty(name="额外头部骨骼", description="额外子树根骨骼，英文逗号分隔；用于不挂在頭下面的头发")
    threshold: FloatProperty(name="头部权重阈值", min=0.01, max=1.0, default=0.5)
    auto_scale: BoolProperty(name="自动匹配头部比例", description="优先按实际眼部几何匹配；眼骨骼位置不可靠时提示检查", default=True)
    scale: FloatProperty(name="头部比例微调", min=0.01, max=10.0, default=1.0)
    offset: FloatVectorProperty(name="位置微调", description="PMX 单位：X 左右，Y 上下，Z 前后；1 PMX 单位导入后默认 0.08 Blender 单位", size=3, default=(0, 0, 0), step=1, precision=3)
    physics: BoolProperty(name="保留新头部刚体和关节", default=True)
    fit_neck: BoolProperty(name="自动颈部贴合", description="相同点数局部贴合，不同点数生成皮肤连接面；保留权重与顶点表情", default=True)
    match_skin: BoolProperty(name="颈部肤色过渡", description="为新增连接带采样两端皮肤并生成独立渐变贴图；原脸部与身体贴图不修改", default=True)
    skin_strength: FloatProperty(name="肤色过渡强度", description="0 保留身体贴图；1 将连接带上端匹配头侧皮肤，下端匹配身体", min=0, max=1, default=1)
    import_scale: FloatProperty(name="Blender 导入比例", default=0.08, min=0.001, max=10)
    manual_mode: BoolProperty(name="使用手动参数", description="关闭时自动分析、匹配比例和颈部，忽略旧的微调与材质覆盖", default=False)
    adjustments_open: BoolProperty(name="手动调整与分析", default=False)
    export_open: BoolProperty(name="导出文件", default=False)
    materials_open: BoolProperty(name="材质切分覆盖", default=False)
    advanced_open: BoolProperty(name="骨骼与导入设置", default=False)
    head_materials: CollectionProperty(type=MMDT_Material)
    body_materials: CollectionProperty(type=MMDT_Material)
    head_material_index: bpy.props.IntProperty()
    body_material_index: bpy.props.IntProperty()
    status: StringProperty(default="选择头部与身体 PMX，点击 2 一键生成。")
    last_output: StringProperty(subtype="FILE_PATH")
    preview_collection: PointerProperty(type=bpy.types.Collection)


class MMDT_UL_materials(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        split = layout.split(factor=0.68)
        split.label(text=f"{item.name} ({item.fraction:.0%})")
        split.prop(item, "mode", text="")


class MMDT_OT_pick(bpy.types.Operator, ImportHelper):
    bl_idname = "mmd_transplant.pick"
    bl_label = "选择 PMX 模型"
    filename_ext = ".pmx"
    filter_glob: StringProperty(default="*.pmx", options={"HIDDEN"})
    target: EnumProperty(items=[("HEAD", "头部", ""), ("BODY", "身体", "")])

    def invoke(self, context, event):
        current = context.scene.mmd_transplant.head_path if self.target == "HEAD" else context.scene.mmd_transplant.body_path
        self.filepath = current or str(Path.home()) + "/"
        return super().invoke(context, event)

    def execute(self, context):
        setattr(context.scene.mmd_transplant, "head_path" if self.target == "HEAD" else "body_path", self.filepath)
        return {"FINISHED"}


class MMDT_OT_analyze(bpy.types.Operator):
    bl_idname = "mmd_transplant.analyze"
    bl_label = "1. 分析头部范围"
    bl_description = "识别头骨骼、估算比例，并列出每个材质的头部面比例；不修改模型"

    def execute(self, context):
        s = context.scene.mmd_transplant
        try:
            _, head, body = load_models(s)
            plan = analyze(head, body, options_from(s))
            for collection, stats in ((s.head_materials, plan["head_materials"]), (s.body_materials, plan["body_materials"])):
                previous = {x.index: x.mode for x in collection}
                collection.clear()
                for stat in stats:
                    item = collection.add()
                    item.index, item.name = stat["index"], stat["name"]
                    item.fraction = stat["head_faces"] / max(1, stat["faces"])
                    item.mode = previous.get(item.index, "AUTO")
            s.status = f"头部 {sum(plan['head_faces']):,} 面；移除旧头 {sum(plan['body_head_faces']):,} 面；比例 {plan['scale']:.3f}"
            for warning in plan["warnings"]:
                self.report({"WARNING"}, warning)
            self.report({"INFO"}, s.status)
            return {"FINISHED"}
        except Exception as exc:
            s.status = str(exc)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}


def _remove_preview(collection):
    if collection is None:
        return
    for obj in list(collection.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.collections.remove(collection)


class MMDT_OT_build(bpy.types.Operator):
    bl_idname = "mmd_transplant.build"
    bl_label = "2. 一键生成换头模型"
    bl_description = "自动分析、匹配比例、贴合颈部并过渡肤色，在新集合生成可编辑模型"
    bl_options = {"REGISTER", "UNDO"}
    export: BoolProperty(default=False, options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        return context.scene is not None

    def execute(self, context):
        s = context.scene.mmd_transplant
        old_selection = list(context.selected_objects)
        old_active = context.view_layer.objects.active
        old_mode = old_active.mode if old_active else 'OBJECT'
        before = set(bpy.data.objects)
        new_collection = None
        try:
            pmx, head, body = load_models(s)
            result, report = (transplant(pmx, head, body, options_from(s)) if s.manual_mode
                              else transplant_automatic(pmx, head, body))
            if self.export:
                if not s.output_dir.strip():
                    raise TransplantError("请设置输出文件夹。")
                base = Path(bpy.path.abspath(s.output_dir)).expanduser().resolve()
                folder = base / ("transplant_" + uuid.uuid4().hex[:10])
                folder.mkdir(parents=True)
            else:
                # Keep files for image reloads and reopening saved .blend files.
                folder = Path(bpy.utils.user_resource("DATAFILES", path="mmd_transplant", create=True)) / uuid.uuid4().hex
                folder.mkdir(parents=True)
            from .skin_blender import prepare_bridge_skin, mark_skin_materials
            report['skin_color'] = prepare_bridge_skin(pmx, result, report, folder,
                                                       (s.skin_strength if s.match_skin else 0) if s.manual_mode else 1)
            if report['skin_color']['status'] == 'skipped':
                report['warnings'].append('肤色过渡未应用：'+report['skin_color']['reason'])
            missing = bundle_textures(result, folder)
            report["warnings"].extend(f"缺失贴图：{path}" for path in missing)
            report['quality'] = assess_result(report, report['skin_color'], missing)
            filepath = folder / "model.pmx"
            uv_count = max((len(v.additional_uvs) for v in result.vertices), default=0)
            pmx.save(str(filepath), result, add_uv_count=uv_count)
            validate(pmx.load(str(filepath)))
            for key in ("body_bone_map", "head_bone_map", "donor_vertex_map", "body_vertex_map"):
                report.pop(key, None)
            (folder / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            if not self.export:
                if context.object and context.object.mode != 'OBJECT':
                    bpy.ops.object.mode_set(mode='OBJECT')
                imported = bpy.ops.mmd_tools.import_model(filepath=str(filepath), scale=s.import_scale if s.manual_mode else .08,
                    clean_model=False, remove_doubles=False, save_log=False, log_level="ERROR")
                if "FINISHED" not in imported:
                    raise TransplantError("mmd_tools 导入失败；已生成文件保留在 " + str(filepath))
                created = set(bpy.data.objects) - before
                new_collection = bpy.data.collections.new("MMD 换头预览")
                context.scene.collection.children.link(new_collection)
                from .physics_blender import move_to_preview
                move_to_preview(created, new_collection, context.scene)
                mark_skin_materials(created, scene=context.scene)
                for obj in context.selected_objects:
                    obj.select_set(False)
                armatures = [o for o in created if o.type == "ARMATURE"]
                if armatures:
                    armatures[0].hide_set(False)
                    armatures[0].show_in_front = True
                meshes = [o for o in created if o.type == 'MESH' and any(m.type == 'ARMATURE' for m in o.modifiers)]
                primary = max(meshes, key=lambda o: len(o.data.vertices), default=None)
                if primary:
                    primary.hide_set(False)
                    primary.select_set(True)
                    context.view_layer.objects.active = primary
                if context.screen is not None:
                    for area in context.screen.areas:
                        if area.type == "VIEW_3D":
                            space = area.spaces.active
                            space.overlay.show_overlays = True
                            space.overlay.show_bones = True
                            # Add the result to an existing local view without
                            # isolating the rest of the user's scene ourselves.
                            if space.local_view:
                                for obj in created:
                                    obj.local_view_set(space, True)
                    if primary and context.area and context.area.type == 'VIEW_3D':
                        region = next((r for r in context.area.regions if r.type == 'WINDOW'), None)
                        if region:
                            with context.temp_override(region=region):
                                bpy.ops.view3d.view_selected(use_all_regions=False)
                _remove_preview(s.preview_collection)
                s.preview_collection = new_collection
            s.last_output = str(filepath)
            s.status = ("已导出：" if self.export else "已生成预览：") + f"{report['faces']:,} 面 / {report['morphs']} 表情"
            if report['neck_fit']['status']=='fitted':
                s.status += "；颈部已贴合"
            elif report['neck_fit']['status']=='bridged':
                s.status += "；颈部已桥接"
            elif report['neck_fit']['status']=='skipped':
                s.status += "；颈部未自动贴合"
            if report['skin_color']['status']=='matched':
                s.status += "；颈部肤色已过渡"
            if report.get('automatic_alignment', {}).get('applied'):
                s.status += '；头部位置已自动校正'
            if report['quality']['issues']:
                s.status += '；需检查：'+'、'.join(report['quality']['issues'])
            self.report({"INFO"}, str(filepath))
            for warning in report["warnings"][:5]:
                self.report({"WARNING"}, warning)
            return {"FINISHED"}
        except Exception as exc:
            traceback.print_exc()
            for obj in set(bpy.data.objects) - before:
                bpy.data.objects.remove(obj, do_unlink=True)
            if new_collection and new_collection.name in bpy.data.collections:
                bpy.data.collections.remove(new_collection)
            for obj in old_selection:
                if obj.name in bpy.data.objects:
                    obj.select_set(True)
            if old_active and old_active.name in bpy.data.objects:
                context.view_layer.objects.active = old_active
                if old_mode != 'OBJECT' and old_active.mode != old_mode:
                    try:
                        bpy.ops.object.mode_set(mode=old_mode)
                    except RuntimeError:
                        pass
            s.status = str(exc)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}


class MMDT_OT_clear(bpy.types.Operator):
    bl_idname = "mmd_transplant.clear_preview"
    bl_label = "移除预览"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        s = context.scene.mmd_transplant
        _remove_preview(s.preview_collection)
        s.preview_collection = None
        s.status = "预览已移除；生成的 PMX 和贴图保留。"
        return {"FINISHED"}


class MMDT_OT_physics_repair(bpy.types.Operator):
    bl_idname = 'mmd_transplant.repair_physics'
    bl_label = '修复已有模型物理连接'
    bl_description = '将预览中的刚体和关节重新加入物理世界；保留模型、动作及刚体参数'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.scene.mmd_transplant.preview_collection is not None

    def execute(self, context):
        from .physics_blender import repair_membership
        s = context.scene.mmd_transplant
        try:
            result = repair_membership(s.preview_collection.objects, context.scene)
            if not result['bodies']:
                self.report({'WARNING'}, '当前预览没有刚体；请开启头部物理并重新生成。')
                return {'CANCELLED'}
            s.status = f"物理连接已检查：{result['bodies']} 个刚体 / {result['joints']} 个关节；请从起始帧构建并播放物理"
            if context.scene.rigidbody_world.point_cache.is_baked:
                self.report({'WARNING'}, '当前物理缓存已烘焙，请清除旧缓存并重新烘焙。')
            self.report({'INFO'}, f"已恢复 {result['bodies_linked']} 个刚体和 {result['joints_linked']} 个关节连接")
            return {'FINISHED'}
        except RuntimeError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}


class MMDT_OT_fit_neck(bpy.types.Operator):
    bl_idname = "mmd_transplant.fit_existing_neck"
    bl_label = "修复颈部连接"
    bl_description = "识别颈部边界并补面、重新桥接或局部贴合；保留权重和表情，支持撤销"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls,context):
        return context.mode=='OBJECT' and (context.scene.mmd_transplant.preview_collection is not None
            or context.object is not None and context.object.type=='MESH' and context.object.find_armature() is not None)

    def execute(self,context):
        from .neck_blender import repair_existing
        s=context.scene.mmd_transplant
        active=context.object
        meshes=([active] if active is not None and active.type=='MESH' and active.find_armature() is not None
                else [o for o in s.preview_collection.objects if o.type=='MESH' and o.find_armature() is not None] if s.preview_collection else [])
        if not meshes:
            self.report({'ERROR'},'预览集合中没有绑定骨架的角色网格。')
            return {'CANCELLED'}
        try:
            mesh=max(meshes,key=lambda o:len(o.data.vertices))
            report=repair_existing(mesh)
            if report['status'] in ('bridged','rebridged'):
                from .skin_blender import mark_skin_materials, match_existing_bridge
                mark_skin_materials([mesh],scene=context.scene)
                s.status=f"颈部已{'重新桥接' if report['status']=='rebridged' else '补面连接'}：{report['bridge_faces']} 个连接面；权重与表情已保留"
                if not s.manual_mode or s.match_skin:
                    try:
                        folder=Path(bpy.utils.user_resource('DATAFILES',path='mmd_transplant',create=True))/uuid.uuid4().hex
                        skin=match_existing_bridge(mesh,folder,s.skin_strength if s.manual_mode else 1.0)
                        if skin['status']=='matched':s.status+='；肤色过渡已应用'
                        elif skin['status']=='skipped':s.status+='；肤色未匹配：'+skin.get('reason','请检查材质')
                    except (ValueError,RuntimeError,OSError) as exc:
                        s.status+='；肤色未匹配：'+str(exc)
                self.report({'INFO'},s.status)
                return {'FINISHED'}
            s.status=("颈部已有连接面，边界已闭合，无需重复桥接" if report['status']=='already_bridged'
                      else f"颈部已贴合，无需再次调整：{report['rim_vertices']} 对边界点" if report['status']=='already_fitted'
                      else f"颈部贴合：{report['rim_vertices']} 对边界点；{report['affected_vertices']} 个局部顶点")
            self.report({'INFO'},s.status)
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'},str(exc))
            return {'CANCELLED'}


class MMDT_OT_skin(bpy.types.Operator):
    bl_idname = 'mmd_transplant.match_neck_skin'
    bl_label = '修复当前颈部材质'
    bl_description = '修正颈部连接带皮肤类型；可采样两端生成肤色过渡。保留几何、骨骼和表情，支持撤销'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and (context.scene.mmd_transplant.preview_collection is not None
                                            or context.object is not None and context.object.type == 'MESH')

    def execute(self, context):
        from .skin_blender import is_bridge, match_existing_bridge, mark_skin_materials
        s = context.scene.mmd_transplant
        active = context.object
        has_bridge = lambda o: o.type == 'MESH' and any(is_bridge(o.material_slots[p.material_index].material) for p in o.data.polygons)
        meshes = ([active] if active is not None and has_bridge(active)
                  else list(s.preview_collection.objects) if s.preview_collection else [])
        meshes = [o for o in meshes if o.type == 'MESH'
                  and any(is_bridge(slot.material) for slot in o.material_slots)]
        if not meshes:
            self.report({'ERROR'}, '没有连接带可修复材质。若颈部有裂缝，请先点击“修复颈部连接”；已贴合的颈部可能不需要连接带。')
            return {'CANCELLED'}
        try:
            mesh = max(meshes, key=lambda o: len(o.data.vertices))
            mark_skin_materials([mesh], scene=context.scene)
            report = {'status': 'off'}
            if s.match_skin and s.skin_strength > 0:
                folder = Path(bpy.utils.user_resource('DATAFILES', path='mmd_transplant', create=True)) / uuid.uuid4().hex
                try:
                    report = match_existing_bridge(mesh, folder, s.skin_strength)
                except (ValueError, RuntimeError, OSError) as exc:
                    report = {'status': 'skipped', 'reason': str(exc)}
                    self.report({'WARNING'}, '肤色过渡未应用：'+str(exc))
            s.status = '颈部连接材质已标记为 SKIN'+('；肤色过渡已应用' if report['status'] == 'matched' else '')
            if report['status'] == 'skipped':
                s.status += '；肤色未自动匹配'
            warnings = {slot.material.get('mmd_transplant_shader_warning') for slot in mesh.material_slots if slot.material}
            for warning in warnings - {None}:
                self.report({'WARNING'}, warning)
            self.report({'INFO'}, s.status)
            return {'FINISHED'}
        except (ValueError, RuntimeError, OSError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}


class MMDT_PT_panel(bpy.types.Panel):
    bl_label = "MMD 自动换头"
    bl_idname = "MMDT_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "MMD 换头"

    def draw(self, context):
        s = context.scene.mmd_transplant
        layout = self.layout
        layout.use_property_split = False
        for label, prop, target in (("A · 取头的模型", "head_path", "HEAD"), ("B · 保留身体的模型", "body_path", "BODY")):
            layout.label(text=label)
            row = layout.row(align=True)
            row.prop(s, prop, text="")
            row.operator("mmd_transplant.pick", text="", icon="FILE_FOLDER").target = target
        row = layout.row(align=True)
        row.scale_y = 1.8
        row.operator("mmd_transplant.build", icon="MOD_ARMATURE")
        layout.label(text='手动模式：按已设置的参数生成' if s.manual_mode else '自动分析 · 比例匹配 · 颈部连接 · 肤色过渡')
        if MMDT_OT_fit_neck.poll(context):
            layout.operator('mmd_transplant.fit_existing_neck', icon='MOD_SMOOTH')
        if s.preview_collection:
            layout.operator('mmd_transplant.repair_physics', icon='PHYSICS')
        box = layout.box()
        for index, line in enumerate(s.status.split('；')):
            box.label(text=line, icon='INFO' if index == 0 else 'NONE')
        box = layout.box()
        box.prop(s, 'adjustments_open', icon='TRIA_DOWN' if s.adjustments_open else 'TRIA_RIGHT', emboss=False)
        if s.adjustments_open:
            box.prop(s, 'manual_mode')
            if not s.manual_mode:
                box.label(text='自动模式忽略旧的微调与材质覆盖')
            controls = box.column()
            controls.enabled = s.manual_mode
            controls.operator('mmd_transplant.analyze', icon='VIEWZOOM')
            for prop in ('auto_scale', 'scale', 'offset', 'physics', 'fit_neck', 'match_skin', 'skin_strength'):
                controls.prop(s, prop)
            controls.prop(s, 'materials_open', icon='TRIA_DOWN' if s.materials_open else 'TRIA_RIGHT', emboss=False)
            if s.materials_open:
                controls.label(text='头源：包含头部；身源：移除旧头')
                controls.template_list('MMDT_UL_materials', 'head', s, 'head_materials', s, 'head_material_index', rows=5)
                controls.template_list('MMDT_UL_materials', 'body', s, 'body_materials', s, 'body_material_index', rows=5)
            controls.prop(s, 'advanced_open', icon='TRIA_DOWN' if s.advanced_open else 'TRIA_RIGHT', emboss=False)
            if s.advanced_open:
                for prop in ('head_bone', 'body_head_bone', 'extra_bones', 'threshold', 'import_scale'):
                    controls.prop(s, prop)
            box.separator()
            box.operator('mmd_transplant.fit_existing_neck', icon='MOD_SMOOTH')
            box.operator('mmd_transplant.match_neck_skin', icon='MATERIAL')
            box.operator('mmd_transplant.clear_preview', icon='TRASH')
        box = layout.box()
        box.prop(s, 'export_open', icon='TRIA_DOWN' if s.export_open else 'TRIA_RIGHT', emboss=False)
        if s.export_open:
            box.prop(s, 'output_dir')
            box.operator('mmd_transplant.build', text='3. 导出 PMX + 贴图', icon='EXPORT').export = True
            if s.last_output:
                box.prop(s, 'last_output', text='结果文件')
        layout.label(text='重新生成会替换预览，请先保存手动修改')


CLASSES = (MMDT_Material, MMDT_Settings, MMDT_UL_materials, MMDT_OT_pick, MMDT_OT_analyze, MMDT_OT_build, MMDT_OT_clear, MMDT_OT_physics_repair, MMDT_OT_fit_neck, MMDT_OT_skin, MMDT_PT_panel)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.mmd_transplant = PointerProperty(type=MMDT_Settings)


def unregister():
    del bpy.types.Scene.mmd_transplant
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
