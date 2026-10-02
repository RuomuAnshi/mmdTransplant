"""Run Blender --background --python this file, without --factory-startup.

Checks the saved preferences in a fresh process without changing them.
"""
import bpy,importlib,json
from pathlib import Path
import tomllib
name='bl_ext.user_default.mmd_transplant'
assert name in bpy.context.preferences.addons,'Extension is not saved as enabled'
module=importlib.import_module(name)
manifest=tomllib.loads((Path(module.__file__).parent/'blender_manifest.toml').read_text())
assert manifest['version']==tomllib.loads((Path(__file__).resolve().parents[1]/'mmd_transplant/blender_manifest.toml').read_text())['version']
assert hasattr(bpy.types.Scene,'mmd_transplant')
assert bpy.context.scene.mmd_transplant.fit_neck
print('INSTALLED_ADDON_VERIFIED',json.dumps({'module':name,'version':manifest['version'],'enabled':True}))
