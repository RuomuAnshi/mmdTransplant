from pathlib import Path
import zipfile
import tomllib

root = Path(__file__).resolve().parents[1]
source = root / 'mmd_transplant'
(root / 'dist').mkdir(exist_ok=True)
for legacy in (False, True):
    version=tomllib.loads((source/'blender_manifest.toml').read_text())['version']
    filename = 'mmd_transplant-'+version + ('-legacy' if legacy else '') + '.zip'
    with zipfile.ZipFile(root / 'dist' / filename, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.iterdir()):
            if path.is_file() and not path.name.startswith('.') and path.suffix in ('.py', '.toml'):
                archive.write(path, ('mmd_transplant/' if legacy else '') + path.name)
        archive.write(root / 'LICENSE', ('mmd_transplant/' if legacy else '') + 'LICENSE')
    print(root / 'dist' / filename)
