"""Audit staged blobs for private artifacts and machine-specific paths."""
from pathlib import PurePosixPath
import re
import subprocess
import sys


def git(*args):
    return subprocess.check_output(['git',*args])


names=git('ls-files','-z').decode().split('\0')
issues=[]
for name in filter(None,names):
    path=PurePosixPath(name)
    allowed=(name in {'.gitignore','LICENSE','README.md','AGENTS.md'}
             or len(path.parts)==2 and path.parts[0] in {'mmd_transplant','scripts','tests'}
             and path.suffix in {'.py','.toml'})
    if not allowed:
        issues.append((name,'unexpected file type or directory'))
        continue
    blob=git('show',':'+name)
    try:
        text=blob.decode('utf-8')
    except UnicodeDecodeError:
        issues.append((name,'binary content'))
        continue
    if '\0' in text:
        issues.append((name,'binary content'))
    if re.search(r'/(?:Users|Volumes|home)/[^\s\"\']+|[A-Z]:\\Users\\',text):
        issues.append((name,'machine-specific path'))
if issues:
    for name,reason in issues:print(f'BLOCKED {name}: {reason}')
    sys.exit(1)
print(f'Public staged tree verified: {len(list(filter(None,names)))} text source files. ')
