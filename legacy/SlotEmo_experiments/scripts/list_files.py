"""列出SlotEmo_experiments目录结构"""
import os
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

root = str(Path(__file__).resolve().parents[1])

print(f'根目录: {root}')
print('=' * 70)

for dirpath, dirnames, filenames in os.walk(root):
    if '__pycache__' in dirpath:
        continue

    rel = os.path.relpath(dirpath, root)
    depth = 0 if rel == '.' else rel.count(os.sep) + 1

    if rel == '.':
        print(f'\n[{root}]')
    else:
        indent = '  ' * depth
        print(f'\n{indent}{os.path.basename(dirpath)}/')

    for f in sorted(filenames):
        full = os.path.join(dirpath, f)
        size = os.path.getsize(full)
        if size > 1024 * 1024:
            size_str = f'{size/(1024*1024):.1f}MB'
        elif size > 1024:
            size_str = f'{size//1024}KB'
        else:
            size_str = f'{size}B'
        indent = '  ' * depth
        print(f'{indent}  - {f:<40} {size_str}')
