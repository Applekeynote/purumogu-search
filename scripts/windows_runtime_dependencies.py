"""App-local release CRT dependencies from the installed licensed Visual Studio redist.

No system-directory DLLs are copied. Redistributable files retain Microsoft's
license, separate from the application's MIT license.
"""
import hashlib
from pathlib import Path
import shutil
import struct


def imports(path):
    data=path.read_bytes();pe=struct.unpack_from('<I',data,0x3c)[0]
    sections=struct.unpack_from('<H',data,pe+6)[0];optional=pe+24
    length=struct.unpack_from('<H',data,pe+20)[0]
    table=[struct.unpack_from('<8sIIIIIIHHI',data,optional+length+i*40) for i in range(sections)]
    def offset(rva):
        for s in table:
            if s[2]<=rva<s[2]+max(s[1],s[3]):return s[4]+rva-s[2]
        raise ValueError('Invalid PE import address')
    if struct.unpack_from('<H',data,optional)[0]!=0x20b:raise ValueError('Windows x64 binary required')
    rva,_=struct.unpack_from('<II',data,optional+120)
    if not rva:return []
    cursor=offset(rva);result=[]
    while True:
        descriptor=struct.unpack_from('<IIIII',data,cursor)
        if not any(descriptor):break
        start=offset(descriptor[3]);result.append(data[start:data.index(b'\0',start)].decode('ascii'))
        cursor+=20
    return result


def bundle(directory):
    roots=list(Path('C:/Program Files/Microsoft Visual Studio').glob('*/*/VC/Redist/MSVC/*/x64/Microsoft.VC*.CRT'))
    if not roots:raise RuntimeError('Install licensed Visual Studio C++ Build Tools with its x64 release CRT redist')
    source=max(roots,key=lambda p:tuple(int(v) for v in p.parents[1].name.split('.')))
    available={p.name.lower():p for p in source.glob('*.dll')}
    required={};queue=[directory/'onnxruntime.dll',directory/'onnxruntime_providers_shared.dll']
    while queue:
        for name in imports(queue.pop()):
            key=name.lower()
            if not key.startswith(('msvcp140','vcruntime140','concrt140')) or key in required:continue
            if key not in available:raise RuntimeError('Required redistributable missing: '+name)
            required[key]=available[key];queue.append(available[key])
    for name,path in required.items():shutil.copy2(path,directory/name)
    (directory/'MICROSOFT-CRT-NOTICE.txt').write_text(
        'Microsoft Visual C++ Runtime / x64 release redistributable\n'
        'Copyright (c) Microsoft Corporation. All rights reserved.\n'
        'These binaries are distributed under Microsoft Visual Studio license terms, not the project MIT license.\n'
        'Source: installed Visual Studio release redist '+source.parents[1].name+'\n'
        'https://learn.microsoft.com/cpp/windows/redistributing-visual-cpp-files\n'
        'https://visualstudio.microsoft.com/license-terms/\n',encoding='utf-8')
    return dict(version=source.parents[1].name,files={name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in sorted(required)})
