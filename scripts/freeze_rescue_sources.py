"""Archive reproducible simulator/inference sources without data, accounts, or build caches."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile,ZipInfo,ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    out=(ROOT/a.output).resolve()
    if not out.is_relative_to(ROOT/'experiments'):raise ValueError('Use a registered experiment directory')
    archive=out/'source-snapshot.zip';manifest=out/'source-snapshot.json'
    if archive.exists() or manifest.exists():raise ValueError('Source snapshot is immutable')
    paths=set()
    for pattern in ['unity/Assets/**/*.cs','unity/Assets/**/*.shader','unity/Assets/**/*.cs.meta',
                    'unity/ProjectSettings/*','unity/Packages/*.json',
                    'rust/Cargo.*','rust/crates/*/Cargo.toml','rust/apps/*/Cargo.toml',
                    'rust/crates/*/src/**/*.rs','rust/apps/*/src/**/*.rs',
                    'scripts/*.py','scripts/*.ps1','scripts/requirements*.txt','tests/*.py',
                    'requirements*.txt','docs/rescue-generic-learning.md']:
        paths.update(path for path in ROOT.glob(pattern) if path.is_file())
    out.mkdir(parents=True,exist_ok=True);hashes={}
    with ZipFile(archive,'x',compression=ZIP_DEFLATED,compresslevel=9) as bundle:
        for path in sorted(paths):
            name=path.relative_to(ROOT).as_posix();content=path.read_bytes()
            hashes[name]=hashlib.sha256(content).hexdigest()
            info=ZipInfo(name);info.compress_type=ZIP_DEFLATED
            bundle.writestr(info,content)
    result=dict(sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=hashes,
                purpose='Exact source at registered development acquisition; no sensor labels or final TEST data')
    manifest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(files=len(hashes),archive_sha256=result['sha256'])))


if __name__=='__main__':main()
