"""Copy complete sensor-only VALID tensors into one explicit, immutable pool."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def merge(sources,output,expected):
    output=output.resolve()
    if not output.is_relative_to(ROOT/'experiments/raw/rescue') or output.exists():raise ValueError('Fresh workspace preparation output required')
    rows=[];receipts=[];seen=set();profile=None;lineage=None
    for source in sources:
        root=(ROOT/source).resolve();manifest=root/'manifest.json';record=json.loads(manifest.read_bytes())
        if record['status']!='complete' or record['test_opened']:raise ValueError('Complete development preparation required')
        current=ROOT/record['lineage']['model']
        if sha(current)!=record['lineage']['model_sha256']:raise ValueError('Preparation profile changed')
        if profile is not None and sha(current)!=profile:raise ValueError('VALID feature contracts differ')
        profile=sha(current);lineage=record['lineage']
        receipts.append(dict(manifest=manifest.relative_to(ROOT).as_posix(),sha256=sha(manifest)))
        for split in ['valid-known','valid-unknown']:
            for path in sorted((root/split).glob('*/complete.json')):
                identifier=path.parent.name
                if identifier not in record['case_ids'] or identifier in seen:raise ValueError('Unregistered or duplicate VALID tensor')
                seen.add(identifier);rows.append((split,path))
    if len(rows)!=expected:raise ValueError('Exact registered VALID count required')
    output.mkdir(parents=True)
    copies=[]
    for split,path in rows:
        target=output/split/path.parent.name
        shutil.copytree(path.parent,target)
        original={p.relative_to(path.parent).as_posix():sha(p) for p in path.parent.rglob('*') if p.is_file()}
        copied={p.relative_to(target).as_posix():sha(p) for p in target.rglob('*') if p.is_file()}
        if original!=copied:raise ValueError('Copied sensor tensor differs')
        copies.append(dict(id=path.parent.name,split=split,source=path.relative_to(ROOT).as_posix(),source_sha256=sha(path),files=original))
    result=dict(status='complete',test_opened=False,cases=len(rows),case_ids=sorted(seen),
                lineage=dict(lineage,composition='Byte-exact union of explicit VALID-only sensor preparations',sources=receipts),copies=copies)
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return dict(status='complete',cases=len(rows),truth_to_server=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',action='append',required=True);p.add_argument('--output',required=True);p.add_argument('--expected-cases',required=True,type=int);a=p.parse_args()
    print(json.dumps(merge(a.source,ROOT/a.output,a.expected_cases)))


if __name__=='__main__':main()
