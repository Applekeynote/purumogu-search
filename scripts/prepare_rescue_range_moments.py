"""Parallel numerical preprocessing, not agent delegation. TEST labels are never opened."""
import argparse
import json
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tmp/learning-runtime'))
from train_rescue_generic import cases
from rescue_range_moments import cached


def work(row):
    cached(row,row.get('covariance',False),row.get('fine',False));return row['id']


def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['train','valid-known','valid-unknown'],required=True)
    p.add_argument('--covariance',action='store_true');p.add_argument('--fine',action='store_true');a=p.parse_args()
    if a.covariance and a.fine:raise ValueError('Choose one feature representation')
    if a.split=='train':rows=list(cases())
    else:
        rows=[dict(id=path.parent.name,map=path.parent/'map-6000.json') for path in
              (ROOT/'experiments/raw/rescue/generalization-v35'/a.split).glob('*/complete.json')]
    for row in rows:row['covariance']=a.covariance;row['fine']=a.fine
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,name in enumerate(pool.map(work,rows),1):
            if i%10==0 or i==len(rows):print(json.dumps(dict(completed=i,total=len(rows),case=name)),flush=True)


if __name__=='__main__':main()
