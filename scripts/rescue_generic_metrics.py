"""Acceptance aggregates shared by both learners; voxel definitions stay unchanged."""
import numpy as np


def aggregate(group):
    tp=sum(r['matched'] for r in group);fp=sum(r['false_alarms'] for r in group);people=sum(r['people'] for r in group)
    return dict(cases=len(group),macro_iou=float(np.mean([r['iou'] for r in group])),
                micro_iou=sum(r['intersection'] for r in group)/max(1,sum(r['union'] for r in group)),
                minimum_case_iou=min(r['iou'] for r in group),human_precision=tp/(tp+fp) if tp+fp else None,
                human_recall=tp/people if people else None,matched=tp,false_alarms=fp,people=people)


def tables(rows):
    result=dict(summary=aggregate(rows),
                by_node_count={str(n):aggregate([r for r in rows if r['nodes']==n]) for n in sorted({r['nodes'] for r in rows})},
                by_area={s:aggregate([r for r in rows if r['scene']==s]) for s in sorted({r['scene'] for r in rows})})
    result['all_95']=all(v['macro_iou']>=.95 and all(v[k] is None or v[k]>=.95 for k in ['human_precision','human_recall'])
                        and (v['people']>0 or v['false_alarms']==0)
                        for v in [result['summary'],*result['by_node_count'].values(),*result['by_area'].values()])
    return result
