"""Export frozen 3D weights and a CPU runtime bundle; Python/CUDA stay on the training machine."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tmp/deep-runtime'),str(ROOT/'tmp/onnx-runtime'),str(ROOT/'tmp/learning-runtime')]
import numpy as np
import torch
import onnx
import onnxruntime as ort
from rescue_volume_network import build_network
from rescue_local_features import evidence,MINIMUM


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--tracks',required=True)
    p.add_argument('--name',required=True);p.add_argument('--map',required=True);p.add_argument('--bias',type=float,default=0)
    p.add_argument('--navigation-model',help='Optional independently frozen 74-channel exploration model');a=p.parse_args()
    torch.set_num_threads(2);checkpoint=ROOT/a.checkpoint;out=ROOT/'models'/a.name
    if out.exists():raise ValueError('Use a new immutable export name')
    out.mkdir(parents=True);state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if state.get('tiled',False):raise ValueError('Tiled checkpoints need a tiled runtime export; this exporter handles full-volume models only')
    channels=state.get('channels',2)
    net=build_network(state);net.load_state_dict(state['state']);net.eval()
    source=ROOT/a.map;data=json.loads(source.read_bytes());grid=evidence(data['cells']).transpose(3,0,1,2)
    probe=None
    if channels>2:
        from rescue_range_moments import input_grid
        grid=input_grid(dict(map=source),channels);keys=np.argwhere(np.any(grid[2:]!=0,axis=0))
        data['range_moments']=[dict(key=(k+MINIMUM).tolist(),features=grid[2:,k[0],k[1],k[2]].tolist()) for k in keys]
        probe=ROOT/'tmp'/(a.name+'-probe-map.json');probe.write_text(json.dumps(data,separators=(',',':'))+'\n')
    grid=np.pad(grid,((0,0),(0,0),(0,3),(0,2)));tensor=torch.tensor(grid).unsqueeze(0)
    file=out/'volume.onnx'
    torch.onnx.export(net,tensor,file,input_names=['observations'],output_names=['logits'],opset_version=17,dynamo=False)
    onnx.checker.check_model(str(file))
    options=ort.SessionOptions();options.intra_op_num_threads=2
    session=ort.InferenceSession(str(file),sess_options=options,providers=['CPUExecutionProvider'])
    with torch.no_grad():expected=net(tensor).numpy()
    actual=session.run(None,{'observations':tensor.numpy()})[0]
    p0=1/(1+np.exp(-np.clip(expected+a.bias,-40,40)));p1=1/(1+np.exp(-np.clip(actual+a.bias,-40,40)))
    error=float(np.max(np.abs(p0-p1)));disagree=int(np.sum((p0>.6)!=(p1>.6)))
    # Different FP32 convolution/reduction kernels may round differently. Decisions must
    # agree on the probe; acceptance metrics are subsequently measured with the exported engine.
    if error>1e-3 or disagree:raise AssertionError(f'ONNX export parity failed: {error}/{disagree}')
    model=json.loads((ROOT/a.tracks).read_text(encoding='utf-8'));model.update(geometry_trees=[],geometry_feature_map='',geometry_bias=0,
            volume_file='volume.onnx',volume_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),volume_logit_bias=a.bias,volume_channels=channels)
    model['model_id']=a.name
    model['range_tsdf_v1']=state.get('range_tsdf_v1',False)
    model['data_sha256']=hashlib.sha256(checkpoint.read_bytes()+(ROOT/a.tracks).read_bytes()+(checkpoint.parent/'development.json').read_bytes()).hexdigest()
    for key in ['plane_registration_v1','axis_plane_registration_v2']:
        if model.get(key,False)!=state.get(key,False):raise ValueError('Geometry and person heads must use the same pose registration')
    # Never silently inherit an exploration network from another export.
    for key in ['navigation_file','navigation_sha256','navigation_channels','navigation_logit_bias']:model.pop(key,None)
    navigation=None
    if a.navigation_model:
        navigation_path=ROOT/a.navigation_model;navigation=json.loads(navigation_path.read_bytes())
        if navigation.get('volume_file')!='volume.onnx' or navigation.get('volume_channels')!=74 or navigation.get('range_tsdf_v1',False) or navigation.get('prototypes') or navigation.get('geometry_trees'):raise ValueError('Frozen sensor-only 74-channel exploration volume required')
        for key in ['plane_registration_v1','axis_plane_registration_v2','learned_navigation_v1','scout_inferred_routes_v1','footprint_safe_routes_v1']:
            if navigation.get(key,False)!=model.get(key,False):raise ValueError('Exploration feature or route contract differs: '+key)
        navigation_file=navigation_path.parent/'volume.onnx'
        if hashlib.sha256(navigation_file.read_bytes()).hexdigest()!=navigation['volume_sha256']:raise ValueError('Frozen exploration weights changed')
        shutil.copy2(navigation_file,out/'navigation.onnx')
        model.update(navigation_file='navigation.onnx',navigation_sha256=navigation['volume_sha256'],navigation_channels=74,navigation_logit_bias=navigation.get('volume_logit_bias',0))
        model['data_sha256']=hashlib.sha256(model['data_sha256'].encode()+navigation_path.read_bytes()).hexdigest()
        model['train_maps']=sorted(set(model['train_maps'])|set(navigation['train_maps'])|set(navigation.get('calibration_train_maps',[])))
        model['valid_maps']=sorted(set(model['valid_maps'])|set(navigation['valid_maps']))
    for tree in model['track_trees']:
        for node in tree['nodes']:
            if node['feature'] is not None:node['value']=0.0  # internal-node training statistics are never predictions
    development=json.loads((checkpoint.parent/'development.json').read_text(encoding='utf-8'))
    model['train_maps']=sorted(set(model['train_maps'])|set(state.get('pretraining_train_maps',[]))|{r['id'] for r in development['train_audit'] if not r['held']})
    model['valid_maps']=sorted(set(model['valid_maps'])|{r['id'] for r in development['train_audit'] if r['held']})
    overlap=set(model['train_maps'])&set(model['valid_maps'])
    from train_rescue_generic import cases
    registered_train={row['id'] for row in cases()}|set(state.get('pretraining_train_maps',[]))|{row['id'] for row in development['train_audit'] if not row['source'].startswith(('valid-','test-'))}
    if not overlap<=registered_train:raise ValueError('A formal validation world was used for fitting')
    # Different heads may use different development folds of the registered TRAIN pool.
    # Once any head fits a TRAIN world it cannot serve as validation of the composite model.
    model['valid_maps']=sorted(set(model['valid_maps'])-overlap)
    (out/'model.json').write_text(json.dumps(model,separators=(',',':'))+'\n',encoding='utf-8')
    runtime=ROOT/'tmp/onnx-runtime/onnxruntime'
    for name in ['onnxruntime.dll','onnxruntime_providers_shared.dll']:shutil.copy2(runtime/'capi'/name,out/name)
    shutil.copy2(runtime/'LICENSE',out/'ONNXRUNTIME-LICENSE.txt');shutil.copy2(runtime/'ThirdPartyNotices.txt',out/'ONNXRUNTIME-ThirdPartyNotices.txt')
    from windows_runtime_dependencies import bundle
    crt=bundle(out)
    report=dict(status='pass',checkpoint=a.checkpoint,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                onnx_sha256=model['volume_sha256'],map=a.map,maximum_probability_error=error,class_disagreements=disagree,
                cpu_runtime=ort.__version__,requires_python=False,requires_cuda=False,epoch=state['epoch'],numerical_tolerance=1e-3,
                channels=channels,architecture=state.get('architecture','volume-unet-v1'),probe_map=str(probe.relative_to(ROOT)) if probe else a.map)
    report['microsoft_crt']=crt
    report['retired_train_development_holdouts']=sorted(overlap)
    if navigation is not None:report['navigation']=dict(source_model=a.navigation_model,source_model_sha256=hashlib.sha256((ROOT/a.navigation_model).read_bytes()).hexdigest(),onnx_sha256=model['navigation_sha256'],channels=74,purpose='Frozen sensor-only exploration; independent of updated map decoder')
    (out/'export.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
