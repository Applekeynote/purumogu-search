"""Prepare source clips and typeset graphics. Editing/mixing/export happens in Resolve."""
from pathlib import Path
import sys,csv,json,argparse,math
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/'tmp/media-runtime').exists():sys.path.insert(0,str(ROOT/'tmp/media-runtime'))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg

OUT=ROOT/'media/resolve_sources'
PAPER='#f1f6f0';INK='#112a31';MUTED='#60757b';DARK='#11252c';MINT='#42cfa8';AMBER='#edac62'

def font(size,bold=False):return ImageFont.truetype('C:/Windows/Fonts/YuGothB.ttc' if bold else 'C:/Windows/Fonts/YuGothM.ttc',size,index=0)
def text(draw,xy,value,size=36,color=INK,bold=False):draw.text(xy,value,font=font(size,bold),fill=color,spacing=14)
def base(dark=False):
    img=Image.new('RGB',(1920,1080),DARK if dark else PAPER);d=ImageDraw.Draw(img)
    d.rectangle((120,103,195,110),fill=MINT if dark else '#159b80')
    text(d,(120,135),'PURUMOGU SEARCH   /   PHYSICAL SENSING STUDY',24,MUTED if not dark else '#8eaaa9')
    return img,d
def cards():
    OUT.mkdir(parents=True,exist_ok=True)
    img,d=base(True)
    text(d,(120,300),'ぷるもぐサーチ',106,'#f4f7ed',True)
    text(d,(125,475),'群れの反応から、通路の候補を絞る。',48,'#dbe9e1')
    d.line((125,660,1795,660),fill='#365158',width=2)
    text(d,(125,710),'UNITY × RUST',31,MINT,True)
    text(d,(125,775),'物理シミュレーション / 学習・推論 / 次の投入',32,'#c4d5ce')
    text(d,(125,950),'実装・評価  2026.10.27',24,'#8eaaa9')
    img.save(OUT/'01-title.png')

    analysis=json.loads((ROOT/'reports/analysis.json').read_text())
    metrics=json.loads((ROOT/'reports/comparison/metrics.json').read_text())
    img,d=base()
    text(d,(120,220),'固定投入との差は +1ポイント',66,INK,True)
    for x,key,label in [(125,'ghost_flow','GHOST-FLOW'),(735,'fixed','固定投入'),(1345,'random','ランダム')]:
        text(d,(x,365),label,32,MUTED)
        text(d,(x,410),f"{metrics[key]['accuracy']*100:.0f}%",142,'#159b80' if key=='ghost_flow' else INK,True)
    text(d,(125,625),'AI方式の保留率 79%  /  保留しなかった21試行は21正解',31,MUTED)
    y=805;left,right=290,1650
    def sx(v):return left+(v+20)/40*(right-left)
    d.line((left,y,right,y),fill='#bacbc3',width=2)
    for value in [-20,-10,0,10,20]:
        x=sx(value);d.line((x,y-10,x,y+10),fill='#718c86',width=2);text(d,(x-34,y+22),f'{value:+d}' if value else '0',25,MUTED)
    zero=sx(0);d.line((zero,735,zero,865),fill='#b5c6bd',width=2)
    pair=analysis['ghost_minus_fixed_accuracy'];lo,hi=np.array(pair['ci95'])*100;point=pair['mean_difference']*100
    d.line((sx(lo),y,sx(hi),y),fill='#159b80',width=7)
    for v in [lo,hi]:d.line((sx(v),y-18,sx(v),y+18),fill='#159b80',width=4)
    x=sx(point);d.ellipse((x-13,y-13,x+13,y+13),fill='#159b80')
    text(d,(125,935),'ペア比較の95%区間 −12〜+15ポイント  /  未使用50マップ × 2シード',29,INK)
    img.save(OUT/'05-results.png')

    img,d=base(True)
    text(d,(120,225),'未知形状の拒否は未解決',70,'#f4f7ed',True)
    text(d,(125,365),'6 / 8',168,AMBER,True)
    text(d,(760,425),'候補外の形状で誤断定',45,'#f4f7ed',True)
    text(d,(765,505),'2マップ × 欠損4条件\n保留できたのは2試行',35,'#c4d5ce')
    d.line((125,700,1795,700),fill='#365158',width=2)
    text(d,(125,770),'3D地図の完全復元',30,'#8eaaa9');text(d,(125,825),'未実装',41,'#f4f7ed')
    text(d,(760,770),'人由来兆候の検出',30,'#8eaaa9');text(d,(760,825),'未実装',41,'#f4f7ed')
    text(d,(1360,770),'実機・救助性能',30,'#8eaaa9');text(d,(1360,825),'未検証',41,'#f4f7ed')
    text(d,(125,965),'失敗した条件も、ログから追えるようにする。',28,'#c4d5ce')
    img.save(OUT/'06-limits.png')

    img,d=base(True)
    text(d,(125,250),'ぷるもぐサーチ',74,'#f4f7ed',True)
    entries=[('VOICE','音読さん / Ellis'),('MUSIC','Suno AI / Purumogu - Small Traces'),('EDIT & MIX','DaVinci Resolve Studio 21.1'),('SIMULATION','Unity 6000.4.1f1 × Rust 1.94.1')]
    for i,(label,value) in enumerate(entries):
        y=410+i*90;text(d,(125,y),label,24,MINT,True);text(d,(515,y-5),value,33,'#dbe9e1')
    text(d,(125,890),'github.com/Applekeynote/purumogu-search',35,'#f4f7ed',True)
    text(d,(125,960),'コード・固定モデル・実測ログ・PDFを公開',27,'#8eaaa9')
    img.save(OUT/'07-credits.png')
    for name,label,subtitle in [('02-observe','01  /  物理と観測','Unity simulation'),('03-infer','02  /  観測から次の行動へ','GHOST-FLOW'),('04-couple','03  /  親指令と一時連結','別実行・同一条件 / 拡大カメラ')]:
        img=Image.new('RGBA',(1920,1080),(0,0,0,0));d=ImageDraw.Draw(img);d.rectangle((0,0,1920,54),fill=(17,37,44,255));text(d,(80,8),label,27,'#ecf3eb',True);text(d,(1260,11),subtitle,22,'#a5c8bd');img.save(OUT/(name+'.png'))
    (OUT/'README.md').write_text('Source assets for DaVinci Resolve. PNG cards are typeset in code. Source MP4 clips are uncut application recordings, resampled by their saved wall-clock timestamps to 30 fps without motion synthesis. Creative editing, fades, audio normalization, mixing, and final delivery are performed in Resolve.\n',encoding='utf-8')

def clip(shot):
    folder=ROOT/'media/frames'/shot
    with (folder/'timestamps.csv').open() as stream:rows=list(csv.DictReader(stream))
    times=np.array([float(r['wall_time_s']) for r in rows]);indices=np.array([int(r['frame_index']) for r in rows]);steps=np.array([int(r['sim_step']) for r in rows])
    assert len(rows)>500 and np.all(np.diff(times)>0) and np.all(np.diff(steps)>=0)
    duration=times[-1]-times[0];fps=30;total=math.floor(duration*fps)
    writer=imageio_ffmpeg.write_frames(str(OUT/(shot+'.mp4')),(1600,900),fps=fps,codec='libx264',pix_fmt_out='yuv420p',macro_block_size=1,ffmpeg_log_level='error',output_params=['-crf','16','-preset','fast','-movflags','+faststart','-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709'])
    writer.send(None);last=-1;data=None
    for frame in range(total):
        selected=int(np.searchsorted(times,times[0]+frame/fps,side='right')-1);idx=int(indices[max(selected,0)])
        if idx!=last:
            img=Image.open(folder/f'frame-{idx:05d}.png').convert('RGB');data=img.tobytes();last=idx
        writer.send(data)
    writer.close()
    metadata={'source_shot':shot,'capture_frames':len(rows),'source_wall_duration_s':duration,'output_frames':total,'fps':fps,'sim_step_first':int(steps[0]),'sim_step_last':int(steps[-1]),'motion_synthesis':False,'method':'previous captured frame at each 1/30 second wall-clock instant'}
    (OUT/(shot+'-capture.json')).write_text(json.dumps(metadata,indent=2)+'\n');print(metadata)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('shots',nargs='*');args=parser.parse_args();cards()
    for shot in args.shots:clip(shot)
