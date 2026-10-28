"""Typeset source artwork and resample uncut captures; editing and mixing stay in Resolve."""
from pathlib import Path
import sys, math, csv, json, argparse, random, textwrap
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tmp/media-runtime'))
from PIL import Image,ImageDraw,ImageFont
import numpy as np
import imageio_ffmpeg
OUT=ROOT/'media/concept_film_sources'
DARK='#060b10';INK='#eff4f2';MUTED='#8da3ad';MINT='#73f3d0';AMBER='#ffb33e';ROSE='#ff567e';BLUE='#6fa9ff'

def font(size,bold=False,en=False):
    path='C:/Windows/Fonts/'+('bahnschrift.ttf' if en else 'YuGothB.ttc' if bold else 'YuGothM.ttc')
    return ImageFont.truetype(path,size)
def text(d,xy,s,size=40,color=INK,bold=False,en=False):d.text(xy,s,font=font(size,bold,en),fill=color,spacing=12)
def canvas(alpha=False):
    im=Image.new('RGBA' if alpha else 'RGB',(1920,1080),(0,0,0,0) if alpha else DARK);return im,ImageDraw.Draw(im)
def save(im,name):im.save(OUT/(name+'.png'))
def small(d,chapter,kind='CONCEPT FILM'):
    text(d,(104,64),'PURUMOGU  /  SEARCH',22,MUTED,en=True)
    text(d,(1380,64),chapter,22,MUTED,en=True)
    width=d.textlength(kind,font=font(21))
    d.rounded_rectangle((94,880,min(1850,124+width),924),radius=5,fill=(6,11,16,225))
    text(d,(104,885),kind,21,'#b5c4cb')
def head(d,note,title):text(d,(104,190),note,23,MINT,en=True);text(d,(104,250),title,66,INK,True)

def artwork():
    OUT.mkdir(parents=True,exist_ok=True)
    im,d=canvas(True);d.rectangle((0,0,1920,127),fill=DARK);d.rectangle((0,952,1920,1080),fill=DARK);save(im,'cinema-matte')
    for name,title,note,kind in [
        ('hero-title','ぷるもぐサーチ','DISTRIBUTED RESCUE SENSING','機構イメージ / 全長約7cmの設計モデル'),
        ('size','07','MICRO NODE','機構イメージ / 寸法は設計上の基準'),
        ('coupling','先端でも、球体でも。','RECONFIGURABLE','機構イメージ / 接続と分離の動きを説明'),
        ('drop','36台を、現場へ。','DEPLOY / DISPERSE','実アプリ収録 / E02・36台 / 別実行を編集'),
        ('mapping','観測から、地図が立ち上がる。','FUSE / RECONSTRUCT','実アプリ収録 / 観測と補完を区別 / 別実行を編集'),
        ('candidate','人の気配を、確認候補へ。','PRIORITIZE','実アプリ収録 / 人候補は未確認 / 実災害性能は未検証'),
        ('mission','救助判断へ、手がかりを。','THE MISSION','研究構想 / 実機での救助性能は未検証')]:
        im,d=canvas(True);small(d,note,kind)
        for y in range(128,480):
            alpha=int(140*(1-(y-128)/352)**1.4)
            d.line((0,y,1920,y),fill=(6,11,16,alpha))
        if name=='hero-title':
            text(d,(104,747),title,100,INK,True);text(d,(110,715),note,22,MINT,en=True)
        elif name=='size':
            text(d,(104,200),'全長 約',28,MUTED);text(d,(100,236),'07',184,INK,en=True);text(d,(345,352),'cm',46,MINT,en=True)
            text(d,(104,460),'小さなノードを、多数。',35,INK,True)
        else:head(d,note,title)
        if name in ['mapping','candidate']:
            for x,c,s in [(1180,AMBER,'人候補'),(1390,ROSE,'ガス反応'),(1620,BLUE,'浸水')]:
                d.ellipse((x,845,x+12,857),fill=c);text(d,(x+25,829),s,26,c)
        save(im,'overlay-'+name)
    # Opening: a deliberately schematic field of disconnected rescue sites.
    im,d=canvas();small(d,'01 / THE SEARCH GAP','課題の模式図 / 実測データではありません')
    rng=random.Random(428);pts=[(rng.randint(1030,1760),rng.randint(230,820)) for _ in range(36)]
    for i,(x,y) in enumerate(pts):
        if i and i%4:d.line((pts[i-1],(x,y)),fill='#21333d',width=2)
        c=ROSE if i%5==0 else '#56737c';d.ellipse((x-5,y-5,x+5,y+5),fill=c)
        if i%5==0:d.ellipse((x-24,y-24,x+24,y+24),outline='#6a3041',width=2)
    text(d,(104,275),'救助の手は、',79,INK,True);text(d,(104,390),'限られている。',79,INK,True)
    text(d,(110,570),'同時多発する現場。\n届かない内部情報。',34,MUTED)
    text(d,(110,755),'RESCUE IS FINITE.',32,MINT,en=True);save(im,'opening')
    im,d=canvas(True);small(d,'02 / NOTO 2024','写真：国土交通省 / 国道470号 能越自動車道・被災状況')
    text(d,(104,250),'2024',120,INK,en=True);text(d,(110,400),'能登半島地震',39,INK,True)
    text(d,(110,535),'道が、\n途切れた。',68,INK,True);text(d,(110,750),'到達を阻む、道路の寸断。',27,MUTED);save(im,'overlay-noto')
    im,d=canvas();small(d,'03 / THE FIRST DECISION','課題の模式図 / 実測の救助指令ではありません');head(d,'LIMITED RESOURCES','最初に、どこへ。')
    path=[(980,730),(1120,650),(1300,700),(1470,530),(1730,600)]
    d.line(path,fill='#3b5b65',width=4)
    for i,(x,y) in enumerate(path):
        d.ellipse((x-10,y-10,x+10,y+10),fill=AMBER if i==3 else MINT)
        if i==3:d.ellipse((x-45,y-45,x+45,y+45),outline=AMBER,width=2)
    d.line((1120,650,1230,450,1440,410),fill='#3b5b65',width=3);d.line((1300,700,1420,820,1750,800),fill='#3b5b65',width=3)
    text(d,(110,580),'判断に必要な情報が、\n瓦礫の奥から届かない。',38,MUTED);save(im,'question')
    im,d=canvas();small(d,'04 / COMPLEMENTARY TOOLS','既存手段の強みを示す模式図');head(d,'AERIAL / NARROW PATH','それぞれの強みを、つなぐ。')
    for cx,cy in [(500,560)]:
        for dx,dy in [(-120,-65),(120,-65),(-120,65),(120,65)]:
            d.line((cx,cy,cx+dx,cy+dy),fill=INK,width=5);d.ellipse((cx+dx-65,cy+dy-27,cx+dx+65,cy+dy+27),outline=MINT,width=3)
        d.rounded_rectangle((cx-48,cy-35,cx+48,cy+35),radius=16,outline=INK,width=4)
    for i in range(10):
        x=1120+i*49;y=560+math.sin(i*.65)*50;d.ellipse((x-22,y-22,x+22,y+22),outline=MINT,width=4)
    text(d,(320,720),'上空・飛行できる空間',34,INK,True);text(d,(1100,720),'細い経路の詳細観察',34,INK,True);save(im,'comparison')
    im,d=canvas();small(d,'05 / BRANCHING VOIDS','瓦礫内部の経路を示す模式図');head(d,'BEYOND A SINGLE PATH','分岐の奥に、情報の空白。')
    for row in range(4):
        y=430+row*107
        d.line([(180,y),(520,y),(750,550),(1040,550),(1280,y),(1740,y)],fill='#2c444f',width=4)
        for x in [180,520,1280,1740]:d.ellipse((x-5,y-5,x+5,y+5),fill=MINT if row==1 else '#5c7782')
    d.line((180,537,520,537,750,550,1040,550),fill=MINT,width=5)
    text(d,(1190,740),'同時に、広く、調べるために。',31,MUTED);save(im,'branches')
    for name,value,n in [('stop','止まった。','01'),('contact','接触した。','02'),('disconnect','通信が途切れた。','03')]:
        im,d=canvas();small(d,'07 / SIGNALS FROM FAILURE','反応を手がかりにする探索構想');text(d,(104,208),n,34,ROSE,en=True)
        text(d,(104,370),value,104,INK,True);d.line((110,640,1800,640),fill='#263a45',width=2)
        text(d,(110,706),'失敗や欠損も、内部を知る情報へ。',42,MINT,True);save(im,name)
    im,d=canvas();small(d,'08 / FUSION LOOP','探索構想の模式図 / すべての反応融合が実証済みという意味ではありません');head(d,'FAILURE AS EVIDENCE','失敗・欠損も、手がかりへ。')
    for i,(en,jp) in enumerate([('STOP','停止'),('CONTACT','接触'),('LINK LOSS','通信欠損')]):
        x=104+i*590;d.rectangle((x,440,x+530,660),outline='#2f4955',width=2);text(d,(x+35,470),en,29,MINT,en=True);text(d,(x+35,530),jp,57,INK,True)
    text(d,(110,740),'観測を統合し、見えない空洞・閉塞を推定する。',44,INK,True);save(im,'evidence')
    im,d=canvas();small(d,'10 / OBSERVE · INFER · GUIDE','次の投入先まで判断する探索構想');head(d,'A CLOSED INFORMATION LOOP','次の探索に、観測を返す。')
    for i,(en,jp) in enumerate([('OBSERVE','群れで観測'),('INFER','空間を推定'),('GUIDE','次の投入へ')]):
        x=150+i*590;d.ellipse((x,440,x+350,790),outline='#2c4653',width=2);text(d,(x+60,535),en,34,MINT,en=True);text(d,(x+45,610),jp,38,INK,True)
        if i<2:d.line((x+380,610,x+530,610),fill=MINT,width=3);d.polygon([(x+530,610),(x+514,600),(x+514,620)],fill=MINT)
    save(im,'guidance')
    im,d=canvas(True);small(d,'11 / UNITY × RUST','別実行の収録を並べて編集 / 効果・精度の評価は継続中')
    head(d,'15 SCENARIOS · 24 / 36 / 48 NODES','倒壊・斜面崩壊・浸水を想定。')
    for x,s in [(180,'倒壊'),(790,'斜面崩壊'),(1410,'浸水後の閉塞')]:text(d,(x,770),s,34,INK,True)
    save(im,'overlay-scenarios')
    im,d=canvas(True);small(d,'12 / PURUMOGU SEARCH','CONCEPT FILM / 研究構想とシミュレーション')
    text(d,(104,215),'広く、速く、',72,INK,True);text(d,(104,316),'手がかりを。',72,INK,True)
    text(d,(110,760),'ぷるもぐサーチ',88,INK,True);save(im,'overlay-end')
    im,d=canvas();small(d,'CREDITS / 2026.10.28','シミュレーションによる研究試作 / 実機の救助性能は未検証')
    text(d,(104,210),'PURUMOGU SEARCH',67,INK,en=True)
    for y,label,value in [(370,'VOICE','音読さん / Gray'),(465,'MUSIC','Suno AI / Purumogu — Signals in the Void'),(560,'EDIT & MIX','DaVinci Resolve Studio 21.1'),(655,'SIMULATION','Unity × Rust')]:
        text(d,(110,y),label,24,MINT,en=True);text(d,(465,y-8),value,33,INK)
    text(d,(110,760),'被災写真：国土交通省「国道470号 能越自動車道 緊急復旧の状況」',25,MUTED)
    text(d,(110,804),'背景確認：警察庁「令和6年能登半島地震に伴う警察活動」2024年1月10日',23,MUTED)
    save(im,'credits')
    im,d=canvas();save(im,'black')
    # Exact supplied text in manually checked phrase ranges; speech start is 2.5s.
    captions=[
      (0.3,8.4,'大規模災害では、救助隊も救助犬も、\n同時に発生するすべての現場を捜索するには必ず不足します。'),
      (9,17.95,'2024年の能登半島地震では、全国から救助部隊が集結しても、\n道路の寸断が現場への到達を阻みました。'),
      (19,23.2,'限られた救助資源を、どこに最初に向けるのか。'),
      (23.2,27.2,'その判断に必要な情報すら、瓦礫の奥からは届きません。'),
      (28.55,32.65,'ドローンは上空や飛行可能な空間の把握に、'),
      (32.65,36.55,'蛇型ロボットは細い経路の詳細観察に強みがあります。'),
      (37,43.35,'しかし、複雑に分岐した瓦礫内部の広範囲を、\n同時に調べることは容易ではありません。'),
      (44,47.4,'そこで考えたのが、ぷるもぐサーチです。'),
      (48,56.1,'約7センチの小型ノードを多数散布し、瓦礫の隙間へ分散させることで、\n一度に探索できる範囲を広げます。'),
      (57,63.3,'親に探索方針を与えれば、ドローンなどを機体ごとに操縦するような\n個別操作は必要ありません。'),
      (63.3,70.1,'ノード同士が連結・分離し、隊列を組み替えながら、\n狭所の奥まで潜り込むことを目指します。'),
      (70.1,75.25,'独自性は、進めた場所だけを地図にするのではないこと。'),
      (75.25,79.8,'止まった。接触した。通信が途切れた。'),
      (80.85,88,'通常は失敗や欠損として扱われる反応も、\n見えない空洞や閉塞を推定する手がかりに変えます。'),
      (88.85,95.8,'AIは散布したノード群の観測を統合し、\n空洞のつながりや、人の気配が疑われる場所を推定。'),
      (96,100.95,'さらに次の投入先を判断し、探索範囲を効率的に広げます。'),
      (101.8,109,'狙いは、広い被災地の中から、救助隊が優先して確認すべき箇所を\n迅速に絞り込むことです。'),
      (109,114.8,'今回、UnityとRustを使い、\n倒壊・斜面崩壊・浸水後の閉塞を想定した環境で、'),
      (114.8,121.1,'協調侵入と空間推定の有効性を検証しました。'),
      (121.1,128.95,'限られた人員・資材を、一刻も早く、\n救助が必要な可能性の高い場所へ向かわせるために。'),
      (130.1,134.7,'救助の手が足りないときこそ、捜索の手がかりを広く、速く集める。'),
      (134.7,136.55,'ぷるもぐサーチです。')]
    manifest=[]
    for i,(start,end,s) in enumerate(captions):
        im,d=canvas(True);lines=s.split('\n');f=font(31,True)
        for j,line in enumerate(lines):
            width=d.textlength(line,font=f)
            if width>1740:raise ValueError('Caption exceeds safe area: '+line)
            d.text(((1920-width)/2,965+j*43),line,font=f,fill=INK)
        name=f'caption-{i:02d}';save(im,name);manifest.append(dict(file=name+'.png',start_s=start+2.5,end_s=end+2.5,text=s))
    (OUT/'captions.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def capture(shot):
    folder=ROOT/'media/frames'/shot;rows=list(csv.DictReader((folder/'timestamps.csv').open()))
    times=np.array([float(r['wall_time_s']) for r in rows]);indices=[int(r['frame_index']) for r in rows]
    sample=Image.open(folder/f'frame-{indices[0]:05d}.png');size=sample.size;fps=30
    total=math.floor((times[-1]-times[0])*fps)
    writer=imageio_ffmpeg.write_frames(str(OUT/(shot+'.mp4')),size,fps=fps,codec='libx264',pix_fmt_out='yuv420p',macro_block_size=1,ffmpeg_log_level='error',output_params=['-crf','16','-preset','fast','-movflags','+faststart'])
    writer.send(None);last=-1;data=None
    for frame in range(total):
        idx=indices[max(0,int(np.searchsorted(times,times[0]+frame/fps,side='right')-1))]
        if idx!=last:data=Image.open(folder/f'frame-{idx:05d}.png').convert('RGB').tobytes();last=idx
        writer.send(data)
    writer.close()
    (OUT/(shot+'-capture.json')).write_text(json.dumps(dict(source_shot=shot,frames=total,fps=30,size=size,duration_s=total/30,motion_synthesis=False,method='Uncut application backbuffer capture, previous frame by recorded wall-clock timestamp'),indent=2)+'\n')
    print('SOURCE_CLIP',shot,total/30,flush=True)

def hero():
    writer=imageio_ffmpeg.write_frames(str(OUT/'robot-stage.mp4'),(1920,1080),fps=30,codec='libx264',pix_fmt_out='yuv420p',macro_block_size=1,ffmpeg_log_level='error',output_params=['-crf','15','-preset','fast','-movflags','+faststart'])
    writer.send(None)
    for n in range(1,721):writer.send(Image.open(OUT/f'robot-stage/frame-{n:04d}.png').convert('RGB').tobytes())
    writer.close();print('ROBOT_STAGE',24,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--artwork',action='store_true');parser.add_argument('--hero',action='store_true');parser.add_argument('shots',nargs='*');args=parser.parse_args()
    if args.artwork:artwork()
    if args.hero:hero()
    for shot in args.shots:capture(shot)
