"""Typeset still assets; timestamp-resample uncut captures. All editing/mixing stays in Resolve."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/'tmp/media-runtime').exists():sys.path.insert(0,str(ROOT/'tmp/media-runtime'))
from PIL import Image,ImageDraw,ImageFont
import prepare_resolve_media as capture

OUT=ROOT/'media/rescue_resolve_sources'
DARK='#101d24';INK='#edf3ee';MUTED='#9bb3b7';MINT='#71e0bd';AMBER='#efb36c';LINE='#30454d'

def font(size,bold=False):return ImageFont.truetype('C:/Windows/Fonts/YuGothB.ttc' if bold else 'C:/Windows/Fonts/YuGothM.ttc',size,index=0)
def text(d,x,y,value,size=36,color=INK,bold=False):d.text((x,y),value,font=font(size,bold),fill=color,spacing=16)
def base():
    im=Image.new('RGB',(1920,1080),DARK);d=ImageDraw.Draw(im)
    d.rectangle((110,100,168,105),fill=MINT);text(d,110,126,'PURUMOGU SEARCH  /  DISASTER VOID STUDY',24,MUTED)
    return im,d

def cards(metrics=None,prior_test=None):
    OUT.mkdir(parents=True,exist_ok=True)
    im,d=base();text(d,110,280,'ぷるもぐサーチ',105,INK,True)
    text(d,115,465,'7 cm の群れで、災害後の空隙を調べる。',43)
    d.line((115,655,1805,655),fill=LINE,width=2)
    text(d,115,710,'UNITY × RUST',32,MINT,True)
    text(d,115,778,'15 シーン  /  24・36・48 台  /  関節と着脱',34,MUTED)
    text(d,115,950,'シミュレーションによる実装・検証',26,MUTED)
    im.save(OUT/'rescue-title.png')

    im,d=base();text(d,110,235,'災害後の空隙を、15条件で',70,INK,True)
    columns=[('易',['E01  傾いた壁の下','E02  落下した天井材','E03  土砂の排水路','E04  浸水後の通路','E05  無人の倉庫跡']),('中',['M01  パンケーキ状崩壊','M02  V字に落ちた床','M03  分断された暗渠','M04  流木の挟まった配管','M05  無人地下室']),('難',['H01  多層床板の空隙','H02  鉄筋が露出した階段','H03  岩塊で塞がれた斜面','H04  半水没した設備空間','H05  静かな深部空隙'])]
    for index,(tier,rows) in enumerate(columns):
        x=115+index*575;text(d,x,385,tier,54,MINT,True)
        for n,row in enumerate(rows):text(d,x,490+n*70,row,29)
    text(d,115,930,'地震・斜面崩壊・浸水  /  無人条件を含む  /  要救助者の位置も変種ごとに変更',28,MUTED)
    im.save(OUT/'rescue-scenes.png')

    im,d=base();text(d,110,240,'観測と、補完を分けて見る',74,INK,True)
    for y,color,label,note in [(425,MINT,'距離で観測した形状','ノイズと姿勢誤差を含む'),(565,'#ad8de6','学習から補った形状','実際に見えた場所としては数えない'),(705,AMBER,'音響からの人の候補','確認済みの要救助者とは扱わない')]:
        d.rectangle((115,y+14,125,y+76),fill=color);text(d,165,y,label,42,color,True);text(d,165,y+72,note,29,MUTED)
    im.save(OUT/'rescue-evidence.png')

    if metrics:
        m=json.loads((ROOT/metrics).read_text(encoding='utf-8-sig'));im,d=base()
        text(d,110,225,'未学習の変種で測る',76,INK,True)
        values=[('3D地図 IoU',m['map_iou_macro']),('検出の適合率',m['human_precision']),('検出の再現率',m['human_recall'])]
        for index,(label,value) in enumerate(values):
            x=115+index*575;col=MINT if value>=.95 else AMBER
            text(d,x,380,label,34,MUTED);text(d,x,450,f'{value:.1%}',112,col,True)
            d.rectangle((x,645,x+460,656),fill=LINE);d.rectangle((x,645,x+int(value*460),656),fill=col)
            xx=x+int(.95*460);d.line((xx,632,xx,671),fill=INK,width=2)
        status='3指標の目標に到達' if m['target_met'] else '95%の合格条件には未達'
        if m['split']=='valid' and m['target_met']:status='VALIDで目標に到達 / 最終TESTは別途確認'
        distributed=set(m.get('by_node_count',{}))=={'24','36','48'}
        if distributed:
            for index,(count,bucket) in enumerate(sorted(m['by_node_count'].items(),key=lambda item:int(item[0]))):
                text(d,115,686+index*34,f"{count} 台   地図 {bucket['map_iou_macro']:.1%}   適合率 {bucket['human_precision']:.1%}   再現率 {bucket['human_recall']:.1%}",25,MUTED)
        text(d,115,810 if distributed else 760,status,38 if distributed else 48,MINT if m['target_met'] else AMBER,True)
        text(d,115,876 if distributed else 866,f"{m['split'].upper()}  /  {m['maps']} 試行  /  延べ対象数 {m['people']}  /  一致 {m['matched']}・誤警報 {m['false_alarms']}",28,MUTED)
        text(d,115,940,'位置一致は25cm以内・一対一対応  /  地図は未観測の障害物も含む全領域IoU',27,MUTED)
        if prior_test:
            previous=json.loads((ROOT/prior_test).read_bytes());assert previous['split']=='test' and not previous['target_met']
            text(d,115,988,f"初回TESTの再現率 {previous['human_recall']:.1%} は未達。改訂後の結果を別に集計。",24,MUTED)
        im.save(OUT/'rescue-results.png')

    im,d=base();text(d,110,235,'残る課題も、実測で残す',76,INK,True)
    entries=[('垂直の25mm段差','乗り越えは不通過。回り込みは通過。'),('音響・熱・水のモデル','簡略化した観測。実測での校正が必要。'),('実機・実災害での救助','未検証。ここでの数値を適用しない。')]
    for n,(label,note) in enumerate(entries):
        y=425+n*155;text(d,115,y,label,38,AMBER,True);text(d,115,y+65,note,34)
    im.save(OUT/'rescue-limits.png')

    im,d=base();text(d,110,240,'ぷるもぐサーチ',82,INK,True)
    for n,(label,value) in enumerate([('VOICE','音読さん / Ellis'),('MUSIC','Suno AI / Purumogu - Small Traces'),('EDIT & MIX','DaVinci Resolve Studio 21.1'),('SIMULATION','Unity 6000.4.1f1 × Rust 1.94.1')]):
        y=415+n*95;text(d,115,y,label,25,MINT,True);text(d,500,y-5,value,34)
    text(d,115,895,'github.com/Applekeynote/purumogu-search',36,INK,True)
    text(d,115,960,'コード・学習条件・失敗例・説明資料を公開',27,MUTED)
    im.save(OUT/'rescue-credits.png')
    for name,label in [('overview','01 / 災害現場と群れ'),('mechanism','02 / 球体を軸に曲がる・回る'),('mapping','03 / 観測から地図と音響候補へ')]:
        im=Image.new('RGBA',(1920,1080),(0,0,0,0));d=ImageDraw.Draw(im)
        d.rectangle((0,0,1919,53),fill=DARK);d.rectangle((0,1026,1919,1079),fill=DARK)
        d.rectangle((0,54,95,1025),fill=DARK);d.rectangle((1824,54,1919,1025),fill=DARK)
        text(d,100,9,label,27,INK,True);text(d,100,1038,'実アプリの記録 / 複数の実行から編集 / 表示値はその時点の推論',23,MUTED)
        im.save(OUT/('rescue-overlay-'+name+'.png'))
    (OUT/'README.md').write_text('Code-typeset still graphics and timestamp-resampled uncut application captures. Creative editing, scaling into the caption matte, shot cuts, fades, music looping/crossfades, normalization, mixing, and final export are performed in DaVinci Resolve.\n',encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--metrics');parser.add_argument('--prior-test');parser.add_argument('shots',nargs='*');args=parser.parse_args()
    cards(args.metrics,args.prior_test);capture.OUT=OUT
    for shot in args.shots:capture.clip(shot)
