"""Typeset the Japanese project report from measured artifacts, using ReportLab."""
from pathlib import Path
import sys
import json
from xml.sax.saxutils import escape

ROOT=Path(__file__).resolve().parents[1]
if (ROOT/'tmp/media-runtime').exists():sys.path.insert(0,str(ROOT/'tmp/media-runtime'))
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor,white
from reportlab.lib.utils import ImageReader

INK=HexColor('#17303a');MUTED=HexColor('#61757b');MINT=HexColor('#159b80');AMBER=HexColor('#a45f1e');PAPER=HexColor('#f8faf7')
font_path=Path('C:/Windows/Fonts/YuGothM.ttc')
if font_path.exists():pdfmetrics.registerFont(TTFont('HeiseiKakuGo-W5',str(font_path),subfontIndex=0))
else:pdfmetrics.registerFont(UnicodeCIDFont('HeiseiKakuGo-W5'))
STYLE=ParagraphStyle('body',fontName='HeiseiKakuGo-W5',fontSize=10.2,leading=17,textColor=INK,wordWrap='CJK',spaceAfter=8)
SMALL=ParagraphStyle('small',parent=STYLE,fontSize=8.5,leading=13)
TABLE=ParagraphStyle('table',parent=STYLE,fontSize=8.5,leading=13)
W,H=595.276,841.89;X=44;WIDTH=W-88

class Report:
    def __init__(self,path):
        self.c=canvas.Canvas(str(path),pagesize=(W,H));self.c.setTitle('ぷるもぐサーチ - Unity × Rust 実装・評価報告');self.c.setAuthor('Purumogu Search contributors');self.page=0
    def start(self,title,kicker):
        self.page+=1;self.c.setFillColor(PAPER);self.c.rect(0,0,W,H,fill=1,stroke=0)
        self.c.setFillColor(MINT);self.c.rect(X,H-42,34,4,fill=1,stroke=0)
        self.c.setFont('Helvetica',9);self.c.setFillColor(MUTED);self.c.drawString(X,H-60,'PURUMOGU SEARCH  /  IMPLEMENTATION NOTE  /  2026-10-27')
        self.c.setFont('HeiseiKakuGo-W5',23);self.c.setFillColor(INK);self.c.drawString(X,H-101,title)
        self.y=H-130;self.p(kicker,SMALL)
    def p(self,text,style=STYLE):
        p=Paragraph(text,style);_,h=p.wrap(WIDTH,720);assert self.y-h>48,f'Page {self.page} overflow: {text[:40]}';p.drawOn(self.c,X,self.y-h);self.y-=h+8
    def heading(self,text):
        self.y-=6;self.c.setFont('HeiseiKakuGo-W5',13);self.c.setFillColor(MINT);self.c.drawString(X,self.y-15,text);self.y-=29
    def table(self,rows,widths):
        cells=[[Paragraph(escape(str(x)),TABLE) for x in row] for row in rows]
        t=Table(cells,colWidths=widths,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor('#e3ece5')),('TEXTCOLOR',(0,0),(-1,-1),INK),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,0),.6,MINT),('LINEBELOW',(0,1),(-1,-1),.3,HexColor('#d4dfd7'))]));_,h=t.wrap(WIDTH,700);assert self.y-h>48;t.drawOn(self.c,X,self.y-h);self.y-=h+12
    def image(self,path,width=WIDTH):
        img=ImageReader(str(path));iw,ih=img.getSize();height=width*ih/iw;assert self.y-height>48;self.c.drawImage(img,X,self.y-height,width=width,height=height);self.y-=height+10
    def end(self):
        self.c.setStrokeColor(HexColor('#d4dfd7'));self.c.line(X,43,W-X,43);self.c.setFont('Helvetica',8);self.c.setFillColor(MUTED);self.c.drawString(X,28,'Simulation prototype / measured results / no rescue-performance claim');self.c.drawRightString(W-X,28,f'{self.page:02d}');self.c.showPage()
    def save(self):self.c.save()

def main():
    metrics=json.loads((ROOT/'reports/comparison/metrics.json').read_text());analysis=json.loads((ROOT/'reports/analysis.json').read_text());model=json.loads((ROOT/'models/m-v1/model.json').read_text())
    output=ROOT/'reports/Purumogu_Search_Report.pdf';r=Report(output)
    r.start('ぷるもぐサーチ','Unityの物理応答から候補構造を推定し、Rustが次の入口と隊形を選ぶ。設計書v1.1のPhase 1実装・評価報告。')
    r.p('がれき内部を外から正確に測るのではなく、投入した小さなノード群の反応を観測媒体にする。今回は「任意の地図を復元できるか」ではなく、限られた候補族の識別と、次の観測を取りに行く行動を検証した。')
    r.image(ROOT/'media/screenshot.png')
    r.p('図1　実際のUnity実行画面。左は観客向けの断面とノード、右はRustが観測だけから計算した候補分布。形状・ノード座標は推論入力に含めない。',SMALL)
    r.heading('完成した範囲')
    r.p('5形状、20ノードの利用枠、3入口、5個×3バッチ。CLUSTER / CHAIN_INSERTの2隊形と一時連結、Rustの頻度モデル学習・逐次ベイズ更新・期待情報利得、準備・探索・結果・回収の4画面を実装した。ゲートとノードは物理計算で動く。')
    r.p('未使用50マップで、GHOST-FLOW 61%、固定投入60%、ランダム46%のTop-1正解率。固定投入に対する優位性は確認できなかった。AI方式の保留率は79%。未知形状では8試行中6試行で誤断定が残り、ここが最大の弱点である。')
    r.end()

    r.start('観測から行動まで','Unity 6000.4.1f1 / Rust 1.94.1 / TCP 127.0.0.1 / JSON Lines / CPU推論')
    r.heading('データ境界')
    r.table([['Rustへ渡す','評価だけに保存する'],['衝撃、静止類似、受信peerとRSSI風値、時刻、既知の投入・隊形・連結状態','世界座標、剛体真値速度、壁ID、真の部屋・形状、未受信peerの真値、到達深さ']], [WIDTH/2]*2)
    r.p('疑似IMUを50Hz、無線を5Hzでサンプルする。加速度には飽和・量子化・雑音・バイアスを加え、静止類似を確定停止と呼ばない。無線の欠落は壁の存在と一対一には対応しない。受信時刻を基準に整列する。')
    r.heading('学習と逐次更新')
    r.p('TRAINで16特徴と存在マスクを標準化し、16個の観測符号を学習する。候補族×入口×隊形の符号頻度に擬似カウント1を加える。VALIDのエピソード末尾log lossで温度を選び、TEST前に固定した。')
    r.p('<font name="Courier" size="9">p_next(G) proportional p(G) * P(z | G, action)<br/>EIG(a) = H(p) - sum_z P(z | a) H(p(G | z,a))</font>')
    r.p('更新は各バッチの終了時に1回。行動集合は3入口×2隊形、投入数は5個で共通。親は選ばれた指令を子へ配り、CHAINでは近接・相対角条件を満たすノードをFixedJointでつなぐ。座標を連続的に書き換えて移動を演出しない。')
    r.heading('分割と再現性')
    r.table([['用途','マップ数','エピソード数'],['TRAIN（全6開始行動を収集）','50','300'],['VALID（温度だけを選択）','15','90'],['TEST（4方式×2センサーシード）','50','400']], [WIDTH*.62,WIDTH*.17,WIDTH*.21])
    r.p('map_id単位で完全分離。比率は設計書の推奨70/15/15と異なり、未使用マップの件数を厚くした。モデル・分割・全量データのSHA-256を保存。観測ログの推論再生は一致を検査し、PhysXのビット単位の再現は保証しない。',SMALL)
    r.end()

    r.start('同じ予算で比較する','各方式100試行。5族×未使用10マップ×センサー2シード。各試行は15ノード、3バッチ、物理50Hz、27秒。')
    rows=[['方式','Top-1','95%区間','保留率','保留なし正解','Brier']]
    for key,name in [('ghost_flow','GHOST-FLOW'),('fixed','固定'),('random','ランダム'),('passive','受動')]:
        m=metrics[key];lo,hi=m['accuracy_ci95'];selected=analysis['physical'][key]['selected_episodes'];correct=round((m['selective_accuracy'] or 0)*selected)
        rows.append([name,f"{m['accuracy']:.0%}",f'{lo:.0%} - {hi:.0%}',f"{m['abstention_rate']:.0%}",f'{correct}/{selected}',f"{m['brier']:.3f}"])
    r.table(rows,[WIDTH*.22,WIDTH*.12,WIDTH*.19,WIDTH*.13,WIDTH*.16,WIDTH*.18])
    r.p('Top-1は保留を含む全試行の最上位候補一致率。保留なし21/21だけを「精度100%」と表現しない。Brierは小さいほど良く、AI方式は固定方式より悪かった。',SMALL)
    r.image(ROOT/'reports/accuracy.png')
    r.heading('差の根拠と読み方')
    r.p('AI−固定は+1ポイント、ペア比較の95%区間は−12〜+15ポイント。固定方式を上回る証拠にはならない。AI−ランダムは+15ポイント（+3〜+27）だった。ただし、この生成器とテンプレートの範囲での探索的な結果である。')
    r.p('区間はマップを単位に再抽出する。個々のセンサー試行を独立マップのように数えない。ペア差は10,000回、方式別の区間は2,000回のブートストラップ。モデル・温度・棄却閾値はTEST後に調整していない。')
    r.end()

    r.start('失敗を含めて評価する','親の能動移動、観測欠損、候補外の形状を別々に確認した。改善が見えた条件と、解けていない条件を分ける。')
    r.heading('親あり／受動流動')
    r.table([['条件','平均最大到達深さ','平均通過ノード数'],['固定・CLUSTER・推力あり',f"{analysis['physical']['fixed']['mean_depth_m']:.3f} m",f"{analysis['physical']['fixed']['mean_passed_nodes']:.2f}"],['同じ入口順・推力なし',f"{analysis['physical']['passive']['mean_depth_m']:.3f} m",f"{analysis['physical']['passive']['mean_passed_nodes']:.2f}"]],[WIDTH*.50,WIDTH*.27,WIDTH*.23])
    r.p('能動移動による到達深さの差は+0.179m（ペア95%区間+0.097〜+0.269m）。これは親の指令と推力を合わせた差であり、連結だけの効果ではない。鎖状の剛体連結は曲がり角で詰まり、常に奥へ進むとは限らない。')
    r.heading('受信スキャンの欠損')
    r.table([['欠損率','試行数','Top-1','保留率']]+[[f"{x['packet_loss']:.0%}",x['episodes'],f"{x['accuracy']:.0%}",f"{x['abstention_rate']:.0%}"] for x in analysis['faults']],[WIDTH*.25]*4)
    r.p('各条件10試行のみ。欠損が増えるほど保留率が必ず上がる実装にはなっていない。欠損に耐えたという一般的な結論は出せない。独立したリンク損失ではなく、受信側のスキャン全体を欠落させる模型である。',SMALL)
    r.heading('最も大きい未解決点：未知形状')
    r.p('<font color="#a45f1e">候補外の結合パターンは2マップ×4欠損条件の8試行。保留は2試行、誤った断定は6試行だった。</font> OTHERは一様な符号分布と支持範囲検査であり、信頼できる未知形状検出器ではない。今回のTESTで閾値を調整せず、失敗として保存した。')
    r.p('次は新しい形状群をVALID用に収集し、棄却の被覆率と誤断定率を校正する。同一エピソードの摩擦・遮蔽・履歴を持続させる粒子フィルタや履歴条件付き予測器が、単純なモデル大型化より先の改善候補になる。')
    r.end()

    r.start('動かし方と残る課題','公開コード・固定モデル・実測ログ・全試行のCSV/JSONから再検証できる形にした。')
    r.heading('今日から行う3つの操作')
    r.p('1. 画面で動かす：<font name="Courier" size="9">scripts/Start-Demo.ps1</font>。Rustを先に起動し、UnityのPrepareでStart。初回は<font name="Courier" size="9">scripts/Unity-Task.ps1 init</font>でシーンを生成する。')
    r.p('2. 同条件で比較：<font name="Courier" size="9">scripts/Run-Experiments.ps1</font>。通信欠損と未知形状は<font name="Courier" size="9">-Faults</font>。全量収集と別モデルの生成は<font name="Courier" size="9">-Collect</font>。')
    r.p('3. Unityなしで再生：<font name="Courier" size="8.5">cargo run --manifest-path rust/Cargo.toml --release --locked -- replay<br/>--model models/m-v1/model.json --events datasets/sample.jsonl</font>')
    r.heading('今回の境界')
    r.p('利点は、CPUで小さく始め、観測と判断の根拠を残して行動まで閉じられること。弱点は、少数の形状族と簡略センサーに依存し、任意の構造や未知形状には弱いこと。直線と狭窄は同じ連結性を持つため、今回の分類には通路属性の識別も含まれる。')
    r.p('親の電波指令損失、ANCHOR_PULL、RELAY、柔軟な結合機構、GNN、3D occupancy、Human-Cue、実機接続は今後の項目。長期目標のConnectivity F1、voxel IoU、人由来兆候の検出性能は測定していない。')
    r.heading('一次資料・成果物')
    refs=[('公開コード','https://github.com/Applekeynote/purumogu-search'),('Unity Physics.Simulate','https://docs.unity3d.com/6000.4/Documentation/ScriptReference/Physics.Simulate.html'),('Tokio TcpListener 1.52.3','https://docs.rs/tokio/1.52.3/tokio/net/struct.TcpListener.html'),('関連研究：Active Sequential Posterior Estimation (2024)','https://research.google/pubs/active-sequential-posterior-estimation-for-sample-efficient-simulation-based-inference/')]
    for label,url in refs:r.p(f'<link href="{url}" color="#159b80">{escape(label)}</link>',SMALL)
    r.p('関連研究は背景であり、GHOST-FLOWの学術的新規性や今回の性能を保証するものではない。実装と実測値の出典はこのリポジトリ。設計書に引用された講座原資料は独立確認していない。',SMALL)
    r.heading('音声・音楽')
    r.p('音声：音読さん（Beta / Ellis、落ち着いた説明）。音楽：Suno AI「Purumogu - Small Traces」。実行画面は本アプリから収録。音声・音楽の利用条件とクレジットはmedia/CREDITS.mdに記載。YouTube・OmniCampusへの登録は依頼範囲に含めない。',SMALL)
    r.p('モデルSHA-256：<font name="Courier" size="7">'+analysis['model_sha256']+'</font>',SMALL)
    r.end();r.save();print(output)

if __name__=='__main__':main()
