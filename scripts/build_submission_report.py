"""Build the submission PDF from recorded experiments, with explicit result provenance."""
from pathlib import Path
import hashlib
import json
import re
from html import unescape
from xml.sax.saxutils import escape

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor, white
from reportlab.lib.utils import ImageReader
from PIL import Image
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/pdf'
TMP = ROOT / 'tmp/pdfs/submission-20261029'
PDF = OUT / 'Purumogu_Search_Report_20261029.pdf'
REPO = 'https://github.com/Applekeynote/purumogu-search'
BRANCH = REPO + '/tree/codex/disaster-rescue-v2'
VIDEO = 'https://youtu.be/LzZhaq-cYbs'
INK = HexColor('#192f37')
TEAL = HexColor('#087f78')
MUTED = HexColor('#5b6d73')
LINE = HexColor('#d5dfdf')
PALE = HexColor('#eff5f3')
AMBER = HexColor('#a35f00')
W, H = 595.276, 841.89
X, WIDTH = 43, 509.276


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8-sig'))


def pct(v):
    return f'{100*v:.2f}%'


def plain(s):
    return unescape(re.sub('<[^>]+>', '', s.replace('<br/>', '\n')))


class Report:
    def __init__(self):
        self.c = canvas.Canvas(str(PDF), pagesize=(W, H), pageCompression=1)
        self.c.setTitle('ぷるもぐサーチ | 学習方法と到達精度・実装報告')
        self.c.setAuthor('')
        self.c.setSubject('分散小型ノードによる災害空隙探索のシミュレーション')
        self.c.setCreator('Purumogu Search report builder')
        self.page = 0
        self.md = []
        self.layout = []
        self.body = ParagraphStyle('body', fontName='Body', fontSize=10.1, leading=17.2,
                                   textColor=INK, wordWrap='CJK')
        self.small = ParagraphStyle('small', parent=self.body, fontSize=8.1, leading=12.8,
                                    textColor=MUTED)
        self.cell = ParagraphStyle('cell', parent=self.body, fontSize=8.5, leading=13.4)

    def start(self, title, subtitle):
        self.page += 1
        self.c.setFillColor(white)
        self.c.rect(0, 0, W, H, fill=1, stroke=0)
        self.c.setFillColor(TEAL)
        self.c.rect(X, H-43, 22, 3, fill=1, stroke=0)
        self.c.setFont('Helvetica', 8.3)
        self.c.setFillColor(MUTED)
        self.c.drawString(X+31, H-44, 'PURUMOGU SEARCH  /  TECHNICAL REPORT')
        self.c.drawRightString(W-X, H-44, '2026.10.29')
        self.c.setFillColor(INK)
        self.c.setFont('Heading', 24)
        self.c.drawString(X, H-91, title)
        self.y = H-113
        self.md.extend(['', '# '+title, '', subtitle, ''])
        self.p(subtitle, small=True, record=False)

    def p(self, s, small=False, record=True, gap=9):
        sty = self.small if small else self.body
        p = Paragraph(s, sty)
        _, h = p.wrap(WIDTH, 1000)
        self.ensure(h+gap, plain(s)[:35])
        p.drawOn(self.c, X, self.y-h)
        self.y -= h+gap
        if record:
            self.md.extend([plain(s), ''])

    def ensure(self, h, label):
        if self.y-h < 57:
            raise ValueError(f'Page {self.page}: overflow at {label}, y={self.y:.1f}, h={h:.1f}')

    def heading(self, s):
        self.ensure(35, s)
        self.c.setFillColor(TEAL)
        self.c.setFont('Bold', 12)
        self.c.drawString(X, self.y-15, s)
        self.y -= 30
        self.md.extend(['## '+s, ''])

    def table(self, rows, fractions, size=8.5):
        sty = ParagraphStyle('table', parent=self.cell, fontSize=size, leading=size+4.8)
        cells = [[Paragraph(escape(str(x)).replace('\n', '<br/>'), sty) for x in row] for row in rows]
        t = Table(cells, colWidths=[WIDTH*f for f in fractions])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), PALE),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 7),
            ('RIGHTPADDING', (0, 0), (-1, -1), 7),
            ('TOPPADDING', (0, 0), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
            ('LINEBELOW', (0, 0), (-1, 0), .6, TEAL),
            ('LINEBELOW', (0, 1), (-1, -1), .3, LINE),
        ]))
        _, h = t.wrap(WIDTH, 1000)
        self.ensure(h+13, 'table')
        t.drawOn(self.c, X, self.y-h)
        self.y -= h+13
        self.md.append(' | '.join(map(str, rows[0])))
        self.md.append(' | '.join(['---']*len(rows[0])))
        self.md += [' | '.join(map(str, row)) for row in rows[1:]] + ['']

    def image(self, path, caption, width=None, crop=None):
        width = width or WIDTH
        with Image.open(path) as im:
            iw, ih = im.size
        left, top, right, bottom = crop or (0, 0, iw, ih)
        scale = width/(right-left)
        h = (bottom-top)*scale
        self.ensure(h+14, 'image')
        image_x = X+(WIDTH-width)/2
        image_y = self.y-h
        self.c.saveState()
        clip = self.c.beginPath()
        clip.rect(image_x, image_y, width, h)
        self.c.clipPath(clip, stroke=0, fill=0)
        self.c.drawImage(ImageReader(str(path)), image_x-left*scale, image_y-(ih-bottom)*scale,
                         width=iw*scale, height=ih*scale, mask='auto')
        self.c.restoreState()
        self.y -= h+9
        self.p(caption, small=True, gap=12)

    def diagram(self):
        height = 194
        self.ensure(height, 'architecture')
        top = self.y
        boxes = [
            (X, top-66, 239, 58, 'Unity / 物理と観測', '重力・衝突・関節・模擬センサー'),
            (X+270, top-66, 239, 58, 'Rust / 推定と探索制御', '地図・候補履歴・次の探索指令'),
            (X, top-165, 239, 58, 'Truth / 教師・評価専用', '正解地形・人の位置・人数'),
            (X+270, top-165, 239, 58, 'Offline / 学習と再評価', 'TRAINで係数学習、VALIDで選択'),
        ]
        for x, y, w, h, title, text in boxes:
            self.c.setFillColor(PALE)
            self.c.setStrokeColor(LINE)
            self.c.rect(x, y, w, h, fill=1, stroke=1)
            self.c.setFillColor(INK)
            self.c.setFont('Bold', 10.3)
            self.c.drawString(x+11, y+37, title)
            self.c.setFont('Body', 8.3)
            self.c.drawString(x+11, y+17, text)
        self.c.setStrokeColor(TEAL)
        self.c.setLineWidth(1)
        for yy, reverse in [(top-28, False), (top-50, True), (top-136, False)]:
            a, b = (X+241, X+268) if not reverse else (X+268, X+241)
            self.c.line(a, yy, b, yy)
            dx = 4 if reverse else -4
            self.c.line(b, yy, b+dx, yy+3)
            self.c.line(b, yy, b+dx, yy-3)
        self.c.setFont('Body', 8)
        self.c.setFillColor(MUTED)
        self.c.drawString(X+4, top-90, '正解データは上段の推論通信へ入れない。')
        self.y -= height
        self.md.append('構成：Unity ⇄ Rust（観測と制御）、Truth → Offline（教師・評価のみ）。')

    def bars(self, labels, vals, caption):
        self.ensure(134, 'bars')
        left, bw, top = X+142, WIDTH-210, self.y
        for i, (label, v) in enumerate(zip(labels, vals)):
            yy = top-22-i*31
            self.c.setFont('Body', 9)
            self.c.setFillColor(INK)
            self.c.drawString(X, yy, label)
            self.c.setFillColor(PALE)
            self.c.rect(left, yy-2, bw, 12, fill=1, stroke=0)
            self.c.setFillColor(TEAL)
            self.c.rect(left, yy-2, bw*v, 12, fill=1, stroke=0)
            self.c.setFont('Helvetica-Bold', 10)
            self.c.setFillColor(INK)
            self.c.drawRightString(W-X, yy, pct(v))
        self.y -= 107
        self.p(caption, small=True)

    def end(self):
        self.layout.append({'page': self.page, 'content_bottom_pt': round(self.y, 2)})
        self.c.setStrokeColor(LINE)
        self.c.setLineWidth(.5)
        self.c.line(X, 43, W-X, 43)
        self.c.setFillColor(MUTED)
        self.c.setFont('Body', 7.5)
        self.c.drawString(X, 27, 'ぷるもぐサーチ / 学習方法と到達精度・実装報告')
        self.c.setFont('Helvetica', 8)
        self.c.drawRightString(W-X, 27, f'{self.page:02d} / 09')
        self.c.showPage()

    def save(self):
        assert self.page == 9
        self.c.save()
        (OUT / 'Purumogu_Search_Report_20261029.md').write_text('\n'.join(self.md), encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    pdfmetrics.registerFont(TTFont('Body', 'C:/Windows/Fonts/YuGothM.ttc', subfontIndex=0))
    pdfmetrics.registerFont(TTFont('Bold', 'C:/Windows/Fonts/YuGothB.ttc', subfontIndex=0))
    pdfmetrics.registerFont(TTFont('Heading', 'C:/Windows/Fonts/yumin.ttf'))
    v9 = read('reports/rescue/learning-v9-valid-policy1.json')
    v11 = read('reports/rescue/learning-v11-valid-policy2.json')
    va = read('reports/rescue/mission360/learning-v13-valid.json')
    te = read('reports/rescue/mission360/learning-v13-test.json')
    train = read('models/rescue-v13/training-report.json')
    lock = read('reports/rescue/mission360/protocol-lock.json')
    selection = read('models/rescue-v13/selection-report.json')
    physical = read('reports/rescue/distributed-deployment-smoke.json')
    docking = read('reports/rescue/spherical-docking-smoke.json')
    ablation = read('reports/rescue/mission360/ablation-v11-valid.json')
    probe = read('reports/rescue/mission360-probe-valid90/baseline-v17-valid90-summary.json')
    model_hash = hashlib.sha256((ROOT/'models/rescue-v13/model.json').read_bytes()).hexdigest()
    assert model_hash == va['model_sha256'] == te['model_sha256'] == lock['model_sha256'] == train['model_sha256']
    assert va['maps'] == 30 and te['maps'] == 45 and lock['nodes'] == 12
    assert train['train_proposals'] == 1040 and len(read('models/rescue-v13/model.json')['train_maps']) == 150
    assert physical['cases'] == 45 and sum(x['released'] for x in physical['rows']) == 1620
    assert sum(x['lost'] for x in physical['rows']) == 0 and docking['ports_per_node'] == 6
    assert probe['maps'] == 90 and probe['matched'] == 91 and probe['people'] == 96 and probe['false_alarms'] == 5
    assert round(probe['map_iou_macro']*100, 2) == 95.59 and round(probe['human_recall']*100, 2) == 94.79
    assert len(read('models/rescue-v17-probe/model.json')['train_maps']) == 450
    r = Report()

    r.start('ぷるもぐサーチ', '小型ノード群による災害空隙探索 / Unity × Rust 実装・学習報告')
    r.p('瓦礫の奥を一台ずつ調べるのではなく、小さなノードを分散させて捜索の手がかりを集める。約7cmの関節機体、複数センサーの模擬観測、逐次的な3D地図と人候補の推定を、一つのシミュレーターに実装した。')
    r.image(ROOT/'media/rescue-interface-v31.png', '図1　現行3.1の実行画面。H04・36台・探索途中の一例。左上：物理現場、右上：推定地図、左下：ノード状態、右下：評価。画面の瞬間値は集計精度を表す値ではない。')
    r.heading('何を学習し、どこまで到達したか')
    r.p(f'距離観測の補正と形状の事前分布を学習し、音響・温度などの候補履歴を分類した。12台・360秒の条件では、VALID 30変種で地図IoU <b>{pct(va["map_iou_macro"])}</b>、人候補の適合率・再現率はともに <b>{pct(va["human_precision"])}</b> に到達した。モデル固定後のTEST 45変種では、地図IoU {pct(te["map_iou_macro"])}、適合率 {pct(te["human_precision"])}、再現率 {pct(te["human_recall"])} を得た。')
    r.p('現在は24・36・48台の全域散布、中央球を含む着脱、ガス・浸水表示へ拡張した。上記の精度は旧12台条件の結果であり、現行条件の精度として流用しない。学習の手順と改善過程を6ページ、評価の内訳を7ページに記す。')
    r.p('提出物は研究用シミュレーションである。実機による救助性能、実災害への適用、構造安全性は検証していない。', small=True)
    r.end()

    r.start('1　探索の狙いと処理の流れ', '観測を増やす機構と、観測から推定する処理を分けて検証する。')
    r.p('2024年の能登半島地震では、道路の寸断により当初の大型車両の移動が難しく、ヘリや小型車両による移動が行われた［1］。現場への到達と、到達後にどこを詳しく調べるかは別の課題である。本制作は後者に着目し、狭い空隙に分散した観測から、優先確認箇所を絞るための情報を得ることを狙った。')
    r.p('設計の出発点は、停止・接触・通信断も空間の手がかりになる、という考え方だった。詳細な3D形状と人候補の位置を扱う現行版では、距離・姿勢・音響・温度も加えた。イベントだけで任意の空間を復元できたという結論にはしていない。')
    r.diagram()
    r.p('Unityは剛体・衝突・関節とセンサーの観測値を作る。Rustは観測を受け取り、占有地図と候補履歴を更新し、次の移動や観測確認の指令を返す。通信はローカルTCPのJSON Linesで、受信内容を検査してから更新する。Unity APIの適用はメインスレッドに戻す。')
    r.p('正解の壁形状、人の位置、人数、シーンIDは推論へ送らない。一方、指令した放出点は既知の基準点とし、以後の相対移動と姿勢に誤差を加える。この位置の基準が得られるという仮定は、実機化で確かめる必要がある。')
    r.p('原設計のイベントグラフ推定と追加投入ゲートの制御は初期実装として残し、現行版の3D地図とは区別した。現行の全域散布は初期配置の方式であり、初期実装の投入方策比較で優位だったことから、現行版の探索効率まで証明したとは扱わない。', small=True)
    r.end()

    r.start('2　7cmの機体と、着脱する群れ', '形状だけでなく、剛体・関節・接続面を持つモデルとして実装した。')
    r.image(ROOT/'media/report-robot-model.png', '図2　動画制作時の機構説明用CG。中央球と二つの扁平な円錐を表した設計イメージであり、実機写真や物理試験の記録ではない。', crop=(330, 300, 1560, 800))
    r.table([
        ['項目', '現在の実装'],
        ['寸法・質量', '全長70mm、中央球径19mm、総質量24g。いずれも仮の工学パラメーター。'],
        ['関節', '中央球と左右の円錐を別の剛体にし、球を軸とする屈曲と長軸回転を設定。UnityのConfigurableJointを使用［2］。'],
        ['6つの接続面', '円錐先端2面と中央球の左右・上下4面。先端同士、球同士、先端と球を接続・解除できる。'],
        ['接続の条件', '距離・向き・点速度を確認。球を含む捕捉距離5mm、先端同士9mm。障害物越し、多重接続、閉ループを防ぐ。'],
        ['移動', '重力と衝突で落下し、接地後は理想化した接線力で推進。電力やモーターの実機仕様は未確定。'],
    ], [.22, .78])
    r.p('専用試験では3種類の接続・解除と、瓦礫越しの球接続を拒否する動作を確認した。25mmの垂直段差では、そのままの乗り越えは成立せず、迂回は成立した。関節が曲がることと段差を越えられることは別に測る必要がある。［R5、R6］', small=True)
    r.end()

    r.start('3　災害を想定した15シーン', '易・中・難を各5条件。災害の種類だけでなく、遮蔽・段差・雑音・音源数を変えた。')
    scenes = [
        ['易', 'E01', '地震：傾いた壁の下、単一空隙'], ['易', 'E02', '地震：天井材と散在瓦礫'],
        ['易', 'E03', '斜面崩壊：土砂の排水路・暗渠'], ['易', 'E04', '洪水：浸水後の通路、流木、湿潤面'],
        ['易', 'E05', '地震：無人倉庫、機械音を入れた負例'],
        ['中', 'M01', '地震：重なった床板、二段の空隙'], ['中', 'M02', '地震：V字床、露出鉄筋、複数音源'],
        ['中', 'M03', '斜面崩壊：閉塞した分岐暗渠'], ['中', 'M04', '洪水：配管、流木、段差'],
        ['中', 'M05', '地震・浸水：無人地下室、周期的な滴水音'],
        ['難', 'H01', '地震：多層床板、袋小路、複数人'], ['難', 'H02', '地震：階段、鉄筋、不安定な瓦礫片'],
        ['難', 'H03', '斜面崩壊：岩塊、狭窄、二次移動'], ['難', 'H04', '洪水：半水没設備、熱遮蔽、複数音源'],
        ['難', 'H05', '地震：低発声の深部空隙、雑音源'],
    ]
    r.table([['難易度', 'ID', '条件']]+scenes, [.13, .12, .75], size=8.2)
    r.p('現行版は入口に並べず、モデル化した現場全域へ高さ1.05mから12秒間で順次放出する。24・36・48台を切り替え、seedで位置と向きを変える。瓦礫は易64・中104・難144片とし、梁片・鉄筋も衝突と観測遮蔽に使う。')
    r.p('15シーン×3台数の45試験、延べ1620台で放出を確認した。喪失0、推進停止・接続解除後の最寄り接触面までの距離は最大約9.5mmだった。これは接地の診断値であり、探索精度や実機の落下耐性を示す値ではない。［R4］', small=True)
    r.p('対象は局所的な空隙で、建物全体の構造崩壊、土砂の連続体解析、水の流体解析を再現してはいない。', small=True)
    r.end()

    r.start('4　観測が増える過程を地図にする', '未観測の空間を最初から描かず、観測と学習による補完を見分けられる表示にした。')
    r.image(ROOT/'media/rescue-interface-v31.png', '図3　図1の推定地図部分を拡大。青緑は観測、紫は補完。黄橙の人候補、赤紫のガス反応、青い浸水領域をラベルで対応させる。両3D画面はドラッグ回転とホイールズームに対応する。', crop=(810, 125, 1576, 572))
    r.p('距離は最大2m、最大64方向、2Hzで模擬する。距離誤差、姿勢と移動差分の誤差、粉塵による欠落を加える。音響は複数音源の混合、減衰、遮蔽、反射による方向の偏りを、温度は距離と遮蔽物による減衰を近似する。単一の市販センサーが全周を同時に見られるという想定ではない［3］。')
    r.p('表示用の細密点群は、誤差を含む推定姿勢で距離観測の端点を変換し、1cm格子で集約する。2回以上の観測を必要とし、局所平滑化を各軸最大3mmに制限した。1cmは表示の集約間隔で、測定精度ではない。評価と学習に使う占有セルは5cmのままにした。')
    r.table([
        ['表示', '読み取れること', '読み取れないこと'],
        ['黄橙 / 人候補', '音響等から得た未確認候補と優先度', '人の存在の確定、生存の確認'],
        ['赤紫 / ガス反応', 'ノードが観測した相対反応0-1', '漏洩源の確定、ppm、爆発危険度'],
        ['青 / 浸水', '接触式の模擬水深をcmで表示', '水流・浮力や周辺全域の浸水予測'],
    ], [.23, .37, .40])
    r.p('色だけでなく記号・ID・ラベルも併用した。反応がない場所と、まだ観測していない場所は区別する。観測が消えたことを、その場所が安全になった証拠にはしない。', small=True)
    r.end()

    r.start('5　何を、どのデータで学習したか', '地図と人候補を別々に学習し、最後に同じ観測ログをRustで再生して精度を測った。')
    r.table([
        ['学習するもの', '入力と方法'],
        ['距離・音響の補正', 'TRAINの模擬観測と教師値から、距離回帰と声らしさの判別を学習。パワー比のdB換算には10 log10を使用。'],
        ['3D占有地図', '距離観測の占有対数オッズに、学習した占有補正と15形状族の事前分布を統合。形状族は観測の尤度から推定し、正解IDは使わない。'],
        ['人候補の履歴', '観測の継続時間、位置のばらつき、観測ノード数、音響・温度・距離の一致など12特徴を使用。絶対位置を分類特徴に入れない。'],
    ], [.23, .77])
    r.p('v13ではTRAIN 150変種から得た1,040件の候補履歴で、L2正則化ロジスティック分類器をRustのIRLSで学習した。正則化係数0.0001 / 0.001 / 0.01を比較し、VALIDで係数0.0001、判定閾値0.55、候補抑制距離45cmを選んだ。候補数は学習例の件数であり、1,040人を観測したという意味ではない。［R2］')
    r.heading('学習の改善と、到達したVALID精度')
    r.table([
        ['段階', '変更・条件', 'IoU平均', '適合率', '再現率'],
        ['v9', '初期の候補判定\n12台・180秒', pct(v9['map_iou_macro']), pct(v9['human_precision']), pct(v9['human_recall'])],
        ['v11', '候補履歴の学習と\n探索制御改訂 / 180秒', pct(v11['map_iou_macro']), pct(v11['human_precision']), pct(v11['human_recall'])],
        ['v13', '360秒で再収集・地図再学習、\n候補判別の正則化を選択', pct(va['map_iou_macro']), pct(va['human_precision']), pct(va['human_recall'])],
    ], [.10, .36, .18, .18, .18], size=8.1)
    r.p('各段階は15形状族・VALID 30変種で、真の対象は延べ32人。人の観測点は地形とは独立した乱数で配置した。時間予算と探索制御も変わっているため、この差を学習器だけの改善量とは解釈しない。v9からv13で一致数は18人から31人へ増え、誤警報は6件から1件へ減った。', small=True)
    r.p('Pythonはデータ整形、初段の数値学習、評価とモデル選択を担当し、Rustは候補履歴の学習と実行時の推論を担う。全工程をRustだけで学習したとは記載しない。', small=True)
    r.end()

    r.start('6　到達精度をどう確かめたか', 'モデル選択時の成績と、固定後の未知変種に対する成績を分けて示す。')
    r.table([
        ['v13 / 12台・360秒', 'IoU平均 / 合算', '適合率', '再現率'],
        ['VALID 30変種\nモデル・設定の選択用', pct(va['map_iou_macro'])+' / '+pct(va['map_iou_micro']), pct(va['human_precision']), pct(va['human_recall'])],
        ['TEST 45変種\nモデル固定後に測定', pct(te['map_iou_macro'])+' / '+pct(te['map_iou_micro']), pct(te['human_precision']), pct(te['human_recall'])],
    ], [.31, .29, .20, .20])
    r.p('VALIDでは32人中31人が位置一致し、誤警報1件だった。固定後のTESTでは48人中43人が一致し、誤警報1件だった。TESTでも地図と適合率は95%以上を保った一方、人候補の再現率は89.58%となった。ここから、候補を選ぶ際の誤警報の少なさに対して、対象を拾い上げる観測・探索には改善の余地があると読める。［R1］')
    r.p('その後、学習用の変種を450まで増やし、浅い木モデルと音響候補へ近づいて確認する制御を検討した。候補判別には深さ3・60段の木モデルを使い、拡張したVALID 90変種で、地図IoU平均95.59%、適合率・再現率各94.79%（91/96人、誤警報5件）を得た。v13の30変種とは評価集合と制御が異なるため、96.88%からの単純な精度低下や改善とは比較しない。［R3］')
    r.heading('地図の高いIoUは、どこから来たか')
    a = ablation['metrics']
    r.bars(['距離観測だけ', '推定した形状事前分布', 'v11で観測と統合'],
           [a['sensor_only']['map_iou_macro'], a['inferred_family_prior_only']['map_iou_macro'], a['v11_fused']['map_iou_macro']],
           '図4　同じ360秒の記録を使ったv11の切り分け診断（VALID 30）。v13の最終値とは別実験。事前分布単独でも約93%となり、既知の形状族への依存が大きい。［R7］')
    r.p('IoUは5cmセルで評価領域全体を測り、未観測の真の障害物も分母に含める。人候補は25cm以内・一対一対応で一致を判定する。適合率は43/44、再現率は43/48で、同じ「精度」にまとめない。', small=True)
    r.p('評価領域はx=[-0.8,0.8)、y=[-0.15,0.9)、z=[-0.3,3.2)mで固定する。占有閾値は0.6、真値は各セル中央84%の箱とColliderの交差で定義する。結果を見て領域や一致距離を変えない。', small=True)
    r.p('TESTで取りこぼした5人はE03で1人、H03で2人、H04で1人、H05で1人。無人のM05で誤警報が1件あった。場面だけから原因を断定せず、次の観測設計を調べる対象として残した。', small=True)
    r.end()

    r.start('7　現行版へ拡張した意味と課題', '台数を増やした成果は、旧条件の精度と混ぜずに測り直す。')
    r.p('入口から12台を入れる条件では、届かない空間が残る。そこで現在は、24・36・48台を全域へ散布し、中央球にも接続できる形へ変えた。地図の細密化と危険反応表示も加えた。これは、観測を得る機会と画面の読み取りやすさを増やす変更であり、台数に比例して検出精度が上がるとはまだ判断できない。')
    r.table([
        ['区分', '得られた利点', '負担・未検証の点'],
        ['全域への散布', '入口の近くに偏らず、複数地点から観測できる。', '放出位置が既知という仮定。実際の投下損傷・回収・通信は未検証。'],
        ['観測と補完の区別', '地図を作った根拠と、推定した部分を追える。', '姿勢誤差や既知形状の事前分布による誤りは残る。'],
        ['ログによる再生', '物理を再実行せず推論の変更を比べられる。', '探索制御が変わる比較には、対応した物理観測の再収集が必要。'],
    ], [.19, .39, .42])
    r.heading('次に確かめる順序')
    r.p('<b>第一に、新条件の評価をそろえる。</b> 現行3.1はTRAIN 225試行、VALID 90試行、最終確認135試行を事前登録した。同一現場の3台数を学習と評価へ分散させず、各台数と全体で地図IoU・適合率・再現率を測る。現時点の記録には完了した集計がなく、旧12台の数値では代用しない。')
    r.p('<b>第二に、実測で観測モデルを校正する。</b> まず2-5台の機体またはセンサー治具で、乾いた空隙・湿潤面・遮蔽下の距離と音響を測る。特に模擬オドメトリ、反射、熱、接地推進を優先する。同じ生成器で学習回数を増やすだけでは、この誤差は評価できない。')
    r.p('<b>第三に、未知形状と運用の負担を測る。</b> 15形状族と異なる瓦礫配置で再評価し、探索時間、電力、通信、回収可能性も測る。NISTは移動・センサー・通信などを分けた応答ロボット試験を整備しており、この分け方が参考になる［4］。本実装がその標準に適合したという意味ではない。')
    r.end()

    r.start('8　再現手順と参照資料', '提出版の数値は保存済み記録と照合した。氏名・所属・連絡先は記載していない。')
    r.heading('コードを動かして確認する3例')
    r.p('Windows x64、PowerShell 7、Unity 6000.4.1f1、Rust 1.94.1を使用する。リポジトリのcodex/disaster-rescue-v2を取得し、初回はCargoの依存取得とUnityの有効なライセンスが必要。')
    r.p('1. <b>E02・24台で機構を見る。</b> 下記のbuild後、Scene 1を起動する。接続を解除し、屈曲・回転と着地を確認する。<br/><font name="Courier" size="8.3">./scripts/Unity-Task.ps1 build<br/>./scripts/Start-Demo.ps1 -Scene 1 -Nodes 24</font>', small=True)
    r.p('2. <b>E05・36台で負例を見る。</b> 無人条件の機械音とガス反応を、人候補と区別する。<br/><font name="Courier" size="8.3">./scripts/Start-Demo.ps1 -Scene 4 -Nodes 36</font>', small=True)
    r.p('3. <b>H04・48台で複数の反応を見る。</b> 一時停止し、両画面を回転・拡大して候補、水深、接続状態を読む。<br/><font name="Courier" size="8.3">./scripts/Start-Demo.ps1 -Scene 13 -Nodes 48</font>', small=True)
    r.heading('成果物と結果の出典')
    r.p(f'<link href="{BRANCH}" color="#087f78">公開コード・現行ブランチ</link>　/　<link href="{VIDEO}" color="#087f78">動作・構想動画（2分29.5秒）</link><br/>GitHub: {REPO}<br/>YouTube: {VIDEO}', small=True)
    evidence = [
        ('R1', 'reports/rescue/mission360/learning-v13-valid.json', 'v13のVALID / TESTは同階層のlearning-v13-test.json。'),
        ('R2', 'models/rescue-v13/training-report.json', '学習条件。選択は同階層のselection-report.json。'),
        ('R3', 'reports/rescue/mission360-probe-valid90/baseline-v17-valid90-summary.json', '450変種学習後のVALID 90集計。'),
        ('R4', 'reports/rescue/distributed-deployment-smoke.json', '45試験・1620台の散布と接地診断。'),
        ('R5', 'reports/rescue/spherical-docking-smoke.json', '先端・球の接続試験。'),
        ('R6', 'reports/rescue/climbing-diagnostic.json', '25mm段差の乗り越えと迂回。'),
        ('R7', 'reports/rescue/mission360/ablation-v11-valid.json', '形状事前分布への依存を調べた診断。'),
    ]
    for label, path, desc in evidence:
        r.p(f'<link href="{REPO}/blob/codex/disaster-rescue-v2/{path}" color="#087f78">［{label}］{escape(path)}</link><br/>{desc}', small=True, gap=3)
    r.p('v13モデルSHA-256：<font name="Courier" size="6.9">'+model_hash+'</font>', small=True, gap=6)
    r.heading('一次資料（2026年10月29日確認）')
    refs = [
        ('1', '警察庁「令和6年能登半島地震に伴う警察活動について」2024年1月10日', 'https://www.npa.go.jp/bureau/security/biki/disaster/20240110.pdf'),
        ('2', 'Unity 6.4 Manual: Configurable Joint component reference', 'https://docs.unity3d.com/6000.4/Documentation/Manual/class-ConfigurableJoint.html'),
        ('3', 'STMicroelectronics: VL53L5CX Datasheet, DS13754 Rev 13', 'https://www.st.com/resource/en/datasheet/vl53l5cx.pdf'),
        ('4', 'NIST: Standard Test Methods for Response Robots', 'https://www.nist.gov/el/intelligent-systems-division-73500/standard-test-methods-response-robots'),
    ]
    for n, title, url in refs:
        r.p(f'<link href="{url}" color="#087f78">［{n}］{escape(title)}</link>', small=True, gap=3)
    r.p('図1・3：本アプリの実行画面。図2：Blenderによる機構説明CG。動画の音声は音読さん、音楽はSuno AI、編集・音声調整・出力はDaVinci Resolveを使用した。媒体の出典と利用条件はmedia/CREDITS.mdに記録した。', small=True)
    r.end()
    r.save()
    reader = PdfReader(PDF)
    text = '\n'.join(p.extract_text() for p in reader.pages)
    assert len(reader.pages) == 9
    for token in ['95.67%', '96.88%', '95.05%', '97.73%', '89.58%', '1,040', '450', '94.79%']:
        assert token in text, f'Missing fact: {token}'
    assert not re.search(r'rynat|@gmail|@outlook|C:\\Users|氏名[:：]|所属[:：]', text, re.I)
    assert reader.metadata.author in ('', None)
    sources = [p for _, p, _ in evidence] + [
        'reports/rescue/mission360/learning-v13-test.json',
        'reports/rescue/learning-v9-valid-policy1.json',
        'reports/rescue/learning-v11-valid-policy2.json',
        'models/rescue-v13/model.json',
        'models/rescue-v13/selection-report.json',
        'reports/rescue/swarm-v31/protocol-design.json',
    ]
    checks = {'pages': len(reader.pages), 'pdf_sha256': hashlib.sha256(PDF.read_bytes()).hexdigest(),
              'model_sha256': model_hash, 'layout': r.layout,
              'fact_assertions': 'pass', 'privacy_text_and_metadata': 'pass',
              'sources': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}}
    (TMP/'checks.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'pdf': str(PDF), 'pages': len(reader.pages), 'bytes': PDF.stat().st_size,
                      'layout': r.layout}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
