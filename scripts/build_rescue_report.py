"""Typeset the rescue report from a hash-matched model/evaluation; render with Poppler for QA."""
from pathlib import Path
import argparse,hashlib,json,sys
from collections import defaultdict
from xml.sax.saxutils import escape
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/'tmp/media-runtime').exists():sys.path.insert(0,str(ROOT/'tmp/media-runtime'))
from build_report import Report,SMALL,WIDTH


def percentage(value):return f'{value:.1%}'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--metrics',required=True);parser.add_argument('--model',required=True);parser.add_argument('--screenshot',default='media/rescue-screenshot.png');parser.add_argument('--output',default='reports/Purumogu_Rescue_Report.pdf');args=parser.parse_args()
    metric_path=ROOT/args.metrics;model_path=ROOT/args.model;m=json.loads(metric_path.read_text(encoding='utf-8-sig'));model=json.loads(model_path.read_bytes())
    assert hashlib.sha256(model_path.read_bytes()).hexdigest()==m['model_sha256'],'Do not attach scores from a different model'
    assert not set(model['train_maps'])&{row['map_id'] for row in m['rows']}
    assert all(row['map_id'].startswith('rescue-2.5-') for row in m['rows']),'Older fixed-position/SNR diagnostics are not final evidence'
    generation=json.loads((ROOT/'reports/rescue/generation-2.5.json').read_text(encoding='utf-8-sig'))
    screenshot=ROOT/args.screenshot;assert screenshot.exists()
    status='3指標が95%以上' if m['target_met'] else '95%の合格条件には未達'
    split=m['split'].upper();r=Report(ROOT/args.output)

    r.start('ぷるもぐサーチ','7cmの関節機構と着脱する群れで、災害後の局所的な空隙を調べる。Unity × Rustの実装・検証報告。')
    r.p('中央球と二つの扁平な円錐を持つノードを、実際の剛体と関節で動かす。距離・姿勢・音響・温度の模擬観測から、地形と未確認の人の手がかりを推定する。対象は建物全体ではなく、1.5m × 約3.2mの局所空隙である。')
    r.image(screenshot)
    r.p('図1　実アプリの4画面。左上は物理現場と機体拡大、右上は推定地図と音響候補、左下は各機体の状態、右下は真値との比較。この画像は別のデモ実行で、下表の集計結果を代表する試行ではない。',SMALL)
    r.table([['評価データ','全領域3D IoU','検出の適合率','検出の再現率'],[f'{split} / {m["maps"]}変種',percentage(m['map_iou_macro']),percentage(m['human_precision']),percentage(m['human_recall'])]],[WIDTH*.28,WIDTH*.24,WIDTH*.24,WIDTH*.24])
    r.p(f'<b>{status}</b>。人数は{m["people"]}、位置が一致した人は{m["matched"]}、誤警報は{m["false_alarms"]}。分類器のスキャン正解率を、この3指標の代わりには使わない。')
    r.end()

    r.start('機構と観測の前提','機体寸法・質量・駆動力は仮の工学パラメーターであり、実機の測定値ではない。')
    r.table([['項目','実装と確認範囲'],['機体','全長70mm、中央球径19mm、総質量24g。中央球と二つの円錐が別のRigidbodyを持つ。'],['関節と回転','球を軸とする二軸の屈曲と長軸回転。モーターの目標角を与え、姿勢をTransformで演出しない。'],['接続と解除','先端の距離9mm以内・相対向き・速度を確認して接続。2ポート、閉ループなし。荷重破断と停滞時の解除、手動解除に対応。'],['推進と故障','接地中の接線力は理想化。落下・領域外の機体は明示的な喪失状態にし、再配置しない。故障した試行も集計する。'],['距離・姿勢','最大2m、最大64方向、2Hz。姿勢・移動差分に誤差を加え、欠損と粉塵の観測落ちも模擬。'],['音響・温度','複数音源のパワー混合、距離減衰、遮蔽、反射方向の偏り、環境雑音、温度差の減衰を近似。']],[WIDTH*.23,WIDTH*.77])
    r.heading('音の単位を揃える')
    r.p('パワー比は10 log10でdBへ変換する。信号/雑音パワー比が0.1なら-10dB、1なら0dB、10なら10dB。20 log10は振幅比の式であり、旧開発データの誤った換算は最終評価に使わない。観測の混合に人/非人の教師ラベルは使わない。')
    r.p('距離計の参考はSTのVL53L5CX（6.4 × 3 × 1.5mm、8 × 8領域、最大4m）。本モデルの全周観測は複数窓または機体運動の理想化で、同部品一つで全周が見えるとの主張ではない。組み込み寸法・消費電力・粉塵や濡れ面での性能は未検証。',SMALL)
    r.end()

    r.start('15シーンと、独立した配置','易・中・難を各5条件。地震、斜面崩壊、浸水を含む。生存可能性や構造安全性を判定するシミュレーターではない。')
    scenes=[('易','E01','地震：傾いた壁の下'),('易','E02','地震：落下した天井材'),('易','E03','斜面崩壊：土砂の排水路'),('易','E04','洪水：浸水後の通路'),('易','E05','地震：無人倉庫、機械音の負例'),('中','M01','地震：パンケーキ状床板'),('中','M02','地震：V字床、鉄筋、複数音源'),('中','M03','斜面崩壊：閉塞した暗渠'),('中','M04','洪水：流木と湿潤面'),('中','M05','地震・浸水：無人地下室の周期音'),('難','H01','地震：多層床板、袋小路'),('難','H02','地震：階段、鉄筋、不安定な片'),('難','H03','斜面崩壊：岩塊と二次移動'),('難','H04','洪水：半水没設備、遮蔽、複数音源'),('難','H05','地震：低発声の深部空隙')]
    r.table([['難易度','ID','条件']]+scenes,[WIDTH*.12,WIDTH*.14,WIDTH*.74])
    r.p('瓦礫形状と音源配置はseedで変える。音源の乱数列は地形から独立。口元・熱の観測点として、x=-0.5-0.5m、y=0.04-0.12m、z=0.55-2.55mの自由空間へ置く。H05は深部z=1.7-2.65m。点の周囲12mmに遮蔽物がないことと、複数人の点が55cm以上離れることを検査する。人の全身形状や占有体積はモデル化していない。',SMALL)
    r.p('水は摩擦低下と局所抵抗、粉塵は距離観測欠落として近似。流体解析、地盤変形、建物の構造崩壊解析は行わない。',SMALL)
    r.end()

    r.start('学習と評価を混ぜない','正解地形・人ラベル・音源位置は、オフライン学習と評価だけが参照する。推論プロセスへは渡さない。')
    r.table([['分割','変種数','用途'],['TRAIN',len(model['train_maps']),'センサー校正、音響分類、距離回帰、形状の事前分布'],['VALID',len(model['valid_maps']),'モデルと設定の選択'],['今回の集計',f'{split} / {m["maps"]}','固定した指標で全試行を集計']],[WIDTH*.18,WIDTH*.18,WIDTH*.64])
    r.p('シーンIDは通信に含めず、地形族は観測の尤度から推定する。青緑の実測セルと、紫の学習補完を区別する。補完を観測済みセルに数えない。人の配置は変種ごとに変え、学習に使ったmap_idを評価に再利用しない。')
    r.heading('合格条件は3つすべて95%以上')
    r.p('1. 全領域の3D占有IoU：真の占有セルと推定占有セルの交差/和集合。未観測の真の障害物も含める。<br/>2. 人候補の適合率：一致した人の数/全候補数。<br/>3. 人候補の再現率：一致した人の数/真の人数。')
    r.p('地図は5cmセル、x=[-0.8,0.8)、y=[-0.15,0.9)、z=[-0.3,3.2)m。セル中央84%の箱とColliderの交差を真の占有とする。人の一致距離は25cm以内・一対一対応。全マップのIoU平均と、交差/和集合を全マップで合算したIoUの両方を確認する。閾値や領域を結果に合わせて変えない。',SMALL)
    r.p(f'全体IoUの合算値は{m["map_iou_micro"]:.1%}。位置が一致した人{m["matched"]}/{m["people"]}、誤警報{m["false_alarms"]}。'+('今回はモデル選択用のVALID結果で、最終TESTの結果ではない。' if m['split']=='valid' else 'このTESTを係数学習や閾値調整には使用していない。'))
    if m.get('human_precision_wilson95') and m.get('human_recall_wilson95'):
        p=m['human_precision_wilson95'];q=m['human_recall_wilson95'];r.p(f'Wilson 95%区間：適合率{p[0]:.1%}-{p[1]:.1%}、再現率{q[0]:.1%}-{q[1]:.1%}。点推定値の95%到達と、実災害で95%を保証することは異なる。',SMALL)
    r.end()

    r.start('条件ごとの結果','同じ物理時間とノード数で、失敗した試行も含めて測る。無人条件の再現率は定義しない。')
    grouped=defaultdict(list)
    for row in m['rows']:grouped[row['scene']].append(row)
    rows=[['ID','平均IoU','一致/人数','誤警報','適合率','再現率']]
    for scene,items in sorted(grouped.items(),key=lambda v:(v[0][0] not in 'E',v[0][0]=='H',v[0])):
        tp=sum(v['matched'] for v in items);fp=sum(v['false_alarms'] for v in items);people=sum(v['people'] for v in items)
        rows.append([scene,percentage(sum(v['iou'] for v in items)/len(items)),f'{tp}/{people}',str(fp),percentage(tp/(tp+fp)) if tp+fp else '-',percentage(tp/people) if people else '-'])
    r.table(rows,[WIDTH*.10,WIDTH*.19,WIDTH*.19,WIDTH*.14,WIDTH*.19,WIDTH*.19])
    r.p('少数テンプレートの変種に対する結果であり、新しい倒壊形態や実際の音声、未知の構造材への一般化を実証していない。音響候補の優先度は校正済みの人存在確率ではない。保留を人の不在と解釈しない。',SMALL)
    r.heading('重要な弱点と、次の改善')
    r.p('優先順位は、(1) 遮蔽された場所へ観測を取りに行く走行、(2) 複数音源と低発声を分離する音響処理、(3) 距離・姿勢・環境ノイズを実機で校正すること。25mm垂直段差の乗り越えは未確認だが、回り込みは物理試験で確認した。')
    r.end()

    r.start('今日から試す3つの操作','Windows x64 / Unity 6000.4.1f1 / Rust 1.94.1 / CPU学習・推論。Unityの有効なライセンスは各利用者が用意する。')
    command_model=escape(args.model)
    for title,command,note in [('1. アプリを動かす',f'scripts/Unity-Task.ps1 build<br/>scripts/Start-Demo.ps1 -Model {command_model}','シーン選択、ノード拡大、接続解除、断面表示、評価を同じ画面で比較する。'),('2. 機構と通信を検証する','scripts/Unity-Task.ps1 rescue-smoke<br/>python tests/test_rescue_contract.py','接続・解除・屈曲・回転・落下故障と、実TCPの不正入力・重複・分割フレームを確認する。'),('3. 記録した観測を再生する',f'python scripts/evaluate_rescue_replay.py<br/>--model {command_model} --output reports/rescue/replay.json','推論だけを再生する。最終TESTの読み出しには明示フラグが必要。生ログの生成・学習条件は公開リポジトリの手順に従う。')]:
        r.heading(title);r.p('<font name="Courier" size="8.5">'+command+'</font>');r.p(note,SMALL)
    r.p('利点：物理動作、観測、推論、真値評価を同時に比較でき、学習補完と観測を分離できる。欠点：駆動・音響・熱・水は近似で、全身の要救助者形状や実機の制約を十分に表していない。',SMALL)
    r.p('一次資料：<link href="https://docs.unity3d.com/6000.4/Documentation/Manual/class-ConfigurableJoint.html">Unity Configurable Joint</link> / <link href="https://www.st.com/resource/en/datasheet/vl53l5cx.pdf">ST VL53L5CX</link> / <link href="https://www.ni.com/en/shop/labview/using-fast-fourier-transforms-and-power-spectra-in-labview.html">NI パワーとdB</link> / <link href="https://insarag.org/wp-content/uploads/2024/02/AAR_Recommendations-for-INSARAG.pdf">INSARAG 地震対応後の勧告</link>。これらは本モデルの救助性能を認証する資料ではない。',SMALL)
    r.p('映像の音声は音読さん、BGMはSuno AI、編集・ミックス・最終出力はDaVinci Resolve。<link href="https://github.com/Applekeynote/purumogu-search">公開コードと再現手順</link>。',SMALL)
    r.p('モデルSHA-256: '+m['model_sha256']+'<br/>評価真値SHA-256: '+m['truth_sha256'],SMALL)
    r.end();r.save();print(str(ROOT/args.output))


if __name__=='__main__':main()
