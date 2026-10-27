# v2の学習と再現

推論を試すだけなら、UnityプレイヤーとRustサービス、公開済みモデルを使います。学習・PDF・動画素材の生成にはPython 3.12と`requirements-training.txt`の依存が必要です。音声は音読さん、音楽はSuno AIで作成し、動画編集とミックス、最終出力はDaVinci Resolveで行います。

## 分割と観測予算

- TRAIN：15シーン × 10変種、seed 20400000 + 997*r（r=0..9）。
- VALID：15シーン × 2変種、seed 20500000 + 997*r（r=0..1）。
- 最初のTEST：15シーン × 3変種、seed 20600000 + 997*r（r=0..2）。v13固定後に開き、再現率89.58%で不合格となった記録を保存しています。
- 改訂時はVALIDに60変種（20501994 + 997*r、r=0..3）を追加し、合計90変種で選びます。次の未使用確認は75変種（20700000 + 997*r、r=0..4）と改善前に固定しています。

v13の長時間実験は1変種360秒、12台。旧180秒実験とフォルダーを分けます。形状、要救助者の観測点、ノイズの生成条件は同じで、観測時間と探索制御・学習器の違いを記録します。物理計算はPC間のビット単位の一致を保証せず、データとモデルのSHA-256を実行ごとに残します。

## TRAINとVALIDを集める

以下は別の実験名`retrain`を使う例です。モデルv11の音響係数による探索・確認停止を固定し、地図補正と候補判別だけを新たに学習します。v13でも音響・距離回帰の係数は同じです。

```powershell
python -m pip install -r requirements-training.txt
./scripts/Run-Rescue-Evaluation.ps1 -Experiment retrain -Split train -Steps 18000 -Repeats 10 -SeedBase 20400000 -Model models/rescue-v11/model.json -Guidance -Diagnostics
./scripts/Run-Rescue-Evaluation.ps1 -Experiment retrain -Split valid -Steps 18000 -Repeats 2 -SeedBase 20500000 -Model models/rescue-v11/model.json -Guidance -Diagnostics
python scripts/prepare_rescue_sensor_maps.py --split train --raw experiments/raw/rescue/retrain
python scripts/prepare_rescue_sensor_maps.py --split valid --raw experiments/raw/rescue/retrain
```

収集フォルダーは`experiments/raw/rescue/retrain`、実行条件と成否は`reports/rescue/retrain`です。進行中の収集を学習へ混ぜず、manifestのcompleteと予定件数を確認します。未補正マップは記録済みセンサーをモデルなしで再生した結果です。補正済み地図を補正器の教師入力へ戻しません。正解地形・人ラベル・人位置はオフライン学習と評価だけが使います。

## Rustで補正と候補履歴を学習する

```powershell
python scripts/train_rescue_geometry.py --raw experiments/raw/rescue/retrain --report reports/rescue/retrain/train --base models/rescue-v11/model.json --name my-geometry
python scripts/select_rescue_geometry.py --raw experiments/raw/rescue/retrain --source models/my-geometry/model.json --output-model my-geometry-selected --report reports/rescue/retrain/geometry-selection.json
python scripts/export_rescue_tracks.py --raw experiments/raw/rescue/retrain --report reports/rescue/retrain/train --split train
python scripts/train_rescue_tracks.py --raw experiments/raw/rescue/retrain --report reports/rescue/retrain/valid --base models/my-geometry-selected/model.json --proposal-model models/rescue-v11/model.json --name my-final --ridge .0001
python scripts/evaluate_rescue_replay.py --raw experiments/raw/rescue/retrain --rollout-report reports/rescue/retrain/valid --model models/my-final/model.json --output reports/rescue/retrain/valid-replay.json
```

Pythonは入力整形とVALIDでの選択を行い、Rustの`train_tracks`バイナリがIRLSで係数を学習します。地図補正には各TRAIN変種から一様抽出した3000セルを使い、形状事前分布は当該変種を除いたleave-one-map-outで教師を作ります。候補判別は絶対位置を特徴に入れず、継続時間、ノード数、位置分散、音響・温度・距離の一致を使います。位置25cm以内のラベルづけはオフラインだけです。

v13では候補判別のL2係数 .0001 / .001 / .01をTRAINで学習し、VALIDの適合率・再現率で選びました。検出閾値と候補抑制距離もVALIDで選択しています。学習マップIDとVALIDの重複を拒否し、TESTはこの処理に読み込みません。最終TESTの収集には`-FinalTest`と、モデル・バイナリ・物理ソース・時間予算を固定した`protocol-lock.json`が必要です。保存済みTESTの上書きも拒否します。

改訂では、同じTRAIN候補から浅いNewton木を学習して線形モデルと比較します。葉のL2係数5、最小12候補、学習率0.1を固定し、深さ2/3と60/100ラウンドの4候補をVALID90で比較します。絶対座標を特徴に追加しません。損失の二次近似と正則化した葉の更新は[Chen・Guestrinの原論文](https://arxiv.org/abs/1603.02754)の式6・7を参考にした小規模な独自実装です。XGBoostライブラリを使用したとの主張ではありません。

## 出力を確認する3例

1. E02で機体を拡大し、球を軸に曲げ・回転と接続解除を確かめます。断面表示を切ると、隠していた天井材も見られます。Colliderはどちらの表示でも残ります。
2. M02で複数音源を見ます。候補の探索半径は未校正の目安であり、人の存在確率や95%信頼区間ではありません。
3. E05またはM05の無人条件で、誤警報を確かめます。候補が出ない表示は保留であり、人の不在を断定しません。

利点は物理・観測・推論を別々に再生できること。欠点は6分の観測予算、模擬オドメトリ、簡略化した音響・熱・水、15形状族への依存です。新しい倒壊形態への一般化は別に検証する必要があります。実測センサーと小規模実機による校正を、同じ生成器での学習回数追加より優先して検証するのが次の代替案です。
