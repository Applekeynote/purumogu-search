# モデル

`m-v1/model.json` はUnityイベントからRustで学習した固定モデル。手作業の形状別確率表ではない。TRAINで標準化と16符号のコードブックを学び、候補族×6行動ごとの頻度をDirichlet型擬似カウント1で平滑化した。温度はVALIDのlog lossで選んだ。成果物の検証は`SHA256SUMS`で行える。

`OTHER`だけは一様な符号分布を使う。未知形状の校正済み確率ではない。特徴が学習支持範囲を外れた場合や観測不足時も保留するが、この方式で全ての未知形状を拒否できるわけではない。実測の失敗は`reports/ood`に残す。

`feature_schema.json`、`labels.json`、`training_config.json`、`train_manifest.json`、`data_split_hashes.json`に前提を保存する。学習し直す場合は新しい出力フォルダーを選ぶ。serve中のモデルを直接上書きせず、サービス再起動時に新しい成果物を読み込む。
