# 3.5で採用した推論モデル

完了済みのR36地図モデル、G75の人物判定器、独立して固定したR28探索モデルを同梱する。`model.json` と二つのONNX、CPUランタイムは、単体動作を確認した世代79のバンドルから内容を変えずに保存した。学習途中のR39や人物順位付けの重みは使っていない。

配布アプリはCPUのみで実行し、Python・CUDAを必要としない。ONNX RuntimeとMicrosoft CRTの利用条件は、同じフォルダーの各LICENSE・NOTICEを参照。推論モデルの学習と到達精度は [完成範囲](../../docs/v35-completion.md) と [学習記録](../../docs/rescue-generic-learning.md) に記載した。全エリアで3指標95%以上を確認したモデルではない。
