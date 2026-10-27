# 映像の構成

`purumogu-demo.mp4` は115秒、1920×1080、30fps。DaVinci Resolve Studio 21.1で編集・音声調整・書き出しを行った。声は音読さん、BGMはSuno AI。出典と利用条件は[CREDITS.md](CREDITS.md)。

| 時間 | 内容 |
|---|---|
| 0:00–0:10 | 目的 |
| 0:10–0:35 | ゲート、重力、接触、無線観測 |
| 0:35–0:55 | Rustの候補分布と次の投入 |
| 0:55–1:10 | 一時連結の拡大。別実行であることを明記 |
| 1:10–1:30 | 同じ予算での比較と95%区間 |
| 1:30–1:45 | 未知形状の誤断定と実装の限界 |
| 1:45–1:55 | 音声・音楽・編集・公開先 |

`resolve-edit-plan.json` にフレーム単位の配置と音量を保存した。`resolve_sources/` は未編集のアプリ収録クリップと章カード。元のPNG連番は容量が大きいためGitの対象外だが、`scripts/Record-Demo.ps1` と `scripts/prepare_resolve_media.py` で再収録・素材化できる。

ローカル成果物の `Purumogu_Search_Film.drp` はResolveで再編集できる。メディアがオフラインになった場合は、メディアプールの「選択したクリップを再リンク」で `resolve_sources/` とこのディレクトリ内の音声を指定する。DRP内の個人フォルダー名を公開しないため、Gitには含めていない。
