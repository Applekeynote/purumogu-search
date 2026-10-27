# 設計書v1.1との対応

Phase 1の実装範囲。設計書の将来案・長期性能目標と、今回の動作を区別する。成績・研究結果が良いことを「完成」の条件にすり替えない。

| 要件 | 実装・根拠 | 留意点 |
|---|---|---|
| FR-001〜004 | ScenarioBuilderの5族、Rigidbody、3ゲート、ID・IMU・BLE・受信/ノード時計 | 模型寸法・センサーは教育用の近似 |
| FR-005 | Rustのdeny_unknown_fields、feature manifest、TCP不正入力試験 | OSの別アカウント隔離ではない |
| FR-006 | 重複seq、欠番、古いseq、重複spawn、未来時刻、異常値、最大行/イベント数の検証 | 時系列のイベントは受信後に整列 |
| FR-007・011 | 6候補、事後分布、支持範囲と保留、UI・全試行の保留率 | OTHERの棄却性能は悪く、OODで失敗を実測 |
| FR-008 | 学習した尤度からEIGを計算し入口×隊形を選択 | 各5個の共通予算。履歴は既投入数で近似 |
| FR-008A | Parentステーションから子へモード・推力指令 | 親指令のRF損失は未実装 |
| FR-008B・008C | CLUSTER / CHAIN_INSERT、一時連結・離脱、局所壁追従 | v1.1の最小2隊形。Anchor/Relay/Squeezeなどは未実装 |
| FR-009 | GHOST-FLOW / Random / Fixedの同条件400試行、Passiveを追加 | Randomは試行別のpolicy_seedを記録 |
| FR-010 | map単位分割、package_evidenceで3集合の交差を拒否 | 推奨70/15/15比率からは変更 |
| FR-012 | protocol/schema/model/data/build manifest | 人由来兆候の模擬は将来 |
| FR-013 | 実測入出力サンプル、リプレイの確率一致試験 | PhysX再実行のビット単位一致ではない |
| FR-014 | Prepare / Explore / Result / Recover | 回収はステージ清掃・新run化 |
| FR-015 | Rustユニット、実TCP結合、Unity smoke、400連続エピソード | Unityの本格的なEdit/PlayModeテスト群は未整備 |
| FR-016・017 | 1〜3分のMP4、PDF、public GitHub、固定モデル、起動・再学習手順 | 提出先への登録は今回の依頼範囲外 |
| FR-018 | Lite/Mapping/HumanCue/Tracerの型 | センサー実装はLite相当 |
| NFR-001・002 | wire版・run・seq・時刻、ネットワークTask→有界キュー→メインスレッド | ワーカーはUnityEngineを参照しない |
| NFR-003 | 版不一致・切断・タイムアウト・不正応答でSAFE STOP | 既存ノードの推力も無効化 |
| NFR-004 | モデルは不変snapshot。別フォルダーに一時書込後rename | サービス再起動で切替。ホットリロードしない |
| NFR-005 | map/physics/sensor/policy seed、Unity/Rust版、固定stepを記録 | 異なるPhysX環境の完全一致は保証しない |
| NFR-006 | Rustバッチ60,000イベント、TCP1MB、キュー128、run監査64MB | 全量データはディスク空き容量を要する |
| NFR-007・008 | build/data/model manifest。標準化TRAINのみ、温度VALIDのみ | TESTに合わせた事後調整なし |

推奨のsmartcore Random Forest、Burn GNN、任意グラフ探索、3D occupancy、Human-Cue、500個規模、実機は未実装。これらを達成したとは表示しない。今回の価値は、模擬観測→Rustの学習・推論→物理行動→新観測の因果ループと、同条件比較・失敗の再検証である。
