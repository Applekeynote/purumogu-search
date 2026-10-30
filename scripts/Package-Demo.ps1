param([string]$Model='models/swarm-v3-bootstrap/model.json',[string]$Output='dist/PurumoguSearch-Windows-v3.3.zip',[string]$Folder='dist/PurumoguSearch-Windows-v3.3',[string]$Server='rust/target/release/rescue_server.exe')
$ErrorActionPreference='Stop'
$projectRoot=Split-Path $PSScriptRoot -Parent
$playerFolder=Join-Path $projectRoot 'build'
if(!(Test-Path -LiteralPath (Join-Path $playerFolder 'PurumoguSearch.exe'))){throw 'Build the Unity player first'}
$stage=Join-Path $projectRoot $Folder
$outputPath=Join-Path $projectRoot $Output
if((Test-Path -LiteralPath $stage) -or (Test-Path -LiteralPath $outputPath)){throw 'Package output already exists; choose new Folder and Output names'}
New-Item -ItemType Directory -Path $stage -Force | Out-Null
Get-ChildItem -LiteralPath $playerFolder -Force | Where-Object {$_.Name -notmatch 'BackUpThisFolder|BurstDebugInformation'} | Copy-Item -Destination $stage -Recurse
$inference=Join-Path $stage 'PurumoguSearch_Data/StreamingAssets/inference'
New-Item -ItemType Directory -Path $inference -Force | Out-Null
Copy-Item -LiteralPath $(if([IO.Path]::IsPathRooted($Server)){$Server}else{Join-Path $projectRoot $Server}) -Destination (Join-Path $inference 'rescue_server.exe') -Force
Copy-Item -LiteralPath (Join-Path $projectRoot $Model) -Destination (Join-Path $inference 'model.json') -Force
Copy-Item -LiteralPath (Join-Path $projectRoot 'LICENSE') -Destination $stage
$fingerprints=[ordered]@{version='3.3';created_utc=[DateTime]::UtcNow.ToString('o');source_model=$Model;source_server=$Server;steps=18000;seconds=360;node_counts=@(24,36,48);deployment='WholeSceneDrop';launch='PurumoguSearch.exe';inference='owned hidden process, ephemeral loopback port, Windows kill-on-close job';files=[ordered]@{}}
foreach($name in @('PurumoguSearch.exe','UnityPlayer.dll','PurumoguSearch_Data/StreamingAssets/inference/rescue_server.exe','PurumoguSearch_Data/StreamingAssets/inference/model.json')){$fingerprints.files[$name]=(Get-FileHash -LiteralPath (Join-Path $stage $name) -Algorithm SHA256).Hash.ToLowerInvariant()}
$fingerprints | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $stage 'PACKAGE-MANIFEST.json') -Encoding utf8NoBOM
$instructions=@"
ぷるもぐサーチ / Windows x64 / 3.3

ZIPをフォルダーごと展開し、PurumoguSearch.exe をダブルクリックしてください。
「散布・探索を開始」で開始します。

Unity・Unity Hub・Rust・Python・PowerShellの別途インストールは不要です。
同梱のRust推論プロセスはアプリが自動起動し、アプリ終了時に停止します。
同時に複数のアプリを起動しても、別のポートと推論プロセスを使用します。
EXEと同じフォルダーのDLL、PurumoguSearch_Data、MonoBleedingEdgeも必要です。
EXEファイルだけを抜き出さず、フォルダー全体を配布してください。

・24 / 36 / 48台をリセット後に切り替えられます。
・「配置を変える」で散布位置と要救助者の位置が変わります。
・左上の赤球は要救助者の正解位置です。表示設定で切り替えられます。
・右上は黒背景と緑の細いワイヤーフレームだけの反響マップです。
・判断欄・凡例・断面・候補マーカー・波紋・点群は右上に表示しません。
・両3D画面をドラッグで回転、ホイールで拡大・縮小できます。
・表面を連続した四角格子で描き、観測が得られた場所から表示します。
・右下の評価は元の推論データで計算し、表示の滑らかさとは区別します。
・正解位置を推論へ送りません。

このアプリは研究用シミュレーションで、実機救助性能は未検証です。
同梱モデルは24・36・48台条件の開発モデルです。旧12台の評価結果とは区別してください。
https://github.com/Applekeynote/purumogu-search/tree/codex/disaster-rescue-v2
"@
Set-Content -LiteralPath (Join-Path $stage 'README.txt') -Value $instructions -Encoding utf8NoBOM
New-Item -ItemType Directory -Force -Path (Split-Path $outputPath -Parent) | Out-Null
Compress-Archive -LiteralPath (Get-ChildItem -LiteralPath $stage -Force).FullName -DestinationPath $outputPath -CompressionLevel Optimal
Get-FileHash -LiteralPath $outputPath -Algorithm SHA256
