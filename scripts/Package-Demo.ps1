param([string]$Model='models/swarm-v35-generic/model.json',[string]$Output='dist/PurumoguSearch-Windows-v3.5.zip',[string]$Folder='dist/PurumoguSearch-Windows-v3.5',[string]$Server='rust/target/release/rescue_server.exe',[string]$Version='3.5',[string]$Player='build',[ValidateSet('WholeSceneDrop','RoofAndFacadeDrop')][string]$Deployment='RoofAndFacadeDrop',[ValidateRange(15,256)][int]$Scenes=36)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path $PSScriptRoot -Parent
$playerFolder=if([IO.Path]::IsPathRooted($Player)){$Player}else{Join-Path $projectRoot $Player}
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
$descriptor=Get-Content -LiteralPath (Join-Path $inference 'model.json') -Raw | ConvertFrom-Json
$buildInfoPath=Join-Path $inference 'build-info.json'
if(Test-Path -LiteralPath $buildInfoPath){
    $buildInfo=Get-Content -LiteralPath $buildInfoPath -Raw | ConvertFrom-Json
    if($buildInfo.scene_count -lt 15 -or $buildInfo.scene_count -gt 256 -or @($buildInfo.scene_ids).Count -ne $buildInfo.scene_count){throw 'Invalid player scene manifest'}
    if($PSBoundParameters.ContainsKey('Scenes') -and $Scenes -ne $buildInfo.scene_count){throw 'Scene count differs from the compiled player'}
    $Scenes=[int]$buildInfo.scene_count
}
$volume=$descriptor.volume_file -eq 'volume.onnx'
if($volume){
    $modelDirectory=Split-Path (Join-Path $projectRoot $Model) -Parent
    $export=Get-Content -LiteralPath (Join-Path $modelDirectory 'export.json') -Raw | ConvertFrom-Json
    $runtimeFiles=@('volume.onnx','onnxruntime.dll','onnxruntime_providers_shared.dll','ONNXRUNTIME-LICENSE.txt','ONNXRUNTIME-ThirdPartyNotices.txt','MICROSOFT-CRT-NOTICE.txt')
    if($descriptor.navigation_file){
        if($descriptor.navigation_file -ne 'navigation.onnx'){throw 'Invalid exploration model filename'}
        $runtimeFiles+='navigation.onnx'
    }
    foreach($file in $export.microsoft_crt.files.PSObject.Properties){
        if([IO.Path]::GetFileName($file.Name) -ne $file.Name -or [IO.Path]::GetExtension($file.Name) -ne '.dll'){throw 'Invalid runtime filename'}
        $runtimeFiles+=$file.Name
    }
    foreach($file in $runtimeFiles){Copy-Item -LiteralPath (Join-Path $modelDirectory $file) -Destination (Join-Path $inference $file) -Force}
    if((Get-FileHash -LiteralPath (Join-Path $inference 'volume.onnx') -Algorithm SHA256).Hash.ToLowerInvariant() -ne $descriptor.volume_sha256){throw '3D model fingerprint mismatch'}
    if($descriptor.navigation_file -and (Get-FileHash -LiteralPath (Join-Path $inference 'navigation.onnx') -Algorithm SHA256).Hash.ToLowerInvariant() -ne $descriptor.navigation_sha256){throw 'Exploration model fingerprint mismatch'}
    foreach($file in $export.microsoft_crt.files.PSObject.Properties){if((Get-FileHash -LiteralPath (Join-Path $inference $file.Name) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $file.Value){throw 'CRT fingerprint mismatch'}}
}
Copy-Item -LiteralPath (Join-Path $projectRoot 'LICENSE') -Destination $stage
$fingerprints=[ordered]@{version=$Version;created_utc=[DateTime]::UtcNow.ToString('o');source_model=$Model;source_server=$Server;steps=18000;seconds=360;node_counts=@(24,36,48);scenes=$Scenes;geometry_inference=$(if($volume){'sensor-only learned 3D completion; portable ONNX CPU'}else{'sensor-only local planes and supported 3D routes'});deployment=$Deployment;launch='PurumoguSearch.exe';inference='owned hidden process, ephemeral loopback port, Windows kill-on-close job';files=[ordered]@{}}
if(Test-Path -LiteralPath $buildInfoPath){$fingerprints.files['PurumoguSearch_Data/StreamingAssets/inference/build-info.json']=(Get-FileHash -LiteralPath $buildInfoPath -Algorithm SHA256).Hash.ToLowerInvariant()}
foreach($name in @('PurumoguSearch.exe','UnityPlayer.dll','PurumoguSearch_Data/StreamingAssets/inference/rescue_server.exe','PurumoguSearch_Data/StreamingAssets/inference/model.json')){$fingerprints.files[$name]=(Get-FileHash -LiteralPath (Join-Path $stage $name) -Algorithm SHA256).Hash.ToLowerInvariant()}
if($volume){foreach($name in $runtimeFiles){$fingerprints.files['PurumoguSearch_Data/StreamingAssets/inference/'+$name]=(Get-FileHash -LiteralPath (Join-Path $inference $name) -Algorithm SHA256).Hash.ToLowerInvariant()}}
$fingerprints | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $stage 'PACKAGE-MANIFEST.json') -Encoding utf8NoBOM
$completionInstruction=if($volume){'・距離観測と学習した3D構造から欠測を推定します。実観測と推定は線の明るさ・破線で区別します。推定には誤りがあり、未観測領域の安全を保証しません。'}else{'・周辺観測の局所平面から小さな欠測を補います。根拠のない大きな未観測領域は埋めません。'}
$deploymentInstruction=if($Deployment -eq 'RoofAndFacadeDrop'){'・標準では上空と複数階の破損開口から分散投入します。「投入」で上空のみへ切り替えられます。'}else{'・現場全体へ上空から分散投入します。'}
$sceneInstruction=if($Scenes -ge 36){'・全36シーン。多形状・二層・三層に加え、C01〜C06で残存床・落下床片・層状瓦礫の部分崩落を再現します。シーン選択の「地形も変える」で欠損・層間隔・瓦礫の配置も更新できます。'}elseif($Scenes -ge 30){'・既存15に、L/T/Y/U字・中庭・二層・三層の15シーンを追加しています。シーン一覧はスクロールできます。'}else{'・災害別・難易度別の15シーンを選べます。'}
$instructions=@"
ぷるもぐサーチ / Windows x64 / $Version

ZIPをフォルダーごと展開し、PurumoguSearch.exe をダブルクリックしてください。
「散布・探索を開始」で開始します。

Unity・Unity Hub・Rust・Python・PowerShellの別途インストールは不要です。
同梱のRust推論プロセスはアプリが自動起動し、アプリ終了時に停止します。
同時に複数のアプリを起動しても、別のポートと推論プロセスを使用します。
EXEと同じフォルダーのDLL、PurumoguSearch_Data、MonoBleedingEdgeも必要です。
EXEファイルだけを抜き出さず、フォルダー全体を配布してください。

・24 / 36 / 48台をリセット後に切り替えられます。
$deploymentInstruction
・「配置を変える」で散布位置と要救助者の位置が変わります。
・左上の赤球は要救助者の正解位置です。表示設定で切り替えられます。
・右上は緑の格子を基本とし、人候補は橙、浸水反応は青、ガス反応は赤で格子を色分けします。
・推定補完は暗い破線で表示し、観測の実線と区別します。
$sceneInstruction
・判断欄・断面・候補マーカー・波紋・点群は右上に表示しません。
・両3D画面をドラッグで回転、ホイールで拡大・縮小できます。
$completionInstruction
・複数階では、観測された床支持と自由空間を使って3D経路を推定します。
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
