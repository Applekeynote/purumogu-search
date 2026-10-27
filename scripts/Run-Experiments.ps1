param([switch]$Collect,[switch]$Faults,[int]$Port=47110)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path $PSScriptRoot -Parent
$dataset=Join-Path $projectRoot 'datasets\generated\v1'
Push-Location $projectRoot
try {
    cargo build --manifest-path rust/Cargo.toml --release --locked
    if($LASTEXITCODE){throw 'Rust build failed'}
    $server=Join-Path $projectRoot 'rust\target\release\ghost_flow_server.exe'
    if($Collect){
        & "$PSScriptRoot\Unity-Task.ps1" collect -Output $dataset
        $newModel=Join-Path $projectRoot ("models\m-regenerated-"+[guid]::NewGuid().ToString('N').Substring(0,8)+'\model.json')
        & $server train --events "$dataset\observations.jsonl" --truth "$dataset\truth.jsonl" --output $newModel
        if($LASTEXITCODE){throw 'Training failed'}
        $model=$newModel
    } else {$model=Join-Path $projectRoot 'models\m-v1\model.json'}
    $rustProcess=Start-Process -FilePath $server -ArgumentList @('serve','--model',"`"$model`"",'--port',$Port,'--logs',"`"$(Join-Path $projectRoot 'experiments\raw')`"") -WindowStyle Hidden -PassThru
    try {
        # Bounded readiness probe; it opens no run and transmits no data.
        $ready=$false
        for($attempt=0;$attempt -lt 40;$attempt++){try{$client=New-Object Net.Sockets.TcpClient('127.0.0.1',$Port);$client.Dispose();$ready=$true;break}catch{Start-Sleep -Milliseconds 100}}
        if(!$ready){throw 'Rust did not bind'}
        $extra=@('-purumoguPort',"$Port")
        if($Faults){$extra+=@('-purumoguFaults','true','-purumoguTestMaps','2','-purumoguRepeats','1')}
        & "$PSScriptRoot\Unity-Task.ps1" compare -Output $dataset -Extra $extra
        $stem=if($Faults){'fault'}else{'comparison'}
        & $server evaluate --model $model --events "$dataset\$stem.jsonl" --truth "$dataset\$stem-truth.jsonl" --output "reports/$stem"
        if($LASTEXITCODE){throw 'Evaluation failed'}
        if($Faults){& $server evaluate --model $model --events "$dataset\$stem.jsonl" --truth "$dataset\$stem-truth.jsonl" --split ood --output reports/ood}
    } finally {if(!$rustProcess.HasExited){Stop-Process -Id $rustProcess.Id}}
} finally {Pop-Location}
