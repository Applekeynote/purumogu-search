param([int]$Port=47110,[string]$Unity='C:\Program Files\Unity\Hub\Editor\6000.4.1f1\Editor\Unity.exe')
$ErrorActionPreference='Stop'
$projectRoot=Split-Path $PSScriptRoot -Parent
Push-Location $projectRoot
try {
    cargo build --manifest-path rust/Cargo.toml --release --locked
    if ($LASTEXITCODE) { throw 'Rust build failed' }
    $server=Join-Path $projectRoot 'rust\target\release\ghost_flow_server.exe'
    $model=Join-Path $projectRoot 'models\m-v1\model.json'
    $logs=Join-Path $projectRoot 'experiments\raw'
    $rustProcess=Start-Process -FilePath $server -ArgumentList @('serve','--model',"`"$model`"",'--port',$Port,'--logs',"`"$logs`"") -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru
    Write-Host "Rust PID $($rustProcess.Id), localhost:$Port. Stop this process after closing Unity."
    $player=Join-Path $projectRoot 'build\PurumoguSearch.exe'
    if(Test-Path -LiteralPath $player){Start-Process -FilePath $player -ArgumentList @('--port',$Port)}
    else {Start-Process -FilePath $Unity -ArgumentList @('-projectPath',"`"$(Join-Path $projectRoot 'unity')`"");Write-Host 'Open Assets/Scenes/Main.unity and press Play.'}
} finally {Pop-Location}
