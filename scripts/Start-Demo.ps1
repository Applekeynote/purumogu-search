param([int]$Port=47120,[ValidateRange(0,14)][int]$Scene=0,[string]$Model='models/swarm-v3-bootstrap/model.json',[int]$Steps=18000,[string]$Unity='C:\Program Files\Unity\Hub\Editor\6000.4.1f1\Editor\Unity.exe',[ValidateSet(24,36,48)][int]$Nodes=24,[int]$DeploymentSeed=907231)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path $PSScriptRoot -Parent
Push-Location $projectRoot
try {
    $portProbe=[Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback,$Port)
    try {$portProbe.Start()}catch {throw "Inference port $Port is already in use; close the other demo or choose another port"}finally {$portProbe.Stop()}
    cargo build --manifest-path rust/Cargo.toml --release --locked -p rescue_server
    if ($LASTEXITCODE) { throw 'Rust build failed' }
    $server=Join-Path $projectRoot 'rust\target\release\rescue_server.exe'
    $serverArgs=@('--port',$Port)
    $trainedModel=Join-Path $projectRoot $Model
    if(!(Test-Path -LiteralPath $trainedModel)){throw "Trained model missing: $trainedModel"}
    $serverArgs+=@('--model',"`"$trainedModel`"")
    $rustProcess=Start-Process -FilePath $server -ArgumentList $serverArgs -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru
    try {
        if($rustProcess.WaitForExit(150)){throw 'Inference server exited during startup'}
        $player=Join-Path $projectRoot 'build\PurumoguSearch.exe'
        if(Test-Path -LiteralPath $player){Start-Process -FilePath $player -ArgumentList @('--port',$Port,'--scene',$Scene,'--nodes',$Nodes,'--deployment-seed',$DeploymentSeed,'--steps',$Steps) -WindowStyle Normal -Wait}
        else {Start-Process -FilePath $Unity -ArgumentList @('-projectPath',"`"$(Join-Path $projectRoot 'unity')`"",'--port',$Port,'--scene',$Scene,'--nodes',$Nodes,'--deployment-seed',$DeploymentSeed,'--steps',$Steps) -WindowStyle Normal -Wait}
    } finally {if(!$rustProcess.HasExited){Stop-Process -Id $rustProcess.Id}}
} finally {Pop-Location}
