param([int]$Scene=6,[int]$Port=47121,[int]$Frames=600,[int]$Speed=4,[string]$Shot='rescue-preview',[string]$Model='models/swarm-v3-bootstrap/model.json',[int]$Steps=18000,[string]$Server='rust/target/release/rescue_server.exe',[switch]$Closeup,[ValidateSet(24,36,48)][int]$Nodes=24,[int]$DeploymentSeed=907231,[ValidateSet('dashboard','world','map','robot')][string]$View='dashboard')
$ErrorActionPreference='Stop'
$projectRoot=Split-Path $PSScriptRoot -Parent
$server=Join-Path $projectRoot $Server
$player=Join-Path $projectRoot 'build\PurumoguSearch.exe'
$framesPath=Join-Path $projectRoot "media\frames\$Shot"
if(Test-Path -LiteralPath $framesPath){throw 'Recording directory already exists; choose a fresh -Shot name'}
New-Item -ItemType Directory -Path $framesPath | Out-Null
$serverArgs=@('--port',$Port)
$trainedModel=Join-Path $projectRoot $Model
if(!(Test-Path -LiteralPath $trainedModel)){throw "Trained model missing: $trainedModel"}
$serverArgs+=@('--model',"`"$trainedModel`"")
$portProbe=[Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback,$Port)
try {$portProbe.Start()}catch {throw "Inference port $Port is already in use; choose another recording port"}finally {$portProbe.Stop()}
$service=Start-Process -FilePath $server -ArgumentList $serverArgs -WindowStyle Hidden -PassThru
try {
    if($service.WaitForExit(150)){throw 'Recording inference server exited during startup'}
    $captureWidth=if($View -eq 'dashboard'){1600}else{1920};$captureHeight=if($View -eq 'dashboard'){900}else{1080}
    $viewArguments=if($View -eq 'dashboard'){@()}else{@('--film-view',$View)}
    $process=Start-Process -FilePath $player -ArgumentList (@('-screen-width',$captureWidth,'-screen-height',$captureHeight,'-screen-fullscreen','0','--closeup',$(if($Closeup){'true'}else{'false'}),'--port',$Port,'--scene',$Scene,'--nodes',$Nodes,'--deployment-seed',$DeploymentSeed,'--speed',$Speed,'--steps',$Steps,'--capture-count',$Frames,'--frames',"`"$framesPath`"",'-logFile',"`"$(Join-Path $projectRoot "tmp\record-$Shot.log")`"")+$viewArguments) -WindowStyle Normal -Wait -PassThru
    if($process.ExitCode -ne 0){throw 'Rescue player failed'}
}finally{if(!$service.HasExited){Stop-Process -Id $service.Id}}
