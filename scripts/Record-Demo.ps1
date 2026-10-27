param([int]$Family=4,[int]$Port=47113,[int]$MapSeed=2027261,[int]$SensorSeed=42347,[ValidateSet('wide','macro')][string]$View='wide')
$ErrorActionPreference='Stop'
$projectRoot=Split-Path $PSScriptRoot -Parent
$server=Join-Path $projectRoot 'rust\target\release\ghost_flow_server.exe'
$player=Join-Path $projectRoot 'build\PurumoguSearch.exe'
$model=Join-Path $projectRoot 'models\m-v1\model.json'
$shotName=if($View -eq 'macro'){"family-$Family-macro"}else{"family-$Family"}
$frames=Join-Path $projectRoot "media\frames\$shotName"
New-Item -ItemType Directory -Force -Path $frames | Out-Null
$service=Start-Process -FilePath $server -ArgumentList @('serve','--model',"`"$model`"",'--port',$Port,'--logs',"`"$(Join-Path $projectRoot 'experiments\raw')`"") -WindowStyle Hidden -PassThru
try {
    # A visible interactive player is needed: Unity skips backbuffer rendering when hidden.
    $playerProcess=Start-Process -FilePath $player -ArgumentList @('-screen-width','1600','-screen-height','900','-screen-fullscreen','0','--port',$Port,'--family',$Family,'--map-seed',$MapSeed,'--sensor-seed',$SensorSeed,'--view',$View,'--frames',"`"$frames`"",'-logFile',"`"$(Join-Path $projectRoot "tmp\record-$shotName.log")`"") -WindowStyle Normal -PassThru -Wait
    if($playerProcess.ExitCode -ne 0){throw 'Player recording failed'}
    Write-Host "Recorded $frames"
}finally{if(!$service.HasExited){Stop-Process -Id $service.Id}}
