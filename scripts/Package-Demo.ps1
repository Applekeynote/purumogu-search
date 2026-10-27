param([string]$Model='models/rescue-v13/model.json',[string]$Output='dist/PurumoguSearch-Windows.zip')
$ErrorActionPreference='Stop'
$projectRoot=Split-Path $PSScriptRoot -Parent
$playerFolder=Join-Path $projectRoot 'build'
if(!(Test-Path -LiteralPath (Join-Path $playerFolder 'PurumoguSearch.exe'))){throw 'Build the Unity player first'}
$stage=Join-Path $projectRoot ('tmp/package-'+[Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $stage | Out-Null
Get-ChildItem -LiteralPath $playerFolder -Force | Where-Object {$_.Name -notmatch 'BackUpThisFolder|BurstDebugInformation'} | Copy-Item -Destination $stage -Recurse
Copy-Item -LiteralPath (Join-Path $projectRoot 'rust/target/release/rescue_server.exe') -Destination $stage
Copy-Item -LiteralPath (Join-Path $projectRoot $Model) -Destination (Join-Path $stage 'model.json')
Copy-Item -LiteralPath (Join-Path $projectRoot 'LICENSE') -Destination $stage
$launcher=@'
@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -Command "$ErrorActionPreference='Stop'; $service=Start-Process -FilePath '.\rescue_server.exe' -ArgumentList @('--port','47120','--model','model.json') -WindowStyle Hidden -PassThru; try { Start-Process -FilePath '.\PurumoguSearch.exe' -ArgumentList @('--port','47120','--steps','18000') -Wait } finally { if(!$service.HasExited){Stop-Process -Id $service.Id} }"
if errorlevel 1 pause
'@
Set-Content -LiteralPath (Join-Path $stage 'Start-Demo.cmd') -Value $launcher -Encoding ascii
$instructions=@'
Purumogu Search / Windows x64 simulation

Extract the whole ZIP into a writable local folder, then double-click Start-Demo.cmd.
The Unity application opens with 15 scenes, the articulated robot view, sensor-based
map progress, node states and evaluation. Use the Explore button to begin a scene.
The Rust inference service stops when the application closes. Port 47120 must be free.
The experiment budget is 360 seconds per scene. Running this player does not launch the Unity Editor. Rebuilding requires the pinned
Unity Editor and its valid license.

This is a simulation prototype, with simplified drive, audio, heat and water models.
Measured aggregate evaluation and limitations: https://github.com/Applekeynote/purumogu-search
The player uses the computer's Japanese fonts when available. No generated film or
music is bundled in this application package.
'@
Set-Content -LiteralPath (Join-Path $stage 'README.txt') -Value $instructions -Encoding ascii
$outputPath=Join-Path $projectRoot $Output
New-Item -ItemType Directory -Force -Path (Split-Path $outputPath -Parent) | Out-Null
if(Test-Path -LiteralPath $outputPath){throw 'Package output already exists; choose a new filename'}
Compress-Archive -LiteralPath (Get-ChildItem -LiteralPath $stage -Force).FullName -DestinationPath $outputPath -CompressionLevel Optimal
Get-FileHash -LiteralPath $outputPath -Algorithm SHA256
