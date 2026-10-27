param([string]$Model='models/swarm-v3-bootstrap/model.json',[string]$Output='dist/PurumoguSearch-Windows.zip')
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
$fingerprints=[ordered]@{created_utc=[DateTime]::UtcNow.ToString('o');source_model=$Model;steps=18000;seconds=360;files=[ordered]@{}}
foreach($name in @('PurumoguSearch.exe','rescue_server.exe','model.json')){$fingerprints.files[$name]=(Get-FileHash -LiteralPath (Join-Path $stage $name) -Algorithm SHA256).Hash.ToLowerInvariant()}
$fingerprints | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $stage 'PACKAGE-MANIFEST.json') -Encoding utf8NoBOM
$launcher=@'
@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -Command "$ErrorActionPreference='Stop'; $probe=[Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback,47120); try{$probe.Start()}catch{throw 'Port 47120 is busy. Close another demo first.'}finally{$probe.Stop()}; $service=Start-Process -FilePath '.\rescue_server.exe' -ArgumentList @('--port','47120','--model','model.json') -WindowStyle Hidden -PassThru; try { if($service.WaitForExit(150)){throw 'Inference server failed to start'}; Start-Process -FilePath '.\PurumoguSearch.exe' -ArgumentList @('--port','47120','--steps','18000') -Wait } finally { if(!$service.HasExited){Stop-Process -Id $service.Id} }"
if errorlevel 1 pause
'@
Set-Content -LiteralPath (Join-Path $stage 'Start-Demo.cmd') -Value $launcher -Encoding ascii
$instructions=@'
Purumogu Search / Windows x64 simulation

Extract the whole ZIP into a writable local folder, then double-click Start-Demo.cmd.
The Unity application opens with 15 scenes, the articulated robot view, sensor-based
map progress, node states and evaluation. Use the Explore button to begin a scene.
Select 24, 36 or 48 nodes after resetting. Drag inside the upper-left physical view
with the middle mouse button (wheel pressed) or the left button to rotate slowly.
Use the view reset button to restore the initial angle. Translucent slabs retain
visible surfaces and outlines; gravity and collisions determine where nodes land.
The included bootstrap supports the new deployment; its 95% accuracy is unconfirmed.
The Rust inference service stops when the application closes. Port 47120 must be free.
PACKAGE-MANIFEST.json identifies the included model and executables by SHA-256.
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
