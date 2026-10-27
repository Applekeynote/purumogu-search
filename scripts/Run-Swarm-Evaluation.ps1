param(
    [ValidateSet('train','valid','test','development')][string]$Split='development',
    [string]$Model='models/swarm-v3-bootstrap/model.json',
    [ValidatePattern('^[a-zA-Z0-9-]+$')][string]$Experiment='swarm-v3',
    [int]$Port=47128,[int]$Steps=18000,[int]$Repeats=0,[int]$SeedBase=0,[int]$DeploymentSeedBase=0,
    [ValidateSet(0,24,36,48)][int]$Nodes=0,[ValidateRange(-1,14)][int]$Scene=-1,
    [string]$Server='rust/target/swarm/release/rescue_server.exe',[string]$UnityProject='',[switch]$FinalTest
)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path $PSScriptRoot -Parent
Push-Location $projectRoot
try {
    $designPath=Join-Path $projectRoot 'reports/rescue/swarm-v3/protocol-design.json'
    $design=Get-Content -LiteralPath $designPath -Raw | ConvertFrom-Json
    if($Split -ne 'development'){
        $config=$design.splits.$Split
        if(!$Repeats){$Repeats=$config.world_repeats};if(!$SeedBase){$SeedBase=$config.world_seed_base};if(!$DeploymentSeedBase){$DeploymentSeedBase=$config.deployment_seed_base}
        if($Steps -ne $design.mission_steps -or $Repeats -ne $config.world_repeats -or $SeedBase -ne $config.world_seed_base -or $DeploymentSeedBase -ne $config.deployment_seed_base -or $Nodes -ne 0 -or $Scene -ne -1){throw 'TRAIN/VALID/TEST require the complete preregistered paired-world configuration; use development for fixtures'}
    } else {
        if(!$Repeats){$Repeats=1};if(!$SeedBase){$SeedBase=20261027};if(!$DeploymentSeedBase){$DeploymentSeedBase=31800000}
    }
    $modelPath=[IO.Path]::GetFullPath($Model);$serverPath=[IO.Path]::GetFullPath($Server)
    $modelObject=Get-Content -LiteralPath $modelPath -Raw | ConvertFrom-Json
    if($modelObject.distributed_swarm -ne $true){throw 'A distributed_swarm model is required'}
    $rawRoot=Join-Path $projectRoot "experiments/raw/rescue/$Experiment"
    $reportPath=Join-Path $projectRoot "reports/rescue/$Experiment/$Split"
    if((Test-Path -LiteralPath (Join-Path $rawRoot "grids-$Split.jsonl")) -or (Test-Path -LiteralPath (Join-Path $reportPath 'manifest.json'))){throw 'This split already has data; never overwrite a rollout'}
    $sourceHashes=[ordered]@{}
    foreach($file in @('unity/Assets/Scripts/RescueScene.cs','unity/Assets/Scripts/RescueRobot.cs','unity/Assets/Scripts/RescueSimulation.cs','unity/Assets/Scripts/RescueDeployment.cs','unity/Assets/Editor/RescueTests.cs')){$sourcePath=if($UnityProject){Join-Path $UnityProject $file.Substring(6)}else{$file};$sourceHashes[$file]=(Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash.ToLowerInvariant()}
    $modelHash=(Get-FileHash -LiteralPath $modelPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $serverHash=(Get-FileHash -LiteralPath $serverPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $designHash=(Get-FileHash -LiteralPath $designPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if($Split -eq 'test'){
        if(!$FinalTest){throw 'Opening TEST requires -FinalTest after VALID passes and the protocol is locked'}
        $testLock=Get-Content -LiteralPath "reports/rescue/$Experiment/protocol-lock.json" -Raw | ConvertFrom-Json
        if($testLock.valid_target_met -ne $true -or $testLock.model_sha256 -ne $modelHash -or $testLock.server_sha256 -ne $serverHash -or $testLock.protocol_design_sha256 -ne $designHash){throw 'TEST differs from the validated immutable protocol lock'}
        foreach($entry in $sourceHashes.GetEnumerator()){if($testLock.physical_source_sha256.($entry.Key) -ne $entry.Value){throw "TEST source differs from the lock: $($entry.Key)"}}
    }
    $portProbe=[Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback,$Port)
    try{$portProbe.Start()}catch{throw "Inference port $Port is already in use"}finally{$portProbe.Stop()}
    New-Item -ItemType Directory -Force -Path $rawRoot,$reportPath | Out-Null
    [ordered]@{started_utc=[DateTime]::UtcNow.ToString('o');generation='rescue-3.0';split=$Split;experiment=$Experiment;steps=$Steps;repeats=$Repeats;seed_base=$SeedBase;deployment_seed_base=$DeploymentSeedBase;scene_filter=$Scene;node_filter=$Nodes;deployment='WholeSceneDrop';guidance=$true;guidance_policy='observed-free-frontier+raw-acoustic-probe+distributed-budget';diagnostics=$true;model=$Model;model_sha256=$modelHash;server_sha256=$serverHash;source_sha256=$sourceHashes;protocol_design_sha256=$designHash}|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $reportPath 'rollout-provenance.json') -Encoding utf8NoBOM
    $serviceProcess=Start-Process -FilePath $serverPath -ArgumentList @('--port',$Port,'--model',"`"$modelPath`"") -WindowStyle Hidden -PassThru
    try {
        if($serviceProcess.WaitForExit(150)){throw 'Inference server exited during startup'}
        & ./scripts/Unity-Task.ps1 rescue-swarm-evaluate -Project $UnityProject -Extra @('-purumoguPort',"$Port",'-purumoguSteps',"$Steps",'-purumoguRepeats',"$Repeats",'-purumoguSplit',$Split,'-purumoguSeedBase',"$SeedBase",'-purumoguDeploymentSeedBase',"$DeploymentSeedBase",'-purumoguNodeCount',"$Nodes",'-purumoguSceneFilter',"$Scene",'-purumoguRaw',"`"$rawRoot`"",'-purumoguReport',"`"$reportPath`"",'-purumoguDiagnostics','true','-purumoguGuidance','true')
        foreach($entry in $sourceHashes.GetEnumerator()){$sourcePath=if($UnityProject){Join-Path $UnityProject $entry.Key.Substring(6)}else{$entry.Key};if((Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.Value){throw "Rollout source changed during collection: $($entry.Key)"}}
    } finally {if(!$serviceProcess.HasExited){Stop-Process -Id $serviceProcess.Id}}
} finally {Pop-Location}
