param([int]$Port=47122,[int]$Steps=3000,[int]$Repeats=2,[ValidateSet('train','valid','test','development')][string]$Split='development',[int]$SeedBase=20261027,[string]$Model='',[switch]$Guidance,[switch]$SkipBuild,[ValidateRange(-1,14)][int]$Scene=-1,[switch]$Diagnostics,[ValidatePattern('^[a-zA-Z0-9-]*$')][string]$Experiment='',[string]$UnityProject='',[switch]$FinalTest,[string]$Server='rust/target/release/rescue_server.exe')
$ErrorActionPreference='Stop'
$projectRoot=Split-Path $PSScriptRoot -Parent
Push-Location $projectRoot
try {
    $serverPath=if([IO.Path]::IsPathRooted($Server)){$Server}else{Join-Path $projectRoot $Server}
    if($Split -eq 'test'){
        if(!$FinalTest){throw 'Opening TEST requires -FinalTest and an immutable protocol-lock.json'}
        $lockPath=Join-Path $projectRoot "reports/rescue/$Experiment/protocol-lock.json"
        $testLock=Get-Content -LiteralPath $lockPath -Raw | ConvertFrom-Json
        if(!$Model -or (Get-FileHash -LiteralPath $Model -Algorithm SHA256).Hash.ToLowerInvariant() -ne $testLock.model_sha256){throw 'TEST model differs from the locked model'}
        if($Steps -ne $testLock.steps -or $Repeats -ne $testLock.test_repeats -or $SeedBase -ne $testLock.test_seed_base -or $Scene -ne -1 -or !$Guidance){throw 'TEST configuration differs from the lock'}
        if(Test-Path -LiteralPath (Join-Path $projectRoot "experiments/raw/rescue/$Experiment/grids-test.jsonl")){throw 'TEST already exists; never overwrite it'}
    }
    if(!$SkipBuild){
        if($Server -ne 'rust/target/release/rescue_server.exe'){throw 'Use -SkipBuild with an explicitly selected alternate server'}
        cargo build --manifest-path rust/Cargo.toml --release --locked -p rescue_server
        if($LASTEXITCODE){throw 'Rust build failed'}
    }
    if(!(Test-Path -LiteralPath $serverPath)){throw 'Build the selected rescue_server before using -SkipBuild'}
    if($Split -eq 'test'){
        if((Get-FileHash -LiteralPath $serverPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $testLock.server_sha256){throw 'TEST inference binary differs from the validated binary'}
        foreach($entry in $testLock.physical_source_sha256.PSObject.Properties){
            $sourcePath=if($UnityProject){Join-Path $UnityProject $entry.Name.Substring(6)}else{Join-Path $projectRoot $entry.Name}
            if((Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.Value){throw "TEST physical source changed: $($entry.Name)"}
        }
    }
    $serverArguments=@('--port',$Port)
    if($Model){$serverArguments+=@('--model',"`"$Model`"")}
    $portProbe=[Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback,$Port)
    try {$portProbe.Start()}catch {throw "Inference port $Port is already in use; choose a free port before collecting data"}finally {$portProbe.Stop()}
    $serviceProcess=Start-Process -FilePath $serverPath -ArgumentList $serverArguments -WindowStyle Hidden -PassThru
    try {
        if($serviceProcess.WaitForExit(150)){throw 'Selected inference server exited during startup'}
        $rawRoot=Join-Path $projectRoot 'experiments/raw/rescue'; $reportRoot=Join-Path $projectRoot 'reports/rescue'
        if($Experiment){$rawRoot=Join-Path $rawRoot $Experiment;$reportRoot=Join-Path $reportRoot $Experiment}
        $reportPath=Join-Path $reportRoot $Split
        New-Item -ItemType Directory -Force -Path $rawRoot,$reportPath | Out-Null
        $sourceHashes=[ordered]@{}
        foreach($file in @('unity/Assets/Scripts/RescueScene.cs','unity/Assets/Scripts/RescueRobot.cs','unity/Assets/Scripts/RescueSimulation.cs','unity/Assets/Editor/RescueTests.cs')){$sourcePath=if($UnityProject){Join-Path $UnityProject $file.Substring(6)}else{Join-Path $projectRoot $file};$sourceHashes[$file]=(Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash.ToLowerInvariant()}
        $modelHash=if($Model){(Get-FileHash -LiteralPath $Model -Algorithm SHA256).Hash.ToLowerInvariant()}else{$null}
        $guidancePolicy=if(!$Guidance){'disabled'}elseif($Model -and [regex]::IsMatch([IO.File]::ReadAllText((Join-Path $projectRoot $Model)),'"acoustic_probe"\s*:\s*true')){'observed-free-frontier+raw-acoustic-probe'}else{'observed-free-frontier+heat-voice-dwell'}
        [ordered]@{started_utc=[DateTime]::UtcNow.ToString('o');split=$Split;experiment=$Experiment;steps=$Steps;repeats=$Repeats;seed_base=$SeedBase;scene_filter=$Scene;guidance=[bool]$Guidance;guidance_policy=$guidancePolicy;diagnostics=[bool]$Diagnostics;model=$Model;model_sha256=$modelHash;server_sha256=(Get-FileHash -LiteralPath $serverPath -Algorithm SHA256).Hash.ToLowerInvariant();source_sha256=$sourceHashes} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $reportPath 'rollout-provenance.json') -Encoding utf8NoBOM
        & ./scripts/Unity-Task.ps1 rescue-evaluate -Project $UnityProject -Extra @('-purumoguPort',"$Port",'-purumoguSteps',"$Steps",'-purumoguRepeats',"$Repeats",'-purumoguSplit',$Split,'-purumoguSeedBase',"$SeedBase",'-purumoguSceneFilter',"$Scene",'-purumoguRaw',"`"$rawRoot`"",'-purumoguReport',"`"$reportPath`"",'-purumoguDiagnostics',$(if($Diagnostics){'true'}else{'false'}),'-purumoguGuidance',$(if($Guidance){'true'}else{'false'}))
    }
    finally {if(!$serviceProcess.HasExited){Stop-Process -Id $serviceProcess.Id}}
}finally{Pop-Location}
