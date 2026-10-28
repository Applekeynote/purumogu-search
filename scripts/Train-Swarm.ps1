param([string]$Python='python',[ValidatePattern('^[a-zA-Z0-9-]+$')][string]$Experiment='swarm-v31',[string]$Prefix='swarm-v31')
$ErrorActionPreference='Stop'
$projectRoot=Split-Path $PSScriptRoot -Parent
Push-Location $projectRoot
try {
    $raw="experiments/raw/rescue/$Experiment";$reports="reports/rescue/$Experiment"
    function Complete-Manifest([string]$Path,[int]$Expected,[string]$CountField){
        if(!(Test-Path -LiteralPath $Path)){return $false}
        try{$value=Get-Content -LiteralPath $Path -Raw|ConvertFrom-Json}catch{return $false}
        if($CountField -eq 'expected_maps' -and $value.status -ne 'complete'){return $false}
        return $value.$CountField -eq $Expected
    }
    function Run-Python([string[]]$Arguments){& $Python @Arguments;if($LASTEXITCODE){throw "Training stage failed: $($Arguments[0])"}}
    # These are current-job dependencies, not a scheduled task. Never open TEST here.
    while(!( (Complete-Manifest "$reports/train/manifest.json" 225 'expected_maps') -and
             (Complete-Manifest "$reports/valid/manifest.json" 90 'expected_maps') -and
             (Complete-Manifest "$raw/sensor-maps/manifest-train.json" 225 'maps') -and
             (Complete-Manifest "$raw/sensor-maps/manifest-valid.json" 90 'maps') -and
             (Complete-Manifest "$raw/track-proposals/manifest-train.json" 225 'maps') -and
             (Complete-Manifest "$raw/track-proposals/manifest-valid.json" 90 'maps'))){Start-Sleep -Seconds 5}
    $geometry="$Prefix-geometry";$selectedGeometry="$Prefix-geometry-selected"
    Run-Python -Arguments @('scripts/train_rescue_geometry.py','--raw',$raw,'--report',"$reports/train",'--base','models/swarm-v3-bootstrap/model.json','--name',$geometry,'--replace-splits','--trainer','rust/target/release/train_tracks.exe')
    Run-Python -Arguments @('scripts/select_rescue_geometry.py','--raw',$raw,'--source',"models/$geometry/model.json",'--output-model',$selectedGeometry,'--report',"$reports/geometry-selection.json")
    $settings=@(@{Name="$Prefix-linear";Extra=@()},@{Name="$Prefix-boost-d2-r60";Extra=@('--boost','--depth','2','--rounds','60')},@{Name="$Prefix-boost-d2-r100";Extra=@('--boost','--depth','2','--rounds','100')},@{Name="$Prefix-boost-d3-r60";Extra=@('--boost','--depth','3','--rounds','60')},@{Name="$Prefix-boost-d3-r100";Extra=@('--boost','--depth','3','--rounds','100')})
    $candidates=@()
    foreach($setting in $settings){
        Run-Python -Arguments (@('scripts/train_rescue_tracks.py','--raw',$raw,'--report',"$reports/valid",'--base',"models/$selectedGeometry/model.json",'--name',$setting.Name,'--proposal-model','models/swarm-v3-bootstrap/model.json','--trainer','rust/target/release/train_tracks.exe')+$setting.Extra)
        $fit=Get-Content -LiteralPath "models/$($setting.Name)/training-report.json" -Raw|ConvertFrom-Json
        $values=@();foreach($entry in $fit.selected.by_node_count.PSObject.Properties){$values+=@($entry.Value.human_precision,$entry.Value.human_recall)}
        $candidates+= [pscustomobject]@{Name=$setting.Name;Worst=($values|Measure-Object -Minimum).Minimum;Overall=[Math]::Min($fit.selected.human_precision,$fit.selected.human_recall);Sum=$fit.selected.human_precision+$fit.selected.human_recall;False=$fit.selected.false_alarms}
    }
    $best=$candidates|Sort-Object @{Expression='Worst';Descending=$true},@{Expression='Overall';Descending=$true},@{Expression='Sum';Descending=$true},@{Expression='False';Descending=$false}|Select-Object -First 1
    $model="models/$($best.Name)/model.json"
    Run-Python -Arguments @('scripts/evaluate_rescue_replay.py','--raw',$raw,'--rollout-report',"$reports/valid",'--model',$model,'--server','rust/target/release/rescue_server.exe','--output',"$reports/learning-valid.json",'--workers','3')
    $metrics=Get-Content -LiteralPath "$reports/learning-valid.json" -Raw|ConvertFrom-Json
    [ordered]@{status=$(if($metrics.target_met){'valid_passed_confirmation_pending'}else{'valid_failed'});model=$model;model_sha256=$metrics.model_sha256;server_sha256=$metrics.server_sha256;candidates=$candidates;valid_target_met=$metrics.target_met;test_accessed=$false;note='TRAIN-only coefficients; VALID-only selection; independent Rust replay; TEST remains unopened'}|ConvertTo-Json -Depth 8|Set-Content -LiteralPath "$reports/training-workflow.json" -Encoding utf8NoBOM
    Write-Host "VALID target met: $($metrics.target_met). TEST remains unopened. Model: $model"
} finally {Pop-Location}
