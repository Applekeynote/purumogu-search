param([int]$Port=47122,[int]$Steps=3000,[int]$Repeats=2,[ValidateSet('train','valid','test','development')][string]$Split='development',[int]$SeedBase=20261027,[string]$Model='',[switch]$Guidance,[switch]$SkipBuild)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path $PSScriptRoot -Parent
Push-Location $projectRoot
try {
    if(!$SkipBuild){
        cargo build --manifest-path rust/Cargo.toml --release --locked -p rescue_server
        if($LASTEXITCODE){throw 'Rust build failed'}
    }
    if(!(Test-Path -LiteralPath (Join-Path $projectRoot 'rust/target/release/rescue_server.exe'))){throw 'Build rescue_server before using -SkipBuild'}
    $args=@('--port',$Port)
    if($Model){$args+=@('--model',"`"$Model`"")}
    $server=Start-Process -FilePath (Join-Path $projectRoot 'rust/target/release/rescue_server.exe') -ArgumentList $args -WindowStyle Hidden -PassThru
    try {& ./scripts/Unity-Task.ps1 rescue-evaluate -Extra @('-purumoguPort',"$Port",'-purumoguSteps',"$Steps",'-purumoguRepeats',"$Repeats",'-purumoguSplit',$Split,'-purumoguSeedBase',"$SeedBase",'-purumoguGuidance',$(if($Guidance){'true'}else{'false'}))}
    finally {if(!$server.HasExited){Stop-Process -Id $server.Id}}
}finally{Pop-Location}
