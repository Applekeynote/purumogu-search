param([string]$Folder='dist/PurumoguSearch-Windows-v3.4')
$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
$folderPath=[IO.Path]::GetFullPath((Join-Path $root $Folder))
$manifest=Get-Content -Raw -LiteralPath (Join-Path $folderPath 'PACKAGE-MANIFEST.json') | ConvertFrom-Json
foreach($item in $manifest.files.PSObject.Properties){if((Get-FileHash -LiteralPath (Join-Path $folderPath $item.Name)).Hash.ToLowerInvariant() -ne $item.Value){throw "Package hash mismatch: $($item.Name)"}}
$game=$null;$owned=$null
try {
    # The working directory and PATH contain no project, Cargo, Unity Editor, or Python.
    $game=Start-Process -FilePath (Join-Path $folderPath 'PurumoguSearch.exe') -WorkingDirectory $env:TEMP -Environment @{PATH="$env:SystemRoot\System32"} -ArgumentList @('-batchmode','-nographics','-logFile',('"{0}"' -f (Join-Path $root 'tmp/portable-isolated-player.log'))) -WindowStyle Hidden -PassThru
    for($retry=0;$retry -lt 100;$retry++){
        $owned=Get-CimInstance Win32_Process -Filter "ParentProcessId = $($game.Id)" | Where-Object {$_.Name -ieq 'rescue_server.exe'}
        if($owned){break};if($game.HasExited){throw 'Isolated player exited before starting inference'};Start-Sleep -Milliseconds 100
    }
    if(@($owned).Count -ne 1){throw 'Expected exactly one owned inference process'}
    if($owned.ExecutablePath -ne (Join-Path $folderPath 'PurumoguSearch_Data/StreamingAssets/inference/rescue_server.exe')){throw 'Inference came from outside the package'}
    $port=[int]([regex]::Match($owned.CommandLine,'--port (\d+)').Groups[1].Value)
    $client=$null
    for($retry=0;$retry -lt 100;$retry++){try{$client=[Net.Sockets.TcpClient]::new('127.0.0.1',$port);break}catch{Start-Sleep -Milliseconds 50}}
    if(!$client){throw 'Packaged inference did not listen'}
    try {
        $client.ReceiveTimeout=10000;$stream=$client.GetStream();$writer=[IO.StreamWriter]::new($stream,[Text.UTF8Encoding]::new($false));$writer.AutoFlush=$true
        $reader=[IO.StreamReader]::new($stream)
        $packet=@{protocol_version='2.0';run_id='isolated-package';seq_no=0;sim_step=25;cell_m=.05;request_map=$true;scans=@(@{node_id='N00';deployment_anchor=@(0,.025,0);odometry_delta=@(0,0,0);pose_sigma_m=.001;rays=@(@{direction=@(0,0,1);distance_m=.4;hit=$true});bearing=@(0,0,1);bearing_sigma_deg=10;acoustic_snr_db=5;voice_band_ratio=.6;modulation=.6;temperature_delta_c=.2})}
        $writer.WriteLine(($packet|ConvertTo-Json -Depth 8 -Compress));$reply=$reader.ReadLine()|ConvertFrom-Json
        if($reply.protocol_version -ne '2.0' -or !$reply.distributed_swarm -or $reply.cells.Count -lt 1){throw 'Packaged service did not return a real inferred map'}
    }finally{$client.Dispose()}
    Stop-Process -Id $game.Id;$game.WaitForExit()
    for($retry=0;$retry -lt 50;$retry++){if(!(Get-Process -Id $owned.ProcessId -ErrorAction SilentlyContinue)){break};Start-Sleep -Milliseconds 100}
    if(Get-Process -Id $owned.ProcessId -ErrorAction SilentlyContinue){throw 'Owned inference survived a forced player exit'}
    $result=@{status='pass';isolated_working_directory=$true;developer_tools_removed_from_child_path=$true;real_inference_map=$true;forced_exit_child_cleanup=$true;manifest_hashes_verified=$true}
    $result|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $root 'reports/rescue/portable-package-v34.json') -Encoding utf8NoBOM
    $result|ConvertTo-Json
}finally{
    if($game -and !$game.HasExited){Stop-Process -Id $game.Id}
}
