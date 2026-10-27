param(
    [ValidateSet('init','smoke','collect','compare','build','rescue-smoke','rescue-evaluate','rescue-audit')][string]$Task='rescue-smoke',
    [string]$Unity='C:\Program Files\Unity\Hub\Editor\6000.4.1f1\Editor\Unity.exe',
    [string]$Output='',
    [string[]]$Extra=@()
)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path $PSScriptRoot -Parent
if (!(Test-Path -LiteralPath $Unity)) { throw "Unity 6000.4.1f1 is required: $Unity" }
$unityProject=Join-Path $projectRoot 'unity'
$logRoot=Join-Path $projectRoot 'tmp'
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$logPath=Join-Path $logRoot "unity-$Task.log"
$arguments=@('-batchmode','-nographics','-projectPath',"`"$unityProject`"",'-executeMethod','Purumogu.Editor.ProjectTools.Run','-purumoguTask',$Task,'-logFile',"`"$logPath`"")
if ($Output) { $arguments+=@('-purumoguOutput',"`"$Output`"") }
$arguments+=$Extra
$process=Start-Process -FilePath $Unity -ArgumentList $arguments -WindowStyle Hidden -PassThru -Wait
if ($process.ExitCode -ne 0) { Get-Content -LiteralPath $logPath -Tail 60; throw "Unity task failed ($($process.ExitCode)); see $logPath" }
Write-Host "Completed $Task. Log: $logPath"
