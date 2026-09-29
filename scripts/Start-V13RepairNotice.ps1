param(
    [Parameter(Mandatory=$true)][int]$Candidate,
    [string]$Python = 'python'
)
$ErrorActionPreference = 'Stop'
$taskRepo = Split-Path -Parent $PSScriptRoot
$taskName = 'v13-repair-candidate-{0:D2}' -f $Candidate
$taskConfig = Join-Path $env:USERPROFILE ('.oaciss-secrets/' + $taskName + '-local.json')
$taskSettings = Get-Content -LiteralPath $taskConfig -Raw | ConvertFrom-Json
$taskState = [string]$taskSettings.state_dir
$taskReceipt = Join-Path $taskState 'receiver-start.json'
if (Test-Path -LiteralPath $taskReceipt) { throw 'Receiver start receipt already exists; inspect it before recovery.' }
$taskScript = Join-Path $PSScriptRoot 'telegram_codex_v13_repair.py'
$taskProcess = Start-Process -FilePath $Python -ArgumentList @('-X', 'utf8', ('"' + $taskScript + '"'), ('"' + $taskConfig + '"')) -WorkingDirectory $taskRepo -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskState 'receiver.stdout.log') -RedirectStandardError (Join-Path $taskState 'receiver.stderr.log') -PassThru
$taskValue = @{ pid=$taskProcess.Id; started_at=$taskProcess.StartTime.ToUniversalTime().ToString('o'); script=$taskScript; armed=$false } | ConvertTo-Json
[System.IO.File]::WriteAllText($taskReceipt, $taskValue, [System.Text.UTF8Encoding]::new($false))
$taskValue
