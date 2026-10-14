$ErrorActionPreference='Stop'
$taskRoot=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$taskActive=Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {
    $_.CommandLine -like "*$taskRoot*" -and ($_.CommandLine -like '*run_remaining.py*' -or $_.CommandLine -like '*baseline_length_isolated.py*')
}
if ($taskActive) {
    $taskActive | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress
    exit 0
}
# WMI owns this hidden process, so an SSH/client turn interruption does not
# terminate the experiment's process tree. No recurring task/service is created.
$taskStartup=New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}
$taskCommand='cmd.exe /d /s /c ""'+$taskRoot+'\.venv\Scripts\python.exe" -u "'+$taskRoot+'\scripts\run_remaining.py" > "'+$taskRoot+'\outputs\logs\durable_runner.log" 2>&1"'
$taskResult=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$taskCommand;CurrentDirectory=$taskRoot;ProcessStartupInformation=$taskStartup}
if ($taskResult.ReturnValue -ne 0) { throw "WMI process creation failed: $($taskResult.ReturnValue)" }
$taskRecord=@{launcher='WMI Win32_Process.Create';process_id=$taskResult.ProcessId;created_utc=[DateTime]::UtcNow.ToString('o');script='scripts/run_remaining.py';scope='single existing experiment, hidden process, resumable'}
$taskRecord | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskRoot 'outputs/background_job.json') -Encoding UTF8
$taskRecord | ConvertTo-Json -Compress
