$ErrorActionPreference='Stop'
$start=(Get-Date).AddHours(-6)
$events=@(Get-WinEvent -FilterHashtable @{LogName='Application';Id=1000,1001;StartTime=$start} -ErrorAction SilentlyContinue | Select-Object -First 12)
$system=@(Get-WinEvent -FilterHashtable @{LogName='System';StartTime=$start;Level=1,2,3} -ErrorAction SilentlyContinue | Where-Object {$_.ProviderName -match 'Display|nvld|WHEA'})
@{application=@($events | ForEach-Object {@{time=$_.TimeCreated.ToString('o');id=$_.Id;xml=$_.ToXml()}});system=@($system | ForEach-Object {@{time=$_.TimeCreated.ToString('o');provider=$_.ProviderName;xml=$_.ToXml()}})} | ConvertTo-Json -Depth 5 | Set-Content 'D:\solver_consistent_stage0\artifacts\windows_crash_events.json' -Encoding UTF8
$system | ForEach-Object {$_.ToXml()}
nvidia-smi
