$taskRoot=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$taskEvents=Get-WinEvent -FilterHashtable @{LogName='Application';Id=1000;StartTime=(Get-Date).AddHours(-8)} -MaxEvents 80 -ErrorAction SilentlyContinue
$taskRows=@(foreach ($taskEvent in $taskEvents) {
    $taskValues=@($taskEvent.Properties | ForEach-Object {$_.Value})
    if ($taskValues.Count -ge 9 -and $taskValues[0] -like '*python*') {
        @{utc=$taskEvent.TimeCreated.ToUniversalTime().ToString('o');app=$taskValues[0];module=$taskValues[3];exception_code=$taskValues[6];process_id_hex=$taskValues[8]}
    }
})
$taskCpu=Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors
$taskReport=@{processor=$taskCpu;python_application_events=$taskRows;diagnosis='Unresolved. Events alone do not establish a hardware fault. Task process affinity is an operational workaround.';event_scope='Python application faults in the execution window; not every historical event is assigned to this experiment.'}
$taskJson=$taskReport | ConvertTo-Json -Depth 6
[IO.File]::WriteAllText((Join-Path $taskRoot 'outputs/native_crash_evidence.json'),$taskJson,(New-Object Text.UTF8Encoding($false)))
Write-Output ('Saved '+$taskRows.Count+' Python application-event records without personal paths.')
