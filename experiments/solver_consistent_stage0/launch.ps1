$stageRoot = 'D:\solver_consistent_stage0'
$stagePython = Join-Path $stageRoot '.venv\Scripts\python.exe'
$stageArtifacts = Join-Path $stageRoot 'artifacts'
New-Item -ItemType Directory -Path $stageArtifacts -Force | Out-Null
$process = Start-Process -FilePath $stagePython -ArgumentList '-u','run_pipeline.py' -WorkingDirectory $stageRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $stageArtifacts 'pipeline.stdout.log') -RedirectStandardError (Join-Path $stageArtifacts 'pipeline.stderr.log') -PassThru
$process.Id | Set-Content (Join-Path $stageArtifacts 'pipeline.pid')
Write-Output "Started Stage-0 pipeline PID $($process.Id)"
