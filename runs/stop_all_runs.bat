@echo off
setlocal EnableExtensions

set "ROOT=%~dp0.."
echo [INFO] Stopping all StockNetV2 month run processes...

powershell -NoProfile -Command ^
  "$root='%ROOT%';" ^
  "$runRoot=Join-Path $root 'research_runs';" ^
  "$pids=@();" ^
  "if(Test-Path $runRoot){ Get-ChildItem -Path $runRoot -Recurse -Filter 'run.pid' -ErrorAction SilentlyContinue | ForEach-Object { try { $pids += [int](Get-Content $_.FullName | Select-Object -First 1) } catch {} } };" ^
  "$pids = $pids | Select-Object -Unique;" ^
  "foreach($mainPid in $pids){ Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $mainPid } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }; Stop-Process -Id $mainPid -Force -ErrorAction SilentlyContinue };" ^
  "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*run_month_graph_compute.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue };" ^
  "if(Test-Path $runRoot){ Get-ChildItem -Path $runRoot -Recurse -Filter 'run.pid' -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue }"

echo [INFO] Stop-all signal completed.
pause
endlocal
