@echo off
setlocal EnableExtensions

if not defined MODE_NAME (
    echo [ERROR] MODE_NAME is not set.
    pause
    exit /b 1
)
if not defined PROFILE (
    echo [ERROR] PROFILE is not set.
    pause
    exit /b 1
)
if not defined RESUME_MODE (
    echo [ERROR] RESUME_MODE is not set.
    pause
    exit /b 1
)
if not defined RUN_NAME (
    echo [ERROR] RUN_NAME is not set.
    pause
    exit /b 1
)

set "ROOT=%~dp0..\.."
set "PACK_ROOT=%ROOT%\data\ready"

if not defined DATE_START set "DATE_START=2026-01-01"
if not defined DATE_END set "DATE_END=2026-05-30"
if not defined SNAPSHOT_START set "SNAPSHOT_START="
if not defined SNAPSHOT_END set "SNAPSHOT_END="
if not defined MAX_WORKERS (
    if "%DTW_BACKEND%"=="torch_cuda" (
        set "MAX_WORKERS=2"
    ) else (
        set "MAX_WORKERS=20"
    )
)
if not defined SNAPSHOT_BLOCK_SIZE set "SNAPSHOT_BLOCK_SIZE=8"
if not defined MAX_TASKS_PER_CHILD set "MAX_TASKS_PER_CHILD=4"
if not defined MAX_IN_FLIGHT_TASKS (
    if "%DTW_BACKEND%"=="torch_cuda" (
        set "MAX_IN_FLIGHT_TASKS=4"
    ) else (
        set "MAX_IN_FLIGHT_TASKS=24"
    )
)
if not defined DTW_PAIR_BATCH_SIZE set "DTW_PAIR_BATCH_SIZE=1024"
if not defined TORCH_ACTIVATION_PAIR_THRESHOLD set "TORCH_ACTIVATION_PAIR_THRESHOLD=1024"
if not defined TORCH_GPU_CHUNK_SIZE set "TORCH_GPU_CHUNK_SIZE=8192"
if not defined SYSTEM_MEMORY_RESERVE_GB set "SYSTEM_MEMORY_RESERVE_GB=10"
if not defined WRAPPER_NAME set "WRAPPER_NAME=%~nx0"
if not defined ORIGINAL_BAT_PATH set "ORIGINAL_BAT_PATH=%~f0"
set "SNAPSHOT_START_ARG="
set "SNAPSHOT_END_ARG="
set "DTW_BACKEND_ARG="
set "DTW_TORCH_DEVICE_ARG="
set "GRAPH_BACKEND_ARG="
set "GRAPH_TORCH_DEVICE_ARG="
if defined SNAPSHOT_START set "SNAPSHOT_START_ARG=--snapshot-start %SNAPSHOT_START%"
if defined SNAPSHOT_END set "SNAPSHOT_END_ARG=--snapshot-end %SNAPSHOT_END%"
if defined DTW_BACKEND set "DTW_BACKEND_ARG=--dtw-backend %DTW_BACKEND%"
if defined DTW_TORCH_DEVICE set "DTW_TORCH_DEVICE_ARG=--dtw-torch-device %DTW_TORCH_DEVICE%"
if defined GRAPH_BACKEND set "GRAPH_BACKEND_ARG=--graph-backend %GRAPH_BACKEND%"
if defined GRAPH_TORCH_DEVICE set "GRAPH_TORCH_DEVICE_ARG=--graph-torch-device %GRAPH_TORCH_DEVICE%"

set "OUTPUT_ROOT=%ROOT%\research_runs\%RUN_NAME%"
set "RUN_SCRIPT=%ROOT%\scripts\run_month_graph_compute.py"
set "RUN_BAT_ARCHIVE=%OUTPUT_ROOT%\%WRAPPER_NAME%"
set "RUN_STDOUT=%OUTPUT_ROOT%\launcher.stdout.log"
set "RUN_STDERR=%OUTPUT_ROOT%\launcher.stderr.log"
set "RUN_LAUNCH_DIAG=%OUTPUT_ROOT%\launcher.diagnostic.log"
set "RUN_LOG=%OUTPUT_ROOT%\run.log"
set "RUN_PROGRESS=%OUTPUT_ROOT%\progress.jsonl"
set "RUN_PID_FILE=%OUTPUT_ROOT%\run.pid"
set "MONITOR_PORT=3030"
set "MONITOR_STDOUT=%OUTPUT_ROOT%\monitor.stdout.log"
set "MONITOR_STDERR=%OUTPUT_ROOT%\monitor.stderr.log"
set "MONITOR_PID_FILE=%OUTPUT_ROOT%\monitor.pid"
set "MONITOR_PORT_FILE=%OUTPUT_ROOT%\monitor.port"

if /I "%~1"=="stop" goto STOP_RUN

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found in PATH.
    pause
    exit /b 1
)
for /f "delims=" %%I in ('where python') do (
    if not defined PYTHON_EXE set "PYTHON_EXE=%%I"
)
if not defined PYTHON_EXE (
    echo [ERROR] Unable to resolve python executable from PATH.
    pause
    exit /b 1
)

if not exist "%RUN_SCRIPT%" (
    echo [ERROR] Missing run script:
    echo         %RUN_SCRIPT%
    pause
    exit /b 1
)

if not exist "%PACK_ROOT%" (
    echo [ERROR] Missing pack root:
    echo         %PACK_ROOT%
    pause
    exit /b 1
)

if not exist "%OUTPUT_ROOT%" mkdir "%OUTPUT_ROOT%"
copy /Y "%ORIGINAL_BAT_PATH%" "%RUN_BAT_ARCHIVE%" >nul

set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
set "NUMEXPR_NUM_THREADS=1"
set "VECLIB_MAXIMUM_THREADS=1"

echo.
echo ============================================================
if /I "%RESUME_MODE%"=="log" (
    echo Starting StockNetV2 %MODE_NAME% resume run
) else (
    echo Starting StockNetV2 %MODE_NAME% fresh run
)
echo ============================================================
echo Run name       : %RUN_NAME%
echo Mode           : %MODE_NAME%
echo Profile        : %PROFILE%
echo Date start     : %DATE_START%
echo Date end       : %DATE_END%
echo Date workers   : %MAX_WORKERS%
echo Layer workers  : 1
echo Resume mode    : %RESUME_MODE%
echo Python exe     : %PYTHON_EXE%
if defined DTW_BACKEND echo DTW backend    : %DTW_BACKEND%
if defined DTW_TORCH_DEVICE echo DTW device     : %DTW_TORCH_DEVICE%
if defined TORCH_ACTIVATION_PAIR_THRESHOLD echo DTW threshold  : %TORCH_ACTIVATION_PAIR_THRESHOLD%
if defined TORCH_GPU_CHUNK_SIZE echo DTW chunk size : %TORCH_GPU_CHUNK_SIZE%
if defined SYSTEM_MEMORY_RESERVE_GB echo Mem reserve GB : %SYSTEM_MEMORY_RESERVE_GB%
echo Output root    : %OUTPUT_ROOT%
echo ============================================================
echo.

for /f "usebackq" %%I in (`powershell -NoProfile -Command "$procs = Get-CimInstance Win32_Process; foreach($p in $procs){ if($p.Name -eq 'node.exe' -and $p.CommandLine -like '*server.js*'){ try { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue } catch {} } }; 'ok'"`) do set "NODE_CLEANUP=%%I"
for /f %%P in ('powershell -NoProfile -Command "$ports=3030..3055; foreach($port in $ports){ try { $listener=[System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback,$port); $listener.Start(); $listener.Stop(); $port; break } catch { if($listener){ try { $listener.Stop() } catch {} } } }"') do set "MONITOR_PORT=%%P"
> "%MONITOR_PORT_FILE%" echo %MONITOR_PORT%
for /f %%I in ('powershell -NoProfile -Command "$env:PORT='%MONITOR_PORT%'; $env:STOCKNETV2_MONTH_RUN_ROOT='%OUTPUT_ROOT%'; $p=Start-Process node -ArgumentList @('%ROOT%\server.js') -RedirectStandardOutput '%MONITOR_STDOUT%' -RedirectStandardError '%MONITOR_STDERR%' -PassThru -WindowStyle Hidden; $p.Id"') do set "MONITOR_PID=%%I"
if defined MONITOR_PID (
    > "%MONITOR_PID_FILE%" echo %MONITOR_PID%
    rem start "" "http://127.0.0.1:%MONITOR_PORT%/month-progress"
)

set "STATUS_LINE=Launching python worker..."
> "%RUN_LAUNCH_DIAG%" (
    echo [%date% %time%] python_exe=%PYTHON_EXE%
    echo [%date% %time%] run_script=%RUN_SCRIPT%
    echo [%date% %time%] output_root=%OUTPUT_ROOT%
)
>> "%RUN_LAUNCH_DIAG%" echo [%date% %time%] launching_with_cmd_start=1
start "" /b "%PYTHON_EXE%" "%RUN_SCRIPT%" --pack-root "%PACK_ROOT%" --output-root "%OUTPUT_ROOT%" --run-name "%RUN_NAME%" --profile "%PROFILE%" --date-start "%DATE_START%" --date-end "%DATE_END%" --max-workers "%MAX_WORKERS%" --snapshot-block-size "%SNAPSHOT_BLOCK_SIZE%" --max-tasks-per-child "%MAX_TASKS_PER_CHILD%" --max-in-flight-tasks "%MAX_IN_FLIGHT_TASKS%" --resume-mode "%RESUME_MODE%" --dtw-pair-batch-size "%DTW_PAIR_BATCH_SIZE%" --torch-activation-pair-threshold "%TORCH_ACTIVATION_PAIR_THRESHOLD%" --torch-gpu-chunk-size "%TORCH_GPU_CHUNK_SIZE%" --system-memory-reserve-gb "%SYSTEM_MEMORY_RESERVE_GB%" %SNAPSHOT_START_ARG% %SNAPSHOT_END_ARG% %DTW_BACKEND_ARG% %DTW_TORCH_DEVICE_ARG% %GRAPH_BACKEND_ARG% %GRAPH_TORCH_DEVICE_ARG% 1>> "%RUN_STDOUT%" 2>> "%RUN_STDERR%"
timeout /t 2 /nobreak >nul
for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "$runName='%RUN_NAME%'; Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like ('*' + $runName + '*') -and $_.CommandLine -like '*run_month_graph_compute.py*' } | Select-Object -First 1 -ExpandProperty ProcessId"`) do set "RUN_PID=%%I"
if not defined RUN_PID (
    echo [ERROR] Failed to start python process.
    if exist "%RUN_LAUNCH_DIAG%" type "%RUN_LAUNCH_DIAG%"
    if exist "%RUN_STDERR%" powershell -NoProfile -Command "Get-Content '%RUN_STDERR%' -Tail 20"
    pause
    exit /b 1
)
> "%RUN_PID_FILE%" echo %RUN_PID%

echo [INFO] Python PID: %RUN_PID%
if defined MONITOR_PID echo [INFO] Frontend URL : http://127.0.0.1:%MONITOR_PORT%/month-progress
echo [INFO] Stop command: "%WRAPPER_NAME%" stop
echo [INFO] Frontend progress monitor started.

:MONITOR
tasklist /FI "PID eq %RUN_PID%" 2>nul | find "%RUN_PID%" >nul
if errorlevel 1 goto DONE

for /f "usebackq delims=" %%L in (`powershell -NoProfile -Command "$progress='%RUN_PROGRESS%'; $config='%OUTPUT_ROOT%\run_config.json'; $failures='%OUTPUT_ROOT%\failures.jsonl'; $planned=0; if(Test-Path $config){ try { $planned=[int]((Get-Content $config -Raw | ConvertFrom-Json).planned_snapshots) } catch {} }; $completed=(Get-ChildItem -Path '%OUTPUT_ROOT%' -Recurse -Filter '_PROFILE_SUCCESS' -ErrorAction SilentlyContinue | Measure-Object).Count; $failureCount=0; if(Test-Path $failures){ $failureCount=(Get-Content $failures | Measure-Object).Count }; if(!(Test-Path $progress)){ 'Progress: completed='+$completed+'/'+$planned+' failures='+$failureCount+' status=starting'; exit }; $lines=Get-Content $progress; if($lines.Count -eq 0){ 'Progress: completed='+$completed+'/'+$planned+' failures='+$failureCount+' status=starting'; exit }; $obj=$lines[-1] | ConvertFrom-Json; $snapshotText=$obj.snapshot_clock; try { $ts=[datetimeoffset]::Parse($obj.snapshot_id); $tz=[System.TimeZoneInfo]::FindSystemTimeZoneById('Eastern Standard Time'); $ny=[System.TimeZoneInfo]::ConvertTime($ts,$tz); $snapshotText=([int]((($ny.Hour*60+$ny.Minute)-(9*60+35))/5)+1).ToString()+'/78 market_clock='+$ny.ToString('HH:mm') } catch {}; $worker='n/a'; if($obj.worker_pid){ $worker=[string]$obj.worker_pid }; $edges='n/a'; if($obj.PSObject.Properties.Name -contains 'edge_count'){ $edges=[string]$obj.edge_count }; 'Progress: completed='+$completed+'/'+$planned+' failures='+$failureCount+' current_status='+$obj.status+' date='+$obj.trade_date+' snapshot='+$snapshotText+' edges='+$edges+' worker='+$worker"`) do set "STATUS_LINE=%%L"

cls
echo ============================================================
echo StockNetV2 %MODE_NAME% monitor
echo ============================================================
echo Run name       : %RUN_NAME%
echo Date range     : %DATE_START% ^> %DATE_END%
echo Date workers   : %MAX_WORKERS%
echo Profile        : %PROFILE%
echo PID            : %RUN_PID%
echo.
echo %STATUS_LINE%
echo.
echo Logs:
echo   %RUN_PROGRESS%
echo   %RUN_LOG%
echo   %RUN_STDERR%
echo ============================================================
timeout /t 5 /nobreak >nul
goto MONITOR

:DONE
echo.
echo [INFO] Run process exited.
if exist "%RUN_STDERR%" (
  for %%S in ("%RUN_STDERR%") do if %%~zS GTR 0 (
    echo [WARN] stderr is not empty:
    powershell -NoProfile -Command "Get-Content '%RUN_STDERR%' -Tail 20"
  )
)
echo [INFO] stdout: %RUN_STDOUT%
echo [INFO] stderr: %RUN_STDERR%
if defined MONITOR_PID echo [INFO] frontend: http://127.0.0.1:%MONITOR_PORT%/month-progress
pause

endlocal
exit /b 0

:STOP_RUN
echo [INFO] Stopping run processes for %RUN_NAME%...
powershell -NoProfile -Command ^
  "$pidFile='%RUN_PID_FILE%';" ^
  "$monitorPidFile='%MONITOR_PID_FILE%';" ^
  "$monitorPortFile='%MONITOR_PORT_FILE%';" ^
  "$runName='%RUN_NAME%';" ^
  "$mainPid=$null;" ^
  "$monitorPid=$null;" ^
  "if(Test-Path $pidFile){ try { $mainPid=[int](Get-Content $pidFile | Select-Object -First 1) } catch {} };" ^
  "if(Test-Path $monitorPidFile){ try { $monitorPid=[int](Get-Content $monitorPidFile | Select-Object -First 1) } catch {} };" ^
  "if($mainPid){ Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $mainPid } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }; Stop-Process -Id $mainPid -Force -ErrorAction SilentlyContinue };" ^
  "if($monitorPid){ Stop-Process -Id $monitorPid -Force -ErrorAction SilentlyContinue };" ^
  "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like ('*' + $runName + '*') -and $_.CommandLine -like '*run_month_graph_compute.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue };" ^
  "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'node.exe' -and $_.CommandLine -like '*server.js*' -and $_.CommandLine -like ('*' + $runName + '*') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue };" ^
  "if(Test-Path $pidFile){ Remove-Item $pidFile -Force -ErrorAction SilentlyContinue }; if(Test-Path $monitorPidFile){ Remove-Item $monitorPidFile -Force -ErrorAction SilentlyContinue }; if(Test-Path $monitorPortFile){ Remove-Item $monitorPortFile -Force -ErrorAction SilentlyContinue }"
echo [INFO] Stop signal completed.
pause
endlocal
