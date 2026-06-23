@echo off
setlocal EnableExtensions

set "ROOT=%~dp0.."
set "PACK_ROOT=%ROOT%\data\ready"
set "RUN_SCRIPT=%ROOT%\scripts\run_month_graph_compute.py"
set "RUN_NAME=run_2026_01_01_2026_05_30_cpu24req_cpu_only_dtw_aggressive_flat"
set "PROFILE=cpu_only_dtw"

set "DATE_START=2026-01-01"
set "DATE_END=2026-05-30"
set "MAX_WORKERS=24"
set "SNAPSHOT_BLOCK_SIZE=8"
set "MAX_TASKS_PER_CHILD=4"
set "MAX_IN_FLIGHT_TASKS=32"
set "DTW_PAIR_BATCH_SIZE=1024"
set "DTW_BACKEND=cpu_python"
set "DTW_TORCH_DEVICE=cpu"
set "SYSTEM_MEMORY_RESERVE_GB=4"
set "STOCKNETV2_CPU_ONLY_DTW_HARD_MAX_WORKERS=20"
set "STOCKNETV2_CPU_ONLY_DTW_IN_FLIGHT_BUFFER=6"
set "STOCKNETV2_CPU_ONLY_DTW_MEMORY_SAFETY_MULTIPLIER=3"

set "OUTPUT_ROOT=%ROOT%\research_runs\%RUN_NAME%"
set "RUN_STDOUT=%OUTPUT_ROOT%\launcher.stdout.log"
set "RUN_STDERR=%OUTPUT_ROOT%\launcher.stderr.log"
set "RUN_DIAG=%OUTPUT_ROOT%\launcher.diagnostic.log"
set "RUN_LOG=%OUTPUT_ROOT%\run.log"
set "RUN_PROGRESS=%OUTPUT_ROOT%\progress.jsonl"
set "RUN_PID_FILE=%OUTPUT_ROOT%\run.pid"
set "RUN_BAT_ARCHIVE=%OUTPUT_ROOT%\run_month_cpu_only_dtw_aggressive_flat_resume.bat"

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
copy /Y "%~f0" "%RUN_BAT_ARCHIVE%" >nul

set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
set "NUMEXPR_NUM_THREADS=1"
set "VECLIB_MAXIMUM_THREADS=1"

echo.
echo ============================================================
echo Resuming StockNetV2 cpu_only_dtw aggressive flat run
echo ============================================================
echo Run name       : %RUN_NAME%
echo Python exe     : %PYTHON_EXE%
echo Date start     : %DATE_START%
echo Date end       : %DATE_END%
echo Date workers   : %MAX_WORKERS%
echo Profile        : %PROFILE%
echo In flight      : %MAX_IN_FLIGHT_TASKS%
echo Mem reserve GB : %SYSTEM_MEMORY_RESERVE_GB%
echo Output root    : %OUTPUT_ROOT%
echo ============================================================
echo.

> "%RUN_DIAG%" (
    echo [%date% %time%] python_exe=%PYTHON_EXE%
    echo [%date% %time%] run_script=%RUN_SCRIPT%
    echo [%date% %time%] output_root=%OUTPUT_ROOT%
)

start "" /b "%PYTHON_EXE%" "%RUN_SCRIPT%" ^
  --pack-root "%PACK_ROOT%" ^
  --output-root "%OUTPUT_ROOT%" ^
  --run-name "%RUN_NAME%" ^
  --profile "%PROFILE%" ^
  --date-start "%DATE_START%" ^
  --date-end "%DATE_END%" ^
  --max-workers "%MAX_WORKERS%" ^
  --snapshot-block-size "%SNAPSHOT_BLOCK_SIZE%" ^
  --max-tasks-per-child "%MAX_TASKS_PER_CHILD%" ^
  --max-in-flight-tasks "%MAX_IN_FLIGHT_TASKS%" ^
  --resume-mode log ^
  --dtw-pair-batch-size "%DTW_PAIR_BATCH_SIZE%" ^
  --dtw-backend "%DTW_BACKEND%" ^
  --dtw-torch-device "%DTW_TORCH_DEVICE%" ^
  --system-memory-reserve-gb "%SYSTEM_MEMORY_RESERVE_GB%" ^
  1>> "%RUN_STDOUT%" 2>> "%RUN_STDERR%"

timeout /t 3 /nobreak >nul
for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "$runName='%RUN_NAME%'; Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like ('*' + $runName + '*') -and $_.CommandLine -like '*run_month_graph_compute.py*' } | Select-Object -First 1 -ExpandProperty ProcessId"`) do set "RUN_PID=%%I"

if not defined RUN_PID (
    echo [ERROR] Failed to start python process.
    if exist "%RUN_DIAG%" type "%RUN_DIAG%"
    if exist "%RUN_STDERR%" type "%RUN_STDERR%"
    pause
    exit /b 1
)

> "%RUN_PID_FILE%" echo %RUN_PID%

echo [INFO] Python PID: %RUN_PID%
echo [INFO] Stop command: "%~nx0" stop
echo [INFO] Progress file: %RUN_PROGRESS%

:MONITOR
tasklist /FI "PID eq %RUN_PID%" 2>nul | find "%RUN_PID%" >nul
if errorlevel 1 goto DONE

set "STATUS_LINE=waiting for progress..."
if exist "%RUN_PROGRESS%" (
    for /f "usebackq delims=" %%L in (`powershell -NoProfile -Command "$progress='%RUN_PROGRESS%'; $config='%OUTPUT_ROOT%\run_config.json'; $planned=0; if(Test-Path $config){ try { $planned=[int]((Get-Content $config -Raw | ConvertFrom-Json).planned_snapshots) } catch {} }; $completed=(Get-ChildItem -Path '%OUTPUT_ROOT%' -Recurse -Filter '_PROFILE_SUCCESS' -ErrorAction SilentlyContinue | Measure-Object).Count; $lines=Get-Content $progress; if($lines.Count -eq 0){ 'completed='+$completed+'/'+$planned+' status=starting'; exit }; $obj=$lines[-1] | ConvertFrom-Json; 'completed='+$completed+'/'+$planned+' status='+$obj.status+' date='+$obj.trade_date+' snapshot='+$obj.snapshot_clock"`) do set "STATUS_LINE=%%L"
)

cls
echo ============================================================
echo StockNetV2 cpu_only_dtw aggressive flat monitor
echo ============================================================
echo Run name       : %RUN_NAME%
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
        type "%RUN_STDERR%"
    )
)
echo [INFO] stdout: %RUN_STDOUT%
echo [INFO] stderr: %RUN_STDERR%
pause
endlocal
exit /b 0

:STOP_RUN
echo [INFO] Stopping run processes for %RUN_NAME%...
powershell -NoProfile -Command ^
  "$pidFile='%RUN_PID_FILE%';" ^
  "$runName='%RUN_NAME%';" ^
  "$mainPid=$null;" ^
  "if(Test-Path $pidFile){ try { $mainPid=[int](Get-Content $pidFile | Select-Object -First 1) } catch {} };" ^
  "if($mainPid){ Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $mainPid } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }; Stop-Process -Id $mainPid -Force -ErrorAction SilentlyContinue };" ^
  "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like ('*' + $runName + '*') -and $_.CommandLine -like '*run_month_graph_compute.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue };" ^
  "if(Test-Path $pidFile){ Remove-Item $pidFile -Force -ErrorAction SilentlyContinue }"
echo [INFO] Stop signal completed.
pause
endlocal
