@echo off
setlocal EnableExtensions

set "ROOT=%~dp0.."
set "PACK_ROOT=%ROOT%\data\ready"
set "RUN_NAME=run_2026_01_01_2026_05_30_cpu18"
set "PROFILE=cpu_no_dtw"

set "DATE_START=2026-01-01"
set "DATE_END=2026-05-30"
set "SNAPSHOT_START="
set "SNAPSHOT_END="

set "MAX_WORKERS=18"
set "SNAPSHOT_BLOCK_SIZE=8"
set "MAX_TASKS_PER_CHILD=4"
set "MAX_IN_FLIGHT_TASKS=22"

set "CPU_DTW_BACKEND=cpu_python"
set "DTW_PAIR_BATCH_SIZE=1024"
set "RESUME_MODE=log"

set "OUTPUT_ROOT=%ROOT%\research_runs\%RUN_NAME%"
set "RUN_SCRIPT=%ROOT%\scripts\run_month_graph_compute.py"
set "RUN_BAT_ARCHIVE=%OUTPUT_ROOT%\run_month_cpu_resume.bat"
set "RUN_STDOUT=%OUTPUT_ROOT%\launcher.stdout.log"
set "RUN_STDERR=%OUTPUT_ROOT%\launcher.stderr.log"
set "RUN_LOG=%OUTPUT_ROOT%\run.log"
set "RUN_PID_FILE=%OUTPUT_ROOT%\run.pid"

if /I "%~1"=="stop" goto STOP_RUN

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found in PATH.
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
echo Starting StockNetV2 CPU run (headless)
echo ============================================================
echo Run name       : %RUN_NAME%
echo Date start     : %DATE_START%
echo Date end       : %DATE_END%
echo Date workers   : %MAX_WORKERS%
echo Layer workers  : 1
echo Profile        : %PROFILE%
echo Output root    : %OUTPUT_ROOT%
echo ============================================================
echo.

for /f %%I in ('powershell -NoProfile -Command "$p=Start-Process python -ArgumentList @('%RUN_SCRIPT%','--pack-root','%PACK_ROOT%','--output-root','%OUTPUT_ROOT%','--run-name','%RUN_NAME%','--profile','%PROFILE%','--date-start','%DATE_START%','--date-end','%DATE_END%','--snapshot-start','%SNAPSHOT_START%','--snapshot-end','%SNAPSHOT_END%','--max-workers','%MAX_WORKERS%','--snapshot-block-size','%SNAPSHOT_BLOCK_SIZE%','--max-tasks-per-child','%MAX_TASKS_PER_CHILD%','--max-in-flight-tasks','%MAX_IN_FLIGHT_TASKS%','--resume-mode','%RESUME_MODE%','--cpu-dtw-backend','%CPU_DTW_BACKEND%','--dtw-pair-batch-size','%DTW_PAIR_BATCH_SIZE%') -RedirectStandardOutput '%RUN_STDOUT%' -RedirectStandardError '%RUN_STDERR%' -PassThru -WindowStyle Hidden; $p.Id"') do set "RUN_PID=%%I"
> "%RUN_PID_FILE%" echo %RUN_PID%

echo [INFO] Python PID: %RUN_PID%
echo [INFO] Stop command: "%~nx0" stop
echo [INFO] Monitoring run.log for progress...

:MONITOR
tasklist /FI "PID eq %RUN_PID%" 2>nul | find "%RUN_PID%" >nul
if errorlevel 1 goto DONE

for /f "usebackq delims=" %%L in (`powershell -NoProfile -Command "$log='%RUN_LOG%'; if(!(Test-Path $log)){ 'waiting for first completed snapshot...'; exit }; $lines=Get-Content $log; if($lines.Count -eq 0){ 'waiting for first completed snapshot...'; exit }; $obj=$lines[-1] | ConvertFrom-Json; $ts=[datetimeoffset]::Parse($obj.snapshot_id); $tz=[System.TimeZoneInfo]::FindSystemTimeZoneById('Eastern Standard Time'); $ny=[System.TimeZoneInfo]::ConvertTime($ts,$tz); $idx=[int]((($ny.Hour*60+$ny.Minute)-(9*60+35))/5)+1; 'Current: date='+$obj.trade_date+' snapshot='+$idx+'/78 market_clock='+$ny.ToString('HH:mm')+' edges='+$obj.edge_count+' worker='+$obj.worker_pid"`) do set "STATUS_LINE=%%L"

cls
echo ============================================================
echo StockNetV2 CPU run (headless monitor)
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
pause

endlocal
exit /b 0

:STOP_RUN
echo [INFO] Stopping run processes for %RUN_NAME%...
powershell -NoProfile -Command ^
  "$pidFile='%RUN_PID_FILE%';" ^
  "$runName='%RUN_NAME%';" ^
  "$root='%ROOT%';" ^
  "$mainPid=$null;" ^
  "if(Test-Path $pidFile){ try { $mainPid=[int](Get-Content $pidFile | Select-Object -First 1) } catch {} };" ^
  "if($mainPid){ Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $mainPid } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }; Stop-Process -Id $mainPid -Force -ErrorAction SilentlyContinue };" ^
  "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like ('*' + $runName + '*') -and $_.CommandLine -like '*run_month_graph_compute.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue };" ^
  "if(Test-Path $pidFile){ Remove-Item $pidFile -Force -ErrorAction SilentlyContinue }"
echo [INFO] Stop signal completed.
pause
endlocal
