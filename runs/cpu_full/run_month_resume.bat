@echo off
setlocal
set "MODE_NAME=cpu_full"
set "PROFILE=cpu_full"
set "RESUME_MODE=log"
set "RUN_NAME=run_2026_01_01_2026_05_30_cpu18_cpu_full"
set "WRAPPER_NAME=%~nx0"
set "ORIGINAL_BAT_PATH=%~f0"
call "%~dp0..\_common\run_month_mode.bat" %*
endlocal
