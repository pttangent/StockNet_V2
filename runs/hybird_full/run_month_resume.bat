@echo off
setlocal
set "MODE_NAME=hybird_full"
set "PROFILE=hybird_full"
set "RESUME_MODE=log"
set "RUN_NAME=run_2026_01_01_2026_05_30_cpu18_hybird_full"
set "DTW_BACKEND=torch_cuda"
set "DTW_TORCH_DEVICE=cuda"
set "WRAPPER_NAME=%~nx0"
set "ORIGINAL_BAT_PATH=%~f0"
call "%~dp0..\_common\run_month_mode.bat" %*
endlocal
