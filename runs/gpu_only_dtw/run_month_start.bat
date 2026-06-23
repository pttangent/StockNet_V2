@echo off
:: GPU DTW Mode Startup Script
:: Recommended: MAX_WORKERS=2, MAX_IN_FLIGHT_TASKS=4 to prevent GPU context contention.
:: For more details, see: docs/plans/gpu-dtw-runtime-notes.md
setlocal
set "MODE_NAME=gpu_only_dtw"
set "PROFILE=gpu_only_dtw"
set "RESUME_MODE=off"
set "RUN_NAME=run_2026_01_01_2026_05_30_cpu2_gpu_only_dtw"
set "DTW_BACKEND=torch_cuda"
set "DTW_TORCH_DEVICE=cuda"
set "WRAPPER_NAME=%~nx0"
set "ORIGINAL_BAT_PATH=%~f0"
call "%~dp0..\_common\run_month_mode.bat" %*
endlocal
