@echo off
if /I not "%~1"=="__inner__" (
    start "StockNetV2 cpu_only_dtw aggressive" cmd /k call "%~f0" __inner__
    exit /b 0
)
shift /1
setlocal
rem More aggressive cpu_only_dtw launcher for high-memory hosts.
rem Requests full CPU fan-out and lowers the memory guard reserve.
set "MODE_NAME=cpu_only_dtw_aggressive"
set "PROFILE=cpu_only_dtw"
set "RESUME_MODE=off"
set "RUN_NAME=run_2026_01_01_2026_05_30_cpu24req_cpu_only_dtw_aggressive"
set "MAX_WORKERS=18"
set "MAX_IN_FLIGHT_TASKS=24"
set "MAX_TASKS_PER_CHILD=4"
set "SNAPSHOT_BLOCK_SIZE=8"
set "DTW_PAIR_BATCH_SIZE=1024"
set "SYSTEM_MEMORY_RESERVE_GB=4"
set "STOCKNETV2_CPU_ONLY_DTW_HARD_MAX_WORKERS=16"
set "STOCKNETV2_CPU_ONLY_DTW_IN_FLIGHT_BUFFER=6"
set "STOCKNETV2_CPU_ONLY_DTW_MEMORY_SAFETY_MULTIPLIER=3"
set "DTW_BACKEND=cpu_python"
set "DTW_TORCH_DEVICE=cpu"
set "WRAPPER_NAME=%~nx0"
set "ORIGINAL_BAT_PATH=%~f0"
call "%~dp0..\_common\run_month_mode.bat" %*
endlocal
