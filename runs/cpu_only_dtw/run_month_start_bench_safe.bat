@echo off
if /I not "%~1"=="__inner__" (
    start "StockNetV2 cpu_only_dtw bench safe" cmd /k call "%~f0" __inner__
    exit /b 0
)
shift /1
setlocal
rem Benchmark-aligned cpu_only_dtw launcher.
rem Requests the same 18/22 concurrency as the 5-minute benchmark,
rem while run_month_graph_compute.py now applies a memory guard so the
rem effective worker count is reduced automatically when the host cannot
rem safely sustain that fan-out.
set "MODE_NAME=cpu_only_dtw_bench_safe"
set "PROFILE=cpu_only_dtw"
set "RESUME_MODE=off"
set "RUN_NAME=run_2026_01_01_2026_05_30_cpu18req_cpu_only_dtw_safe"
set "MAX_WORKERS=16"
set "MAX_IN_FLIGHT_TASKS=22"
set "MAX_TASKS_PER_CHILD=4"
set "SNAPSHOT_BLOCK_SIZE=8"
set "DTW_PAIR_BATCH_SIZE=1024"
set "DTW_BACKEND=cpu_python"
set "DTW_TORCH_DEVICE=cpu"
set "WRAPPER_NAME=%~nx0"
set "ORIGINAL_BAT_PATH=%~f0"
call "%~dp0..\_common\run_month_mode.bat" %*
endlocal
