@echo off
setlocal

set "ROOT=%~dp0.."
set "PACK_ROOT=%ROOT%\data\ready"
set "RUN_NAME=year_2025_cpu"
set "PROFILE=cpu_no_dtw"

set "DATE_START=2025-01-01"
set "DATE_END=2025-12-31"
set "SNAPSHOT_START="
set "SNAPSHOT_END="

set "MAX_WORKERS=30"
set "SNAPSHOT_BLOCK_SIZE=8"
set "MAX_TASKS_PER_CHILD=4"
set "MAX_IN_FLIGHT_TASKS=34"

set "CPU_DTW_BACKEND=cpu_python"
set "DTW_PAIR_BATCH_SIZE=1024"
set "RESUME_MODE=log"

set "OUTPUT_ROOT=%ROOT%\research_runs\%RUN_NAME%"
set "RUN_SCRIPT=%ROOT%\scripts\run_month_graph_compute.py"

set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
set "NUMEXPR_NUM_THREADS=1"
set "VECLIB_MAXIMUM_THREADS=1"

python "%RUN_SCRIPT%" ^
  --pack-root "%PACK_ROOT%" ^
  --output-root "%OUTPUT_ROOT%" ^
  --run-name "%RUN_NAME%" ^
  --profile "%PROFILE%" ^
  --date-start "%DATE_START%" ^
  --date-end "%DATE_END%" ^
  --snapshot-start "%SNAPSHOT_START%" ^
  --snapshot-end "%SNAPSHOT_END%" ^
  --max-workers %MAX_WORKERS% ^
  --snapshot-block-size %SNAPSHOT_BLOCK_SIZE% ^
  --max-tasks-per-child %MAX_TASKS_PER_CHILD% ^
  --max-in-flight-tasks %MAX_IN_FLIGHT_TASKS% ^
  --resume-mode %RESUME_MODE% ^
  --cpu-dtw-backend %CPU_DTW_BACKEND% ^
  --dtw-pair-batch-size %DTW_PAIR_BATCH_SIZE%

endlocal
