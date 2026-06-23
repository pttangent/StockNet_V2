@echo off
setlocal EnableExtensions

set "ROOT=%~dp0.."
set "RUN_ROOT=%ROOT%\research_runs"

echo [INFO] taskkill python.exe /T ...
taskkill /F /IM python.exe /T

if exist "%RUN_ROOT%" (
    del /S /Q "%RUN_ROOT%\run.pid" >nul 2>nul
    del /S /Q "%RUN_ROOT%\monitor.pid" >nul 2>nul
    del /S /Q "%RUN_ROOT%\monitor.port" >nul 2>nul
)

echo [INFO] Stop-all signal completed.
pause
endlocal
