@echo off
cd /d "%~dp0"
echo Futbol Tahmin Sistemi baslatiliyor...
echo http://localhost:5000
echo.
if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    set "PY=python"
)
echo Eski port 5000 temizleniyor...
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 5000 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique | Where-Object {$_ -ne $PID} | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }"
echo.
%PY% run_web.py
pause
