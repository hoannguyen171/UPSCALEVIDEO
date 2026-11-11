@echo off
chcp 65001 >nul
title Valo Upscale - GPU Kaggle (ReCaVSR int8)
cd /d "%~dp0"

where pythonw >nul 2>nul && (start "" pythonw "run_win.py" & exit /b)
where python  >nul 2>nul && (start "" python  "run_win.py" & exit /b)

echo Khong tim thay Python. Cai Python 3.9+ tu python.org roi chay lai.
pause
