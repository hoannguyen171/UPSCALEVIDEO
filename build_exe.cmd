@echo off
chcp 65001 >nul
title Build ValoUpscale.exe
cd /d "%~dp0"

echo [1/4] cai pyinstaller...
python -m pip install -q --upgrade pyinstaller || goto :err

echo [2/4] xoa ban cu...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo [3/4] dong goi 1 file...
python -m PyInstaller --noconfirm --clean --onefile --noconsole ^
  --name ValoUpscale ^
  --add-data "kaggle_jobs\recavsr_worker.py;kaggle_jobs" ^
  --hidden-import winapp.app --hidden-import winapp.cli --hidden-import winapp.job ^
  --hidden-import winapp.kaggle_client --hidden-import winapp.share --hidden-import winapp.tunnel ^
  --exclude-module valo_upscale --exclude-module numpy --exclude-module PIL ^
  --exclude-module torch --exclude-module cv2 --exclude-module matplotlib ^
  run_win.py || goto :err

echo [4/4] xong:
dir /b dist\ValoUpscale.exe
echo.
echo File chay: %cd%\dist\ValoUpscale.exe
pause
exit /b 0

:err
echo.
echo BUILD LOI - xem thong bao phia tren.
pause
exit /b 1
