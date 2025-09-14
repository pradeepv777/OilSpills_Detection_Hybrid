@echo off
echo 🌊 SAR Oil Spill Detection - Activating Virtual Environment
echo ============================================================
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.
echo ✅ Virtual environment activated!
echo.
echo 🎯 You can now run:
echo   - python quick_train.py    (Quick training)
echo   - python train.py          (Advanced training)
echo   - python predict.py        (Make predictions)
echo   - python demo.py           (Test the system)
echo.
echo To deactivate, simply type: deactivate
echo.
cmd /k
