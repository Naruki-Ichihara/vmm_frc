@echo off
REM Build script for ACSC Windows Installer
REM This script creates a standalone Windows installer for ACSC

setlocal EnableDelayedExpansion

echo ============================================
echo  ACSC Windows Installer Build Script
echo ============================================
echo.

REM Check for Python
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    exit /b 1
)

REM Check for PyInstaller
python -c "import PyInstaller" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

REM Check for required dependencies
echo Checking dependencies...
pip install -e ".[build]" --quiet

echo.
echo Step 0: Creating icon file...
echo ------------------------------------------------
python create_icon.py
if %ERRORLEVEL% neq 0 (
    echo WARNING: Icon creation failed, continuing without custom icon
)

echo.
echo Step 1: Building executable with PyInstaller...
echo ------------------------------------------------
python -m PyInstaller acsc.spec --noconfirm --clean

if %ERRORLEVEL% neq 0 (
    echo ERROR: PyInstaller build failed
    exit /b 1
)

echo.
echo ============================================
echo  Build Complete!
echo ============================================
echo.
echo Portable version: dist\ACSC\
echo.
echo Run dist\ACSC\ACSC.exe to test the application.
echo.

REM Exit here for portable-only build (comment out to enable installer)
REM exit /b 0

echo.
echo Step 2: Creating Windows installer with Inno Setup...
echo ------------------------------------------------------

REM Check for Inno Setup
set ISCC_PATH=
if exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" (
    set "ISCC_PATH=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
) else if exist "C:\Program Files\Inno Setup 6\ISCC.exe" (
    set "ISCC_PATH=C:\Program Files\Inno Setup 6\ISCC.exe"
)

if "!ISCC_PATH!"=="" (
    echo WARNING: Inno Setup 6 not found.
    echo Please install Inno Setup 6 from: https://jrsoftware.org/isinfo.php
    echo.
    echo The PyInstaller build is complete. You can find the portable version at:
    echo   dist\ACSC\
    echo.
    echo After installing Inno Setup, run this script again or manually compile:
    echo   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer.iss
    exit /b 0
)

"!ISCC_PATH!" installer.iss

if %ERRORLEVEL% neq 0 (
    echo ERROR: Inno Setup compilation failed
    exit /b 1
)

echo.
echo ============================================
echo  Build Complete!
echo ============================================
echo.
echo Portable version: dist\ACSC\
echo Installer:        installer_output\ACSC_Setup_0.0.1.exe
echo.

endlocal
