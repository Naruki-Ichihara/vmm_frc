@echo off
REM Build script for VMM-FRC Windows Installer
REM This script creates a standalone Windows installer for VMM-FRC

setlocal EnableDelayedExpansion

echo ============================================
echo  VMM-FRC Windows Installer Build Script
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

REM Get version from vmm/__init__.py and update installer.iss
echo.
echo Updating version in installer.iss...
python -c "from vmm import __version__; print(__version__)" > temp_version.txt
set /p VERSION=<temp_version.txt
del temp_version.txt
echo Version: %VERSION%

REM Update installer.iss with current version using a separate Python script
python -c "import re; from vmm import __version__; f=open('installer.iss','r',encoding='utf-8'); c=f.read(); f.close(); c=re.sub(r'#define MyAppVersion \"[^\"]+\"', f'#define MyAppVersion \"{__version__}\"', c); f=open('installer.iss','w',encoding='utf-8'); f.write(c); f.close(); print(f'Updated installer.iss to version {__version__}')"

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
python -m PyInstaller vmm.spec --noconfirm --clean

if %ERRORLEVEL% neq 0 (
    echo ERROR: PyInstaller build failed
    exit /b 1
)

echo.
echo ============================================
echo  Build Complete!
echo ============================================
echo.
echo Portable version: dist\VMM-FRC\
echo.
echo Run dist\VMM-FRC\VMM-FRC.exe to test the application.
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
    echo   dist\VMM-FRC\
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
echo Portable version: dist\VMM-FRC\
echo Installer:        installer_output\VMM-FRC_Setup_%VERSION%.exe
echo.

endlocal
