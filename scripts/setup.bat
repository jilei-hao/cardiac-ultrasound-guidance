@echo off
REM Windows batch script
set PROJECT_ROOT=%~dp0..
set CAST_API_DIR=%PROJECT_ROOT%\src\data_access\clarius
set PATH=%CAST_API_DIR%;%PATH%
echo Clarius library path configured: %CAST_API_DIR%