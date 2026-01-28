@echo off
setlocal enabledelayedexpansion

set "cast_api_windows_url=https://github.com/clariusdev/cast/releases/download/v12.2.4/cast-12.2.4-windows.x86_64.zip"

echo Setting up Cardiac Ultrasound Guidance environment...
echo Detected OS: Windows

if exist cast_api rmdir /s /q cast_api
mkdir cast_api

echo Downloading Cast API for Windows...
powershell -Command "Invoke-WebRequest -Uri '!cast_api_windows_url!' -OutFile 'cast_api_windows.zip'"
powershell -Command "Expand-Archive -Path 'cast_api_windows.zip' -DestinationPath 'cast_api'"
del cast_api_windows.zip

copy cast_api\cast.dll src\data_access\clarius\cast.dll
copy cast_api\python310\pyclariuscast.pyd src\data_access\clarius\pyclariuscast.pyd

echo Setup completed.
rmdir /s /q cast_api