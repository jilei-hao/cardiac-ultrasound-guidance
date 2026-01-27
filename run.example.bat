REM filepath: /Users/jileihao/dev/paradigm/cardiac-ultrasound-guidance/run.example.bat
@echo off
REM Windows batch script to run recording utility

REM Set up environment
call scripts\setup.bat

REM Run recording utility
REM -- duration in seconds
REM -- interval in milliseconds
cusg-record ^
  --ip 66.66.66.160 --port 5828 ^
  --outputdir ./output ^
  --duration 10 ^
  --interval 50 ^
  --preserve-date

pause