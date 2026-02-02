REM filepath: /Users/jileihao/dev/paradigm/cardiac-ultrasound-guidance/run.example.bat
@echo off
REM Windows batch script to run recording utility

REM Set up environment
call scripts\setup.bat

REM Run async recording utility
REM -- interval in milliseconds
REM -- writers: number of parallel writer processes
cusg-record-async ^
  --ip 192.168.1.1 --port 5828 ^
  --outputdir .\output ^
  --interval 50 ^
  --writers 1 ^
  --preserve-date

pause