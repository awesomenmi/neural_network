@echo off
for /D %%i in ("C:\Users\eenmi\diploma\data1\train\*") do set /p "=%%~nxi - "<NUL & 2>NUL dir /b /s /a-d "%%i" | find /C /V ""
pause