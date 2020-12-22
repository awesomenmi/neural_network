@echo off
for /d %%i in ("D:\recognition\*") do set /p "=%%~nxi - "<NUL & 2>NUL dir /b /s /a-d "%%i" | find /C /V ""
pause