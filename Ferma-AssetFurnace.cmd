@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Ferma-AssetFurnace.ps1"
if errorlevel 1 pause
