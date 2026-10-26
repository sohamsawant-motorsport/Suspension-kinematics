@echo off
REM Launch the suspension kinematics workbench.
REM
REM Optional: point at a folder of private reference data (outside this repo).
REM set GT3_PRIVATE_DIR=C:\path\to\private-benchmarks

cd /d "%~dp0"
python software\explorer.py
if errorlevel 1 pause
