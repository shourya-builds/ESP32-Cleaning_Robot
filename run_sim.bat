@echo off
title ESP32 HIL Robot Simulator
cd /d "%~dp0"
python sim/robot.py --mock
pause
