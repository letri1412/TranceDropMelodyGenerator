@echo off
rem Run the trance_lora CLI inside the project venv.
rem Usage:  trance.cmd generate --version v2 --out out\demo.mid
setlocal
set HERE=%~dp0
set PYTHONPATH=%HERE%_bootstrap;%HERE%src
set PYTHONIOENCODING=utf-8
if not exist "%HERE%tmp" mkdir "%HERE%tmp"
set TMP=%HERE%tmp
set TEMP=%HERE%tmp
pushd "%HERE%"
"%HERE%.venv\Scripts\python.exe" -m trance_lora %*
set CODE=%ERRORLEVEL%
popd
exit /b %CODE%
