@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul
cd /d "%~dp0"
cl /nologo /O2 /EHsc /F 4000000 /DCHECK bench.cpp /Fe:bench_check.exe >nul
bench_check.exe
