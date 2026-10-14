@echo off
rem Build bench.exe with MSVC 2022 (/O2), as in the original submission.
call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%~dp0"
cl /nologo /O2 /EHsc /F 4000000 bench.cpp /Fe:bench.exe
