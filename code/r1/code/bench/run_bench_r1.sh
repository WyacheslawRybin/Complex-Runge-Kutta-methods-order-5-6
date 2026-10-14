#!/usr/bin/env bash
# Five runs of the CPU benchmark with all fourteen methods on the eleven problems (Bruss2D included) in each
# run; the CPU tables of R1 are then written by code/tables_cpu_r1.py.
# Run on an otherwise idle machine (mains power, no other computations): the script refuses to start
# if the average CPU load over 5 s exceeds 10 %, because a loaded machine distorts the time ratios
# between real and complex arithmetic.
#   cd "Submission MDPI R1/code/bench" && bash run_bench_r1.sh
set -e
cd "$(dirname "$0")"
load=$(powershell -NoProfile -Command "(Get-CimInstance Win32_Processor | Measure-Object LoadPercentage -Average).Average; Start-Sleep 5; (Get-CimInstance Win32_Processor | Measure-Object LoadPercentage -Average).Average" | tr -d '\r' | awk '{s+=$1} END {printf "%d", s/NR}')
echo "CPU load: ${load} %"
if [ "${load}" -gt 10 ] && [ "${FORCE:-0}" != "1" ]; then
  echo "machine is busy; stop other computations or set FORCE=1"; exit 1
fi
mkdir -p ../../data/bench_r1
rm -f ../../data/bench_r1/run*.csv
for k in 1 2 3 4 5; do
  start=$(date +%s)
  ./bench.exe > "../../data/bench_r1/run$k.csv"
  echo "run $k: $(( $(date +%s) - start )) s"
done
