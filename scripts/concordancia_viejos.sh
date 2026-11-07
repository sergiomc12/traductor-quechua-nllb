#!/bin/bash
cd /d/Backend/traductorquechua || exit 1
args=""
for n in exp_aleatorio15_s0 exp_filtrado15_s0 exp_bt30_s0 exp_bt40_s0 auto_it1; do args="$args --modelo $n=models/$n/mejor"; done
.venv/Scripts/python.exe -X utf8 src/examen_concordancia.py $args > results/logs/concordancia_viejos.log 2>&1
echo "TERMINADO $(date +%T)" >> results/logs/concordancia_viejos.log
