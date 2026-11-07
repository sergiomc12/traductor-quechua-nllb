#!/bin/bash
cd /d/Backend/traductorquechua || exit 1
.venv/Scripts/python.exe -X utf8 src/finetune.py --nombre t4_emulada_s0 --semilla 0 --epocas 3 --batch 32 --lr 3e-5 --precision fp16 --acumulacion 4 --limite-vram-gb 15 --no-guardar-pesos > results/logs/t4_emulada_s0.log 2>&1
echo "TERMINADO $(date +%T)" >> results/logs/t4_emulada_s0.log
