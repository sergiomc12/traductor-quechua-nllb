"""Lanza en secuencia los experimentos de configs/experimentos.json (omite los ya terminados).

  python -X utf8 src/correr_experimentos.py            # todos
  python -X utf8 src/correr_experimentos.py principal_s0
"""
import json
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
cfg = json.loads((RAIZ / "configs" / "experimentos.json").read_text(encoding="utf-8"))
solo = set(sys.argv[1:])
for exp in cfg["experimentos"]:
    if solo and exp["nombre"] not in solo:
        continue
    if (RAIZ / "results" / exp["nombre"] / "test.json").exists():
        print(f"[omitido] {exp['nombre']} ya tiene resultados", flush=True)
        continue
    p = {**cfg["comun"], **exp}
    cmd = [sys.executable, "-X", "utf8", str(RAIZ / "src" / "finetune.py"), "--nombre", p["nombre"], "--semilla", str(p["semilla"]),
           "--epocas", str(p["epocas"]), "--batch", str(p["batch"]), "--lr", str(p["lr"]), "--precision", p["precision"],
           "--frac-biblico", str(p["frac_biblico"]), "--guardar-pesos" if p["guardar_pesos"] else "--no-guardar-pesos"]
    print("[run]", " ".join(cmd), flush=True)
    if subprocess.run(cmd, check=False).returncode != 0:
        print(f"[ERROR] {exp['nombre']} fallo; se continua con el siguiente", flush=True)
