"""Sube los pesos del modelo final a Hugging Face (repositorio de modelo) con su ficha técnica.

Lee HF_TOKEN del entorno o de .env (nunca lo imprime). El repositorio se crea PRIVADO; para hacerlo público:
  python scripts/subir_pesos_hf.py --publico

  python scripts/subir_pesos_hf.py                      # sube models/exp_norm15k_s0/mejor a sergiomc12/nllb600m-es-quy
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

RAIZ = Path(__file__).resolve().parent.parent
REPO = "sergiomc12/nllb600m-es-quy"
PESOS = RAIZ / "models" / "exp_norm15k_s0" / "mejor"
FICHA = RAIZ / "docs" / "ficha_modelo_hf.md"


def token() -> str:
    valor = os.environ.get("HF_TOKEN")
    env = RAIZ / ".env"
    if not valor and env.exists():
        for linea in env.read_text(encoding="utf-8").splitlines():
            if linea.startswith("HF_TOKEN="):
                valor = linea.split("=", 1)[1].strip().strip("\"'")
    if not valor:
        raise SystemExit("Falta HF_TOKEN (en el entorno o en .env).")
    return valor


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--publico", action="store_true", help="hace público el repositorio (por defecto es privado)")
    args = ap.parse_args()
    api = HfApi(token=token())
    api.create_repo(REPO, repo_type="model", private=not args.publico, exist_ok=True)
    if args.publico:
        api.update_repo_settings(REPO, repo_type="model", private=False)
    api.upload_folder(folder_path=str(PESOS), repo_id=REPO, repo_type="model", commit_message="Pesos del modelo final (exp_norm15k_s0)")
    api.upload_file(path_or_fileobj=str(FICHA), path_in_repo="README.md", repo_id=REPO, repo_type="model", commit_message="Ficha del modelo")
    info = api.model_info(REPO)
    print(f"https://huggingface.co/{REPO}  privado={info.private}  archivos={[s.rfilename for s in info.siblings]}")


if __name__ == "__main__":
    main()
