"""Prueba de concepto: traducir quechua -> espanol con la API de Polytranslator.

Uso (desde la raiz del proyecto):
    python scripts/polytranslator_poc.py quote               # cotizacion gratis del documento
    python scripts/polytranslator_poc.py text "Rimaykullayki" # texto corto via POST /translate
    python scripts/polytranslator_poc.py doc                 # traduce el archivo completo (gasta creditos)

La clave se lee de POLYTRANSLATOR_API_KEY (.env). Nunca se imprime.
"""
import argparse
import html
import os
import re
import sys
import uuid
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://www.polytranslator.com/api/v1"
SRC, TGT = "quy_Latn", "spa_Latn"
DEFAULT_FILE = ROOT / "docs" / "literatura-quechua-central.txt"
OUT_DIR = ROOT / "results" / "polytranslator"


def load_key() -> str:
    key = os.environ.get("POLYTRANSLATOR_API_KEY")
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("POLYTRANSLATOR_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"')
    if not key:
        sys.exit("Falta POLYTRANSLATOR_API_KEY en .env")
    return key


def check(resp: requests.Response) -> dict:
    if resp.status_code >= 400:
        err = resp.json().get("error", {}) if resp.headers.get("content-type", "").startswith("application/json") else {}
        sys.exit(f"HTTP {resp.status_code} {err.get('code', '')}: {err.get('message', resp.text[:200])}")
    return resp.json()


def quote(auth: dict, path: Path, tier: str) -> dict:
    with path.open("rb") as f:
        resp = requests.post(
            f"{BASE}/documents/quote",
            headers=auth,
            files={"file": f},
            data={"src": SRC, "tgt": TGT, "tier": tier, "output": "text"},
            timeout=120,
        )
    return check(resp)


def translate_text(auth: dict, text: str, tier: str) -> str:
    resp = requests.post(
        f"{BASE}/translate",
        headers={**auth, "Idempotency-Key": str(uuid.uuid4())},
        json={"src": SRC, "tgt": TGT, "text": text, "tier": tier, "max_credits": 5},
        timeout=120,
    )
    return check(resp)


def translate_document(auth: dict, path: Path, tier: str, max_credits: int) -> Path:
    q = quote(auth, path, tier)
    print(f"Cotizacion: {q['credits']} creditos | saldo: {q['balance_credits']}")
    if not q["sufficient_credits"]:
        sys.exit("Saldo insuficiente: compra creditos en el dashboard de Polytranslator y reintenta.")
    request_id = str(uuid.uuid4())
    with path.open("rb") as f:
        doc = check(requests.post(
            f"{BASE}/documents",
            headers={**auth, "Idempotency-Key": request_id},
            files={"file": f},
            data={"src": SRC, "tgt": TGT, "tier": tier, "output": "text",
                  "quote_id": q["quote_id"], "max_credits": max_credits},
            timeout=300,
        ))
    while doc["status"] in ("queued", "processing"):
        doc = check(requests.get(f"{BASE}/documents/{doc['id']}?wait=60", headers=auth, timeout=90))
        print("estado:", doc["status"], doc.get("progress"))
    if doc["status"] != "completed":
        sys.exit(f"Fallo: {doc.get('error')}")
    raw = requests.get(doc["files"]["document"], headers=auth, timeout=300)
    raw.raise_for_status()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{path.stem}.es.html"
    out.write_bytes(raw.content)  # copia original (se conserva 30 dias en el servidor)
    plain = html.unescape(re.sub(r"<[^>]+>", "\n", raw.text))
    txt = OUT_DIR / f"{path.stem}.es.txt"
    txt.write_text(re.sub(r"\n{3,}", "\n\n", plain).strip(), encoding="utf-8")
    print("creditos:", doc.get("credits"))
    return txt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("modo", choices=["quote", "text", "doc"])
    p.add_argument("texto", nargs="?", default="Huk uchukllallan runa patrunpa hatun chakranman aywanaq.")
    p.add_argument("--archivo", type=Path, default=DEFAULT_FILE)
    p.add_argument("--tier", choices=["fast", "advanced"], default="fast")
    p.add_argument("--max-credits", type=int, default=100)
    a = p.parse_args()
    auth = {"Authorization": f"Bearer {load_key()}"}

    if a.modo == "quote":
        print(quote(auth, a.archivo, a.tier))
    elif a.modo == "text":
        print(translate_text(auth, a.texto, a.tier))
    else:
        print("Guardado en:", translate_document(auth, a.archivo, a.tier, a.max_credits))


if __name__ == "__main__":
    main()
