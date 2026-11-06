"""Servicio HTTP de traduccion es<->qu (Ayacucho) con el NLLB ajustado.

  MODELO_DIR=models/principal_s0/mejor uvicorn src.servicio.api:app --host 127.0.0.1 --port 8088
Endpoints: GET /salud, POST /traducir {"texto","origen","destino"}.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

CODIGOS = {"es": "spa_Latn", "qu": "quy_Latn"}
MAX_CHARS = 1000
MAX_LEN = 128
estado: dict = {}


class Peticion(BaseModel):
    texto: str = Field(min_length=1, max_length=MAX_CHARS)
    origen: Literal["es", "qu"]
    destino: Literal["es", "qu"]


@asynccontextmanager
async def ciclo(_: FastAPI):
    ruta = os.environ.get("MODELO_DIR", "models/principal_s0/mejor")
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    estado["tok"] = AutoTokenizer.from_pretrained(ruta)
    estado["modelo"] = AutoModelForSeq2SeqLM.from_pretrained(ruta).to(dispositivo).eval()
    estado["dispositivo"] = dispositivo
    yield
    estado.clear()


app = FastAPI(title="Traductor Quechua-Espanol", version="1.0", lifespan=ciclo)


@app.get("/salud")
def salud() -> dict:
    return {"ok": "modelo" in estado, "dispositivo": estado.get("dispositivo")}


@app.post("/traducir")
def traducir(p: Peticion) -> dict:
    if p.origen == p.destino:
        raise HTTPException(422, "origen y destino deben ser distintos")
    tok, modelo = estado["tok"], estado["modelo"]
    tok.src_lang = CODIGOS[p.origen]
    enc = tok(p.texto, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(estado["dispositivo"])
    with torch.inference_mode():
        gen = modelo.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(CODIGOS[p.destino]), num_beams=4, no_repeat_ngram_size=3, max_new_tokens=MAX_LEN)
    return {"traduccion": tok.batch_decode(gen, skip_special_tokens=True)[0], "origen": p.origen, "destino": p.destino}
