#!/usr/bin/env bash
# Local (Git Bash). Empaqueta codigo + pesos, sube por pscp y ejecuta la instalacion remota.
# Uso: bash deploy/desplegar.sh traductor.midominio.pe correo@dominio.pe
set -euo pipefail
cd "$(dirname "$0")/.."
DOMINIO="${1:?uso: desplegar.sh <subdominio> <correo>}"; EMAIL="${2:?falta correo}"
set -a; . ./.env; set +a
PUTTY="/c/Program Files/PuTTY"
OPC=(-batch -pw "$PASSWORD_REMOTE_HOST")
DEST="$USERNAME_REMOTE_HOST@$IP_REMOTE_HOST"

tar -czf /tmp/traductor_paquete.tgz src/servicio src/__init__.py deploy models/principal_s0/mejor 2>/dev/null \
  || tar -czf /tmp/traductor_paquete.tgz src/servicio deploy models/principal_s0/mejor
"$PUTTY/pscp.exe" "${OPC[@]}" /tmp/traductor_paquete.tgz deploy/instalar_en_servidor.sh "$DEST:/tmp/"
"$PUTTY/plink.exe" "${OPC[@]}" "$DEST" "DOMINIO=$DOMINIO EMAIL=$EMAIL bash /tmp/instalar_en_servidor.sh"
