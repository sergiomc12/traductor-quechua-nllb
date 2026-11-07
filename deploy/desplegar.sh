#!/usr/bin/env bash
# Local (Git Bash). Empaqueta codigo + pesos, sube por pscp y ejecuta la instalacion remota.
# Uso: bash deploy/desplegar.sh quechua.racinghorse.net correo@dominio.pe
set -euo pipefail
cd "$(dirname "$0")/.."
DOMINIO="${1:?uso: desplegar.sh <subdominio> <correo>}"; EMAIL="${2:?falta correo}"
set -a; . ./.env; set +a
PUTTY="/c/Program Files/PuTTY"
OPC=(-batch -pw "$PASSWORD_REMOTE_HOST")
DEST="$USERNAME_REMOTE_HOST@$IP_REMOTE_HOST"

mkdir -p build
export MSYS_NO_PATHCONV=1
tar -czf build/quechua_paquete.tgz src/__init__.py src/servicio deploy models/exp_norm15k_s0/mejor
"$PUTTY/pscp.exe" "${OPC[@]}" build/quechua_paquete.tgz deploy/instalar_en_servidor.sh "$DEST:/tmp/"
"$PUTTY/plink.exe" "${OPC[@]}" "$DEST" "DOMINIO=$DOMINIO EMAIL=$EMAIL bash /tmp/instalar_en_servidor.sh"
