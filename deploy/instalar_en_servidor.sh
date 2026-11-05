#!/usr/bin/env bash
# Se ejecuta EN el servidor (como root). Uso: DOMINIO=traductor.midominio.pe EMAIL=yo@x.pe bash instalar_en_servidor.sh
# Idempotente: se puede repetir. No toca otros sitios de nginx.
set -euo pipefail
: "${DOMINIO:?falta DOMINIO}" "${EMAIL:?falta EMAIL para certbot}"
APP=/opt/traductorquechua

command -v apt-get >/dev/null || { echo "Solo Debian/Ubuntu soportado"; exit 1; }
apt-get update -y
DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-venv python3-pip nginx certbot python3-certbot-nginx

id traductor >/dev/null 2>&1 || useradd --system --home "$APP" --shell /usr/sbin/nologin traductor
mkdir -p "$APP"
tar -xzf /tmp/traductor_paquete.tgz -C "$APP"
chown -R traductor:traductor "$APP"

# Entorno Python (torch CPU: el servidor no necesita GPU para un traductor de baja carga)
[ -x "$APP/.venv/bin/python" ] || python3 -m venv "$APP/.venv"
"$APP/.venv/bin/pip" install --upgrade pip
"$APP/.venv/bin/pip" install torch --index-url https://download.pytorch.org/whl/cpu
"$APP/.venv/bin/pip" install transformers sentencepiece fastapi "uvicorn[standard]" "pydantic>=2"

install -m 644 "$APP/deploy/traductor.service" /etc/systemd/system/traductor.service
systemctl daemon-reload
systemctl enable --now traductor

# nginx: un server block propio por subdominio
sed "s/__DOMINIO__/$DOMINIO/g" "$APP/deploy/nginx.conf.tpl" > /etc/nginx/conf.d/traductor.conf
nginx -t
systemctl reload nginx

# TLS (requiere que el DNS de $DOMINIO ya apunte a esta IP)
certbot --nginx -d "$DOMINIO" -m "$EMAIL" --agree-tos --no-eff-email --redirect -n

# Verificacion
for i in $(seq 1 30); do curl -fs http://127.0.0.1:8088/salud && break || sleep 5; done
echo; curl -fs "https://$DOMINIO/salud"
