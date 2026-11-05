#!/usr/bin/env bash
# Se ejecuta EN el servidor (AlmaLinux, como root). Uso: DOMINIO=quechua.racinghorse.net EMAIL=x@y bash instalar_en_servidor.sh
# Aislado de otras apps: usuario 'quechua', /opt/quechua, puerto 8090, servicio quechua-api, conf quechua.*.conf.
# NO toca traductor.service / traductor.racinghorse.net (tutor de ingles) ni cyberops.
set -euo pipefail
: "${DOMINIO:?falta DOMINIO}" "${EMAIL:?falta EMAIL}"
APP=/opt/quechua
CONF=/etc/nginx/conf.d/$DOMINIO.conf

# El DNS debe resolver a esta IP, o Let's Encrypt fallara.
IP_LOCAL=$(curl -fs https://ifconfig.me || true)
IP_DNS=$(getent hosts "$DOMINIO" | awk '{print $1; exit}' || true)
[ -n "$IP_DNS" ] && [ "$IP_DNS" = "$IP_LOCAL" ] || { echo "DNS de $DOMINIO ($IP_DNS) no apunta a esta IP ($IP_LOCAL). Crea el registro A y reintenta."; exit 2; }

dnf install -y python3 python3-pip nginx certbot
id quechua >/dev/null 2>&1 || useradd --system --home-dir "$APP" --shell /sbin/nologin quechua
mkdir -p "$APP"
tar -xzf /tmp/quechua_paquete.tgz -C "$APP"
chown -R root:quechua "$APP"; chmod -R g+rX,o-rwx "$APP"

[ -x "$APP/.venv/bin/python" ] || python3 -m venv "$APP/.venv"
"$APP/.venv/bin/pip" install --quiet --upgrade pip
"$APP/.venv/bin/pip" install --quiet torch --index-url https://download.pytorch.org/whl/cpu
"$APP/.venv/bin/pip" install --quiet transformers sentencepiece fastapi "uvicorn[standard]" "pydantic>=2"
restorecon -R "$APP" 2>/dev/null || true

install -m 644 "$APP/deploy/quechua-api.service" /etc/systemd/system/quechua-api.service
systemctl daemon-reload
systemctl enable --now quechua-api

# nginx fase 1 (:80, reto ACME) -> certificado -> fase 2 (HTTPS + proxy)
mkdir -p /var/www/certbot
sed "s/__DOMINIO__/$DOMINIO/g" "$APP/deploy/nginx.http.tpl" > "$CONF"
nginx -t && systemctl reload nginx
[ -d "/etc/letsencrypt/live/$DOMINIO" ] || certbot certonly --webroot -w /var/www/certbot -d "$DOMINIO" -m "$EMAIL" --agree-tos --no-eff-email -n
sed "s/__DOMINIO__/$DOMINIO/g" "$APP/deploy/nginx.https.tpl" > "$CONF"
nginx -t && systemctl reload nginx

for i in $(seq 1 40); do curl -fs http://127.0.0.1:8090/salud && break || sleep 5; done
echo; curl -fs "https://$DOMINIO/salud"; echo
