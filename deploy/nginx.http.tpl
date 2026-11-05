# Fase 1 (solo :80): sirve el reto ACME. Reemplazar __DOMINIO__.
server {
    listen 80;
    listen [::]:80;
    server_name __DOMINIO__;
    server_tokens off;

    location /.well-known/acme-challenge/ { root /var/www/certbot; }
    location / { return 301 https://$host$request_uri; }
}
