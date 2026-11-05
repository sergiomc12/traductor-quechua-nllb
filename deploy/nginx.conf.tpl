# Subdominio del traductor. Reemplazar __DOMINIO__ (lo hace deploy/desplegar.sh).
# Primero se sirve en :80 (necesario para el reto de certbot); certbot --nginx agrega el bloque 443.
limit_req_zone $binary_remote_addr zone=traductor_rl:10m rate=5r/s;

server {
    listen 80;
    listen [::]:80;
    server_name __DOMINIO__;

    client_max_body_size 16k;

    location / {
        limit_req zone=traductor_rl burst=10 nodelay;
        proxy_pass http://127.0.0.1:8088;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
    }
}
