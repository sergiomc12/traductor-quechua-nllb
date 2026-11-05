# Fase 2: HTTP->HTTPS + proxy al API. Sin default_server (ya existe uno en conf.d/traductor.racinghorse.net.conf).
limit_req_zone $binary_remote_addr zone=quechua_api:10m rate=5r/s;

server {
    listen 80;
    listen [::]:80;
    server_name __DOMINIO__;
    server_tokens off;
    location /.well-known/acme-challenge/ { root /var/www/certbot; }
    location / { return 301 https://$host$request_uri; }
}

server {
    listen 443 ssl;
    listen [::]:443 ssl;
    http2 on;
    server_name __DOMINIO__;
    server_tokens off;

    ssl_certificate     /etc/letsencrypt/live/__DOMINIO__/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/__DOMINIO__/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:quechua_ssl:5m;
    ssl_session_timeout 1h;
    ssl_session_tickets off;

    add_header Strict-Transport-Security "max-age=31536000" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;
    add_header Referrer-Policy "no-referrer" always;

    client_max_body_size 16k;
    client_body_timeout 10s;
    client_header_timeout 10s;

    location / {
        limit_req zone=quechua_api burst=10 nodelay;
        proxy_pass http://127.0.0.1:8090;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $remote_addr;
        proxy_set_header X-Forwarded-Proto https;
        proxy_set_header Connection "";
        proxy_connect_timeout 5s;
        proxy_read_timeout 120s;   # inferencia en CPU
    }
}
