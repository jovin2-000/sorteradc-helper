#!/bin/bash
cd "$(dirname "$0")"
docker compose down 2>/dev/null
docker compose build
docker compose up -d
echo "Helper running at http://localhost:8089"
docker compose logs -f --tail=50
