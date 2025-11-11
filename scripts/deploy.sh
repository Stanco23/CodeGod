#!/bin/bash

# Code-God Production Deployment Script

set -e

ENVIRONMENT=${1:-production}

echo "======================================"
echo "  Code-God Production Deployment"
echo "  Environment: $ENVIRONMENT"
echo "======================================"
echo ""

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check prerequisites
echo "Checking prerequisites..."

if [ ! -f .env ]; then
    echo -e "${RED}✗ .env file not found${NC}"
    exit 1
fi

source .env

if [ -z "$ANTHROPIC_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}✗ No API keys configured${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Configuration OK${NC}"

# Production-specific checks
if [ "$ENVIRONMENT" = "production" ]; then
    echo ""
    echo "Production deployment checklist:"
    echo -n "  - Strong database password set? "
    if [ "$POSTGRES_PASSWORD" = "changeme" ]; then
        echo -e "${RED}✗ Please change POSTGRES_PASSWORD${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓${NC}"

    echo -n "  - Domain configured? "
    if [ "$DOMAIN" = "localhost" ]; then
        echo -e "${YELLOW}⚠ Using localhost${NC}"
    else
        echo -e "${GREEN}✓ $DOMAIN${NC}"
    fi

    echo -n "  - SSL enabled? "
    if [ "$SSL_ENABLED" = "true" ]; then
        echo -e "${GREEN}✓${NC}"
        if [ ! -f "$SSL_CERT_PATH" ] || [ ! -f "$SSL_KEY_PATH" ]; then
            echo -e "${RED}✗ SSL certificates not found${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}⚠ SSL not enabled${NC}"
    fi
fi

# Create production directories
echo ""
echo "Creating production directories..."
mkdir -p logs/{master,workers,nginx} \
         backups \
         projects \
         config/nginx \
         config/prometheus \
         config/grafana

echo -e "${GREEN}✓ Directories created${NC}"

# Generate Nginx configuration if needed
if [ "$ENVIRONMENT" = "production" ]; then
    echo ""
    echo "Generating Nginx configuration..."
    cat > config/nginx/default.conf << EOF
upstream master_ai {
    server master-ai:8001;
}

server {
    listen 80;
    server_name $DOMAIN;

    location / {
        proxy_pass http://master_ai;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;

        # Timeouts
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
}
EOF

    if [ "$SSL_ENABLED" = "true" ]; then
        cat >> config/nginx/default.conf << EOF

server {
    listen 443 ssl http2;
    server_name $DOMAIN;

    ssl_certificate $SSL_CERT_PATH;
    ssl_certificate_key $SSL_KEY_PATH;

    location / {
        proxy_pass http://master_ai;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF
    fi

    echo -e "${GREEN}✓ Nginx configuration generated${NC}"
fi

# Pull latest images
echo ""
echo "Building production images..."
docker-compose -f docker-compose.yml build --no-cache

echo -e "${GREEN}✓ Images built${NC}"

# Start services
echo ""
echo "Starting services..."

if [ "$ENVIRONMENT" = "production" ]; then
    docker-compose -f docker-compose.yml \
                   -f docker-compose.prod.yml \
                   --profile monitoring \
                   up -d
else
    docker-compose up -d
fi

# Wait for health checks
echo ""
echo "Waiting for services to be healthy..."
sleep 30

# Verify services
echo ""
echo "Verifying services..."

services=("chromadb:8000/api/v1/heartbeat" "master-ai:8001/health")

for service in "${services[@]}"; do
    IFS=':' read -r name port_path <<< "$service"
    echo -n "  $name... "

    for i in {1..60}; do
        if curl -s http://localhost:${port_path} > /dev/null 2>&1; then
            echo -e "${GREEN}✓${NC}"
            break
        fi
        sleep 1
        if [ $i -eq 60 ]; then
            echo -e "${RED}✗ Failed${NC}"
            docker-compose logs $name
            exit 1
        fi
    done
done

# Create backup script
echo ""
echo "Creating backup script..."
cat > scripts/backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR=./backups
DATE=$(date +%Y%m%d_%H%M%S)

# Backup databases
docker-compose exec -T postgres pg_dump -U codegod codegod > $BACKUP_DIR/postgres_$DATE.sql

# Backup ChromaDB
docker-compose exec -T chromadb tar czf - /chroma/chroma > $BACKUP_DIR/chromadb_$DATE.tar.gz

# Backup projects
tar czf $BACKUP_DIR/projects_$DATE.tar.gz projects/

echo "Backup completed: $DATE"
EOF
chmod +x scripts/backup.sh

echo -e "${GREEN}✓ Backup script created${NC}"

# Setup cron for backups (production only)
if [ "$ENVIRONMENT" = "production" ]; then
    echo ""
    echo "Setting up automated backups..."
    echo "Add to crontab: 0 2 * * * $(pwd)/scripts/backup.sh"
fi

# Final status
echo ""
echo "======================================"
echo "  Deployment Complete!"
echo "======================================"
echo ""
docker-compose ps
echo ""
echo "Access points:"
if [ "$ENVIRONMENT" = "production" ]; then
    echo "  API: https://$DOMAIN"
    echo "  Monitoring: http://$DOMAIN:3000"
else
    echo "  API: http://localhost:8001"
    echo "  Docs: http://localhost:8001/docs"
    echo "  Monitoring: http://localhost:3000"
fi
echo ""
echo "Useful commands:"
echo "  View logs:        docker-compose logs -f"
echo "  Stop services:    docker-compose down"
echo "  Backup data:      ./scripts/backup.sh"
echo "  Scale workers:    docker-compose up -d --scale worker-backend=4"
echo ""
echo -e "${GREEN}Deployment successful! 🚀${NC}"
