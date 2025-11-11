#!/bin/bash

# Code-God Stop Script
# Gracefully stops all services

set -e

echo "======================================"
echo "  Stopping Code-God System"
echo "======================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "Stopping all services..."
docker-compose down

echo ""
echo -e "${GREEN}✓ All services stopped${NC}"
echo ""
echo "To remove all data (including databases and models):"
echo "  docker-compose down -v"
echo ""
