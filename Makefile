.PHONY: help build up down restart logs shell test clean deploy-gcp

# Default target
help:
	@echo "Notely ML Commands"
	@echo "=================="
	@echo ""
	@echo "Docker Commands:"
	@echo "  make build          - Build Docker images"
	@echo "  make up             - Start services"
	@echo "  make down           - Stop services"
	@echo "  make restart        - Restart services"
	@echo "  make logs           - View logs"
	@echo "  make shell          - Open shell in container"
	@echo "  make test           - Run tests in container"
	@echo "  make clean          - Remove containers, images, and volumes"
	@echo "  make health         - Check service health"
	@echo ""
	@echo "Deployment Commands:"
	@echo "  make deploy-gcp     - Deploy to Google Cloud Run"
	@echo ""

# Build Docker images
build:
	docker-compose build

# Start services in detached mode
up:
	docker-compose up -d

# Stop services
down:
	docker-compose down

# Restart services
restart:
	docker-compose restart

# View logs
logs:
	docker-compose logs -f notely-ml

# Open shell in running container
shell:
	docker-compose exec notely-ml /bin/bash

# Run tests in container
test:
	docker-compose exec notely-ml poetry run pytest

# Check health
health:
	@curl -f http://localhost:8000/api/v1/health || echo "Service is not healthy"

# Clean up everything
clean:
	docker-compose down -v --rmi all

# Build and run
run: build up

# Quick deploy (build, up, and check logs)
deploy: build up logs

# Deploy to Google Cloud Run
deploy-gcp:
	@./deploy-gcp.sh
