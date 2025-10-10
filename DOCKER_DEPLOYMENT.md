# Docker Deployment Guide

This guide explains how to containerize and deploy the Notely ML API using Docker.

## Prerequisites

- Docker Engine 20.10+ installed
- Docker Compose 2.0+ installed
- `.env.local` file configured with your API keys

## Quick Start

### 1. Create Environment File

Create a `.env.local` file in the project root:

```bash
# Application Settings
DEBUG=false
PORT=8000
HOST=0.0.0.0

# API Keys (REQUIRED)
GEMINI_API_KEY=AIza...

# Optional: MongoDB Settings
# MONGODB_URL=mongodb://mongodb:27017
# DATABASE_NAME=notely
```

### 2. Build and Run with Docker Compose

```bash
# Build and start the service
docker-compose up -d

# View logs
docker-compose logs -f notely-ml

# Stop the service
docker-compose down
```

### 3. Access the API

- API: http://localhost:8000
- Health Check: http://localhost:8000/api/v1/health
- API Docs: http://localhost:8000/docs

## Build Options

### Production Build

```bash
# Build the image
docker build -t notely-ml:latest .

# Run the container
docker run -d \
  --name notely-ml-api \
  -p 8000:8000 \
  --env-file .env.local \
  -v $(pwd)/outputs:/app/outputs \
  notely-ml:latest
```

### Development Build

For development with hot-reload, modify docker-compose.yml:

```yaml
services:
  notely-ml:
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      - ./app:/app/app  # Mount app directory for hot-reload
      - ./outputs:/app/outputs
    environment:
      - DEBUG=true
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Container Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | Yes | - | Google Gemini API key |
| `DEBUG` | No | `false` | Enable debug mode (saves bounding box images) |
| `PORT` | No | `8000` | Port to run the server on |
| `HOST` | No | `0.0.0.0` | Host address to bind to |
| `MONGODB_URL` | No | - | MongoDB connection string |
| `DATABASE_NAME` | No | `notely` | MongoDB database name |

### Volumes

- `/app/outputs` - Directory for debug images (when DEBUG=true)
- `/app/data` - Optional data directory for input files

### Health Check

The container includes a health check that runs every 30 seconds:
- Endpoint: `http://localhost:8000/api/v1/health`
- Timeout: 10s
- Retries: 3
- Start period: 40s

## Multi-Stage Build

The Dockerfile uses a multi-stage build for optimization:

1. **Builder stage**: Installs Poetry and dependencies
2. **Final stage**: Copies only necessary files, resulting in a smaller image

Benefits:
- Smaller final image size (~1-2GB vs 3-4GB)
- Faster deployment
- Better security (fewer attack surfaces)

## Docker Compose Services

### Main API Service

```yaml
notely-ml:
  - Runs the FastAPI application
  - Exposes port 8000
  - Includes health checks
  - Runs as non-root user for security
```

### Optional MongoDB Service

Uncomment the `mongodb` service in `docker-compose.yml` if you need a database:

```yaml
mongodb:
  - MongoDB 7 database
  - Persistent data storage
  - Pre-configured credentials
```

## Production Deployment

### Using Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Deploy the stack
docker stack deploy -c docker-compose.yml notely

# Check services
docker service ls

# Scale the service
docker service scale notely_notely-ml=3
```

### Using Kubernetes

Create a deployment:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: notely-ml
spec:
  replicas: 3
  selector:
    matchLabels:
      app: notely-ml
  template:
    metadata:
      labels:
        app: notely-ml
    spec:
      containers:
      - name: notely-ml
        image: notely-ml:latest
        ports:
        - containerPort: 8000
        env:
        - name: GEMINI_API_KEY
          valueFrom:
            secretKeyRef:
              name: notely-secrets
              key: gemini-api-key
        livenessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 40
          periodSeconds: 30
```

### Cloud Deployment

#### AWS ECS

1. Push image to ECR:
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker tag notely-ml:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/notely-ml:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/notely-ml:latest
```

2. Create ECS task definition with the image
3. Deploy to ECS cluster

#### Google Cloud Run

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT_ID/notely-ml

# Deploy to Cloud Run
gcloud run deploy notely-ml \
  --image gcr.io/PROJECT_ID/notely-ml \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY=AIza...
```

#### Azure Container Instances

```bash
# Push to ACR
az acr build --registry myregistry --image notely-ml:latest .

# Deploy to ACI
az container create \
  --resource-group myResourceGroup \
  --name notely-ml \
  --image myregistry.azurecr.io/notely-ml:latest \
  --cpu 2 --memory 4 \
  --ports 8000 \
  --environment-variables GEMINI_API_KEY=AIza...
```

## Troubleshooting

### Container won't start

```bash
# Check logs
docker-compose logs notely-ml

# Check if .env.local exists
ls -la .env.local

# Verify environment variables
docker-compose config
```

### Out of memory errors

Increase container memory limits in docker-compose.yml:

```yaml
services:
  notely-ml:
    deploy:
      resources:
        limits:
          memory: 4G
```

### Permission denied errors

The container runs as user `appuser` (UID 1000). Ensure mounted volumes have correct permissions:

```bash
sudo chown -R 1000:1000 ./outputs
```

## Monitoring

### View Logs

```bash
# Follow logs
docker-compose logs -f

# Last 100 lines
docker-compose logs --tail=100

# Specific service
docker-compose logs notely-ml
```

### Resource Usage

```bash
# Container stats
docker stats notely-ml-api

# Detailed inspect
docker inspect notely-ml-api
```

## Security Best Practices

1. ✅ Runs as non-root user (appuser)
2. ✅ Multi-stage build reduces attack surface
3. ✅ No unnecessary packages in final image
4. ✅ Environment variables for secrets (never hardcode)
5. ✅ Health checks for availability monitoring
6. ⚠️ Use secrets management (Docker secrets, Kubernetes secrets, etc.) in production
7. ⚠️ Enable HTTPS/TLS in production (use reverse proxy like Nginx/Caddy)
8. ⚠️ Implement rate limiting and authentication

## Next Steps

- [ ] Set up CI/CD pipeline for automated builds
- [ ] Configure monitoring and alerting
- [ ] Implement log aggregation (ELK, Datadog, etc.)
- [ ] Set up auto-scaling based on load
- [ ] Configure backup strategy for outputs/data
- [ ] Implement API authentication (OAuth2, JWT)
