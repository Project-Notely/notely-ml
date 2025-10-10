# Docker Quick Start Guide

Get your Notely ML API running in Docker in 3 simple steps!

## Prerequisites

- Docker and Docker Compose installed
- Your Gemini API key

## Step 1: Configure Environment

Copy the sample environment file and add your API key:

```bash
cp .env.docker.sample .env.local
```

Edit `.env.local` and set your Gemini API key:
```bash
GEMINI_API_KEY=AIza_your_actual_key_here
DEBUG=false
```

## Step 2: Build and Run

```bash
# Using Docker Compose (recommended)
docker-compose up -d

# Or using Make
make up
```

## Step 3: Test the API

```bash
# Check health
curl http://localhost:8000/api/v1/health

# View API documentation
open http://localhost:8000/docs
```

## Common Commands

```bash
# View logs
docker-compose logs -f
# or
make logs

# Stop services
docker-compose down
# or
make down

# Restart services
docker-compose restart
# or
make restart

# Open shell in container
docker-compose exec notely-ml /bin/bash
# or
make shell
```

## Test the API

```bash
# Test document segmentation
curl -X POST "http://localhost:8000/api/v1/segment" \
  -F "file=@data/notes.png" \
  -F "query=find the main title"
```

## Troubleshooting

### Container won't start
```bash
# Check logs
docker-compose logs notely-ml

# Verify .env.local exists and has GEMINI_API_KEY
cat .env.local | grep GEMINI_API_KEY
```

### Port already in use
Change the port in `.env.local`:
```bash
PORT=8001
```

Then rebuild:
```bash
docker-compose down
docker-compose up -d
```

### Out of memory
Add memory limits in `docker-compose.yml`:
```yaml
services:
  notely-ml:
    deploy:
      resources:
        limits:
          memory: 4G
```

## Next Steps

- Read [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md) for production deployment
- Configure MongoDB if needed
- Set up monitoring and logging
- Deploy to cloud (AWS, GCP, Azure)

## Support

For issues, check:
1. Container logs: `docker-compose logs`
2. Health endpoint: `http://localhost:8000/api/v1/health`
3. API docs: `http://localhost:8000/docs`
