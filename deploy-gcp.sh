#!/bin/bash

# Notely ML API - GCP Cloud Run Deployment Script
# This script deploys your Docker container to Google Cloud Run

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SERVICE_NAME="notely-ml-api"
REGION="us-central1"
PLATFORM="managed"
PORT=8888
MEMORY="2Gi"
CPU="2"
TIMEOUT="300"

echo -e "${GREEN}🚀 Notely ML API - GCP Deployment${NC}"
echo "======================================"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ gcloud CLI is not installed${NC}"
    echo "Please install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if user is logged in
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
    echo -e "${YELLOW}⚠️  Not logged in to gcloud${NC}"
    echo "Running: gcloud auth login"
    gcloud auth login
fi

# Get current project
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)

if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}❌ No GCP project set${NC}"
    echo "Please set your project:"
    echo "  gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

echo -e "${GREEN}✓ Using project: ${PROJECT_ID}${NC}"
echo ""

# Ask for Gemini API key
echo -e "${YELLOW}Please enter your Gemini API key:${NC}"
read -s GEMINI_API_KEY

if [ -z "$GEMINI_API_KEY" ]; then
    echo -e "${RED}❌ Gemini API key is required${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Deployment Configuration:${NC}"
echo "  Service: $SERVICE_NAME"
echo "  Region: $REGION"
echo "  Memory: $MEMORY"
echo "  CPU: $CPU"
echo "  Port: $PORT"
echo ""

# Ask for confirmation
read -p "Deploy to Cloud Run? (y/N) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled"
    exit 0
fi

echo ""
echo -e "${GREEN}📦 Enabling required APIs...${NC}"

# Enable required APIs
gcloud services enable run.googleapis.com --quiet
gcloud services enable cloudbuild.googleapis.com --quiet

echo -e "${GREEN}✓ APIs enabled${NC}"
echo ""

# Deploy
echo -e "${GREEN}🚀 Deploying to Cloud Run...${NC}"
echo "This may take a few minutes..."
echo ""

gcloud run deploy $SERVICE_NAME \
  --source . \
  --platform $PLATFORM \
  --region $REGION \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY=$GEMINI_API_KEY,DEBUG=false,PORT=$PORT \
  --port $PORT \
  --memory $MEMORY \
  --cpu $CPU \
  --timeout $TIMEOUT \
  --max-instances 10 \
  --min-instances 0

# Get the service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
  --region $REGION \
  --format 'value(status.url)')

echo ""
echo -e "${GREEN}✅ Deployment successful!${NC}"
echo ""
echo "======================================"
echo -e "${GREEN}Your API is live at:${NC}"
echo -e "${YELLOW}$SERVICE_URL${NC}"
echo ""
echo "Endpoints:"
echo "  Health: $SERVICE_URL/api/v1/health"
echo "  Docs:   $SERVICE_URL/docs"
echo "  Segment: $SERVICE_URL/api/v1/segment"
echo ""
echo "======================================"
echo ""

# Test health endpoint
echo -e "${GREEN}🧪 Testing health endpoint...${NC}"
HEALTH_RESPONSE=$(curl -s "$SERVICE_URL/api/v1/health")

if echo "$HEALTH_RESPONSE" | grep -q "healthy"; then
    echo -e "${GREEN}✓ Health check passed${NC}"
    echo "Response: $HEALTH_RESPONSE"
else
    echo -e "${RED}❌ Health check failed${NC}"
    echo "Response: $HEALTH_RESPONSE"
fi

echo ""
echo -e "${GREEN}📊 View logs:${NC}"
echo "  gcloud run services logs read $SERVICE_NAME --region $REGION"
echo ""
echo -e "${GREEN}📈 View service details:${NC}"
echo "  gcloud run services describe $SERVICE_NAME --region $REGION"
echo ""
echo -e "${GREEN}🌐 Open in browser:${NC}"
echo "  $SERVICE_URL/docs"
echo ""
