# Deploying to Google Cloud Platform (GCP)

This guide covers deploying your Notely ML API to Google Cloud Run - a serverless container platform.

## Prerequisites

1. **Google Cloud account** with billing enabled
2. **gcloud CLI** installed ([Install Guide](https://cloud.google.com/sdk/docs/install))
3. **Docker** installed locally
4. Your **Gemini API key**

## Quick Setup

### 1. Install and Configure gcloud CLI

```bash
# Install gcloud CLI (if not already installed)
# macOS
brew install --cask google-cloud-sdk

# Or download from: https://cloud.google.com/sdk/docs/install

# Initialize gcloud
gcloud init

# Login to your Google account
gcloud auth login

# Set your project (replace PROJECT_ID with your actual project ID)
gcloud config set project PROJECT_ID
```

### 2. Create a New GCP Project (Optional)

```bash
# Create a new project
gcloud projects create notely-ml-api --name="Notely ML API"

# Set it as your active project
gcloud config set project notely-ml-api

# Link billing account (required for Cloud Run)
gcloud billing projects link notely-ml-api --billing-account=BILLING_ACCOUNT_ID
```

### 3. Enable Required APIs

```bash
# Enable Cloud Run API
gcloud services enable run.googleapis.com

# Enable Container Registry API
gcloud services enable containerregistry.googleapis.com

# Enable Artifact Registry API (recommended over Container Registry)
gcloud services enable artifactregistry.googleapis.com
```

## Deployment Methods

### Method 1: Direct Deploy from Source (Easiest)

Cloud Run can build and deploy directly from your source code:

```bash
# Navigate to your project directory
cd /path/to/notely-ml

# Deploy directly from source
gcloud run deploy notely-ml-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY=your_gemini_api_key_here \
  --port 8888 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300
```

### Method 2: Build and Push to Artifact Registry (Recommended)

This method gives you more control and is better for production:

#### Step 1: Configure Artifact Registry

```bash
# Create an Artifact Registry repository
gcloud artifacts repositories create notely-ml-repo \
  --repository-format=docker \
  --location=us-central1 \
  --description="Notely ML Docker repository"

# Configure Docker to use gcloud as credential helper
gcloud auth configure-docker us-central1-docker.pkg.dev
```

#### Step 2: Build and Tag Your Image

```bash
# Set variables
PROJECT_ID=$(gcloud config get-value project)
REGION=us-central1
REPO_NAME=notely-ml-repo
IMAGE_NAME=notely-ml-api
TAG=latest

# Build the Docker image
docker build -t ${IMAGE_NAME}:${TAG} .

# Tag for Artifact Registry
docker tag ${IMAGE_NAME}:${TAG} \
  ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${TAG}
```

#### Step 3: Push to Artifact Registry

```bash
# Push the image
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${TAG}
```

#### Step 4: Deploy to Cloud Run

```bash
# Deploy from Artifact Registry
gcloud run deploy notely-ml-api \
  --image ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${TAG} \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY=your_gemini_api_key_here,DEBUG=false \
  --port 8888 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10 \
  --min-instances 0
```

### Method 3: Using Container Registry (Legacy)

```bash
# Configure Docker for GCR
gcloud auth configure-docker

# Tag and push
PROJECT_ID=$(gcloud config get-value project)
docker tag notely-ml-api gcr.io/${PROJECT_ID}/notely-ml-api:latest
docker push gcr.io/${PROJECT_ID}/notely-ml-api:latest

# Deploy
gcloud run deploy notely-ml-api \
  --image gcr.io/${PROJECT_ID}/notely-ml-api:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY=your_key_here \
  --port 8888
```

## Configuration Options

### Environment Variables

Use **Secret Manager** for sensitive data like API keys:

```bash
# Create a secret
echo -n "your_gemini_api_key" | gcloud secrets create gemini-api-key --data-file=-

# Grant Cloud Run access to the secret
gcloud secrets add-iam-policy-binding gemini-api-key \
  --member="serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

# Deploy with secret
gcloud run deploy notely-ml-api \
  --image ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${TAG} \
  --update-secrets=GEMINI_API_KEY=gemini-api-key:latest \
  --region us-central1
```

### Resource Configuration

```bash
# Configure CPU and memory
--memory 2Gi              # Memory allocation (128Mi to 32Gi)
--cpu 2                   # CPU allocation (1, 2, 4, or 8)
--timeout 300             # Request timeout in seconds (max 3600)
--concurrency 80          # Max concurrent requests per instance
--max-instances 10        # Maximum number of instances
--min-instances 0         # Minimum instances (0 for scale to zero)
```

### Custom Domain

```bash
# Map custom domain
gcloud run domain-mappings create \
  --service notely-ml-api \
  --domain api.yourdomain.com \
  --region us-central1

# Follow the instructions to update your DNS records
```

### CORS Configuration

Your CORS is already configured in `app/main.py`. To allow your production frontend:

```python
# Update app/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://yourdomain.com",  # Add your production domain
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Automated Deployment with GitHub Actions

Create `.github/workflows/deploy-gcp.yml`:

```yaml
name: Deploy to Cloud Run

on:
  push:
    branches:
      - main

env:
  PROJECT_ID: your-project-id
  SERVICE_NAME: notely-ml-api
  REGION: us-central1

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Authenticate to Google Cloud
      uses: google-github-actions/auth@v1
      with:
        credentials_json: ${{ secrets.GCP_SA_KEY }}

    - name: Set up Cloud SDK
      uses: google-github-actions/setup-gcloud@v1

    - name: Configure Docker
      run: gcloud auth configure-docker ${{ env.REGION }}-docker.pkg.dev

    - name: Build and push image
      run: |
        docker build -t ${{ env.REGION }}-docker.pkg.dev/${{ env.PROJECT_ID }}/notely-ml-repo/${{ env.SERVICE_NAME }}:${{ github.sha }} .
        docker push ${{ env.REGION }}-docker.pkg.dev/${{ env.PROJECT_ID }}/notely-ml-repo/${{ env.SERVICE_NAME }}:${{ github.sha }}

    - name: Deploy to Cloud Run
      run: |
        gcloud run deploy ${{ env.SERVICE_NAME }} \
          --image ${{ env.REGION }}-docker.pkg.dev/${{ env.PROJECT_ID }}/notely-ml-repo/${{ env.SERVICE_NAME }}:${{ github.sha }} \
          --region ${{ env.REGION }} \
          --platform managed \
          --allow-unauthenticated \
          --set-env-vars DEBUG=false \
          --update-secrets GEMINI_API_KEY=gemini-api-key:latest
```

## Monitoring and Logs

### View Logs

```bash
# View service logs
gcloud run services logs read notely-ml-api --region us-central1

# Follow logs in real-time
gcloud run services logs tail notely-ml-api --region us-central1

# View logs in Cloud Console
# https://console.cloud.google.com/run
```

### Monitoring

```bash
# Get service details
gcloud run services describe notely-ml-api --region us-central1

# View metrics in Cloud Console
# https://console.cloud.google.com/monitoring
```

## Cost Optimization

Cloud Run pricing is based on:
- **CPU and Memory** usage (billed per 100ms)
- **Requests** (first 2 million free per month)
- **Networking** (egress)

### Optimization Tips

1. **Scale to zero** when not in use (`--min-instances 0`)
2. **Right-size resources** (start with 1 CPU / 1Gi memory)
3. **Use request timeouts** to prevent long-running requests
4. **Enable CPU allocation only during requests**:
   ```bash
   --cpu-throttling  # Default, CPU allocated only during requests
   ```

### Estimated Costs

For light usage (< 1M requests/month with 2 CPU, 2Gi RAM):
- **~$0-20/month** (most costs covered by free tier)

For moderate usage (5M requests/month):
- **~$50-100/month**

## Testing Your Deployment

```bash
# Get the service URL
SERVICE_URL=$(gcloud run services describe notely-ml-api \
  --region us-central1 \
  --format 'value(status.url)')

echo "Service URL: $SERVICE_URL"

# Test health endpoint
curl $SERVICE_URL/api/v1/health

# Test segment endpoint
curl -X POST $SERVICE_URL/api/v1/segment \
  -F "file=@data/notes.png" \
  -F "query=find the title"
```

## Troubleshooting

### Service won't start

```bash
# Check logs
gcloud run services logs read notely-ml-api --region us-central1 --limit 50

# Common issues:
# 1. Missing environment variables
# 2. Port mismatch (ensure PORT=8888)
# 3. Container crashes on startup
```

### Authentication errors

```bash
# Re-authenticate
gcloud auth login
gcloud auth application-default login

# Check active account
gcloud auth list
```

### Build failures

```bash
# Enable Cloud Build API
gcloud services enable cloudbuild.googleapis.com

# Check build logs
gcloud builds list --limit 10
gcloud builds log BUILD_ID
```

## Rollback

```bash
# List revisions
gcloud run revisions list --service notely-ml-api --region us-central1

# Rollback to previous revision
gcloud run services update-traffic notely-ml-api \
  --to-revisions REVISION_NAME=100 \
  --region us-central1
```

## Cleanup

```bash
# Delete the service
gcloud run services delete notely-ml-api --region us-central1

# Delete images from Artifact Registry
gcloud artifacts docker images delete \
  us-central1-docker.pkg.dev/PROJECT_ID/notely-ml-repo/notely-ml-api:latest

# Delete the repository
gcloud artifacts repositories delete notely-ml-repo --location us-central1
```

## Security Best Practices

1. ✅ **Use Secret Manager** for API keys
2. ✅ **Enable Cloud Armor** for DDoS protection
3. ✅ **Restrict access** with Cloud IAM
4. ✅ **Use custom domain** with SSL/TLS
5. ✅ **Enable VPC** for private services
6. ✅ **Regular updates** of dependencies
7. ✅ **Monitor and alert** on errors

## Next Steps

- [ ] Set up custom domain
- [ ] Configure CI/CD pipeline
- [ ] Set up monitoring and alerting
- [ ] Implement request authentication
- [ ] Configure auto-scaling policies
- [ ] Set up staging environment
- [ ] Enable Cloud CDN for static assets

## Support

- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Cloud Run Pricing](https://cloud.google.com/run/pricing)
- [Cloud Run Quickstarts](https://cloud.google.com/run/docs/quickstarts)
