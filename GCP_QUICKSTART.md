# GCP Cloud Run - Quick Start

Deploy your Notely ML API to Google Cloud Run in 5 minutes!

## Prerequisites

1. **Google Cloud account** ([Sign up](https://cloud.google.com/))
2. **Billing enabled** on your GCP project
3. **gcloud CLI installed**:
   ```bash
   # macOS
   brew install --cask google-cloud-sdk

   # Or download from:
   # https://cloud.google.com/sdk/docs/install
   ```

## One-Command Deployment

### Option 1: Use the Deployment Script (Easiest)

```bash
# Make sure you're in the project directory
cd /path/to/notely-ml

# Run the deployment script
./deploy-gcp.sh
```

The script will:
- ✅ Check if gcloud is installed and configured
- ✅ Enable required APIs
- ✅ Ask for your Gemini API key
- ✅ Deploy your container to Cloud Run
- ✅ Test the deployment
- ✅ Show you the live URL

### Option 2: Manual Deployment

```bash
# 1. Login to gcloud
gcloud auth login

# 2. Set your project
gcloud config set project YOUR_PROJECT_ID

# 3. Enable APIs
gcloud services enable run.googleapis.com cloudbuild.googleapis.com

# 4. Deploy
gcloud run deploy notely-ml-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY=your_key_here,DEBUG=false \
  --port 8888 \
  --memory 2Gi \
  --cpu 2
```

## What You Get

After deployment, you'll receive a URL like:
```
https://notely-ml-api-RANDOM-uc.a.run.app
```

### Endpoints

- **Health**: `https://your-url/api/v1/health`
- **API Docs**: `https://your-url/docs`
- **Segment**: `POST https://your-url/api/v1/segment`

## Update Your Frontend

Update your frontend to use the Cloud Run URL:

```javascript
// Before (local)
const API_URL = 'http://localhost:8888';

// After (production)
const API_URL = 'https://notely-ml-api-RANDOM-uc.a.run.app';
```

Don't forget to update CORS in `app/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://your-frontend-domain.com",  # Add your frontend URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Then redeploy:
```bash
./deploy-gcp.sh
```

## Cost Estimate

Cloud Run pricing is very affordable:

- **First 2 million requests/month**: FREE
- **CPU/Memory**: ~$0.00002400 per vCPU-second
- **Memory**: ~$0.00000250 per GiB-second
- **Scales to zero**: No cost when not in use

**Estimated cost for light usage**: $0-10/month

## Useful Commands

```bash
# View logs
gcloud run services logs read notely-ml-api --region us-central1

# Get service URL
gcloud run services describe notely-ml-api --region us-central1 --format='value(status.url)'

# Update environment variables
gcloud run services update notely-ml-api \
  --update-env-vars DEBUG=true \
  --region us-central1

# Delete service
gcloud run services delete notely-ml-api --region us-central1
```

## Troubleshooting

### "Permission denied" errors
```bash
# Re-authenticate
gcloud auth login
```

### Service won't start
```bash
# Check logs
gcloud run services logs read notely-ml-api --region us-central1 --limit 50
```

### Build fails
```bash
# Enable Cloud Build API
gcloud services enable cloudbuild.googleapis.com
```

## Next Steps

1. ✅ **Custom Domain**: Map your own domain
2. ✅ **CI/CD**: Set up automated deployments
3. ✅ **Monitoring**: Configure alerts and dashboards
4. ✅ **Security**: Add authentication to your endpoints

See [GCP_DEPLOYMENT.md](GCP_DEPLOYMENT.md) for detailed instructions.

## Support

- 📚 [Cloud Run Documentation](https://cloud.google.com/run/docs)
- 💰 [Pricing Calculator](https://cloud.google.com/products/calculator)
- 🎓 [Cloud Run Tutorials](https://cloud.google.com/run/docs/tutorials)
