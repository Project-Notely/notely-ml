from fastapi import FastAPI
from services.unstructured_segmentation.api import router as unstructured_router

app = FastAPI(
    title="Notely ML API",
    description="Machine Learning services for document processing",
    version="1.0.0"
)

# Include the unstructured segmentation service
app.include_router(unstructured_router)

@app.get("/")
async def root():
    return {"message": "Hello from Notely ML"}

@app.get("/health")
async def health():
    return {"status": "ok"}
