from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello from Notely ML"}

@app.get("/health")
async def health():
    return {"status": "ok"}
