from fastapi import APIRouter
from app.services.simple_ocr_service.main import OCRService
from app.models.api_models import OCRResponse

router = APIRouter()
ocr_service = OCRService()


@router.post("/ocr/process", response_model=OCRResponse)
async def process_image():
    result = await ocr_service.process_image()

    return OCRResponse(
        success=True,
        text=result,
        confidence=0.0,
    )
