import io

from fastapi import APIRouter, File, UploadFile, Form

from app.controllers import segmentation_controller
from app.models.page_segmentation_models import SegmentationResult

router = APIRouter()


@router.post("/segment", response_model=SegmentationResult)
async def segment_document(
    file: UploadFile = File(...),
    query: str = Form(...),
) -> SegmentationResult:
    return await segmentation_controller.segment_document(
        file,
        query,
    )
