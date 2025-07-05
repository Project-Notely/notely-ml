"""
Document segmentation controller
"""

from fastapi import UploadFile, HTTPException
import tempfile
import os
from pathlib import Path

from app.services.page_segmentation.unstructuredio_processor import SegmentationService
from app.models.page_segmentation_models import SegmentationResult


async def segment_document(
    file: UploadFile,
    strategy: str = "hi_res",
    extract_images: bool = True,
    infer_table_structure: bool = True,
) -> SegmentationResult:
    """
    Segment uploaded document

    Args:
        file: Uploaded file
        strategy: Partitioning strategy
        extract_images: Whether to extract images
        infer_table_structure: Whether to infer table structure

    Returns:
        SegmentationResult: Segmentation results
    """
    temp_path = None
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=f"_{file.filename}"
        ) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        # Create segmentation service and process
        segmentation_service = SegmentationService()
        result = await segmentation_service.segment_document(
            file_path=temp_path,
            strategy=strategy,
            extract_images=extract_images,
            infer_table_structure=infer_table_structure,
        )

        return result

    except Exception as e:
        return SegmentationResult(success=False, error=str(e), segments=[])
    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


async def get_supported_formats() -> list[str]:
    """Get list of supported file formats"""
    service = SegmentationService()
    return service.get_supported_formats()
