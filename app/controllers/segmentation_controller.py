import io

from fastapi import HTTPException, UploadFile
from PIL import Image

from app.models.page_segmentation_models import (
    BoundingBox,
    DocumentSegment,
    SegmentationResult,
    SegmentType,
)
from app.services.page_segmentation.gemini_segmentor import GeminiSegmentor
from app.services.query_parser.query_parser import QueryParser


async def segment_document(file: UploadFile, query: str) -> SegmentationResult:
    """Orchestrates the document segmentation by first parsing the query
    and then executing the segmentation.
    """
    try:
        # parse natural language query
        parser = QueryParser()
        processed_query = await parser.execute(user_query=query)

        # read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # segment the document using the processed query
        segmentor = GeminiSegmentor()
        bbox_data = segmentor.execute(image=image, query=processed_query.query)

        # transform bbox_data into DocumentSegment objects
        segments = []
        for item in bbox_data:
            if "pixel_coords" in item:
                coords = item["pixel_coords"]
                segment = DocumentSegment(
                    text=item.get("label", ""),
                    segment_type=SegmentType.TEXT,  # default type
                    bbox=BoundingBox(**coords),
                    confidence=1.0,  # gemini doesn't provide confidence scores
                    metadata=item,
                )
                segments.append(segment)

        return SegmentationResult(
            success=True,
            segments=segments,
            total_segments=len(segments),
            strategy_used="gemini-2.5-flash",
        )
    except ValueError as e:
        # catch specific value errors, e.g., from the segmentor
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # general exception for other errors
        raise HTTPException(status_code=500, detail=f"Failed to process request: {e}")
