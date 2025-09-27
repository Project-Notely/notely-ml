import io

from fastapi import HTTPException, UploadFile
from PIL import Image

from app.models.page_segmentation_models import SegmentationResult
from app.services.page_segmentation.gemini_segmentor import GeminiSegmentor
from app.services.query_parser import QueryParser


async def segment_document(file: UploadFile, query: str) -> SegmentationResult:
    """Orchestrates the document segmentation by first parsing the query
    and then executing the segmentation.
    """
    
    try:
        # 1. Parse the user's natural language query
        parser = QueryParser()
        structured_query = await parser.parse(user_query=query)

        # 2. Read the image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # 3. Segment the document using the structured query
        segmentor = GeminiSegmentor()
        bbox_data = segmentor.segment(image, extracted_query=structured_query.query)

        return SegmentationResult(
            bbox_data=bbox_data,
        )
    except ValueError as e:
        # Catch specific value errors, e.g., from the segmentor
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # General exception for other errors
        raise HTTPException(status_code=500, detail=f"Failed to process request: {e}")
