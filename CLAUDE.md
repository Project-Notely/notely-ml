# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Notely ML is a FastAPI-based ML service for intelligent note-taking with OCR, document segmentation, and drawing capabilities. The application uses Gemini AI for document understanding and TrOCR/ViT models for OCR processing.

## Development Commands

### Running the Application

```bash
# Development server (with hot reload on port 9999)
poetry run dev

# Or directly with uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 9999
```

### Code Quality

```bash
# Lint code (Ruff)
poetry run lint

# Format code (Ruff)
poetry run format

# Type checking (mypy)
poetry run type-check
```

### Testing

```bash
# Run all tests
poetry run test

# Or with pytest directly
pytest

# Run specific test markers
pytest -m unit           # Unit tests only
pytest -m integration    # Integration tests only
pytest -m "not slow"     # Skip slow tests

# With coverage
pytest --cov=app --cov-report=html
```

### Docker

```bash
# Build and run with docker-compose
docker-compose up --build

# The service runs on port 8888 in Docker
# Health check: http://localhost:8888/api/v1/health
```

## Configuration

The application requires a `.env.local` file with:

```bash
GEMINI_API_KEY=AIza...  # Required, must start with "AI"
DEBUG=true              # Optional, defaults to True
PORT=8000              # Optional, defaults to 8000
MONGODB_URL=...        # Optional, for MongoDB integration
DATABASE_NAME=notely   # Optional, defaults to "notely"
```

Configuration is validated on startup via `app/core/config.py` using Pydantic Settings with strict validation.

## Architecture

### API Structure

The FastAPI application follows a layered architecture:

- **Routes** (`app/api/routes/`): API endpoint definitions
  - `/api/v1/health` - Health check
  - `/api/v1/segment` - Document segmentation from SVG drawings
  - `/api/v1/segment-test` - Document segmentation from file uploads
  - `/api/v1/drawing/*` - Drawing-related endpoints

- **Controllers** (`app/controllers/`): Business logic orchestration
  - Handle request validation and coordinate between services
  - Transform service outputs into API responses

- **Services** (`app/services/`): Core ML and processing logic

- **Models** (`app/models/`): Pydantic models for request/response validation

### Page Segmentation Pipeline

The main document processing flow (in `app/controllers/segmentation_controller.py`):

1. **Input Processing**: Accepts SVG drawings or file uploads
2. **Query Parsing**: Natural language query → structured query (`QueryParser`)
3. **Image Conversion**: SVG → PIL Image (via `utils.svg_to_pil_image`)
4. **Segmentation**: `GeminiSegmentor` detects regions using Gemini 2.5 Flash
5. **Coordinate Transformation**: Normalized (0-1000) → pixel coordinates
6. **Response**: Returns `SegmentationResult` with bounding boxes

Key service: `app/services/page_segmentation/gemini_segmentor.py`
- Resizes images to max 640px for Gemini API
- Uses temperature=0 for deterministic results
- Returns bounding boxes with labels

### Page Analyzer Services

Two implementations exist side-by-side:

1. **Current** (`app/services/page_analyzer/`):
   - TrOCR processor for handwritten/printed text
   - ViT processor for image classification
   - Gemini processor for multimodal understanding
   - Abstract interfaces in `interfaces/interfaces.py`

2. **Legacy** (`app/services/legacy_page_analyzer/`):
   - Previous implementation, kept for reference

The page analyzer defines several abstract interfaces:
- `OCRProcessor` - Text recognition from image regions
- `ImageClassifier` - Image classification
- `DocumentSegmenter` - Document layout analysis
- `RegionExtractor` - Extract regions from images
- `TableProcessor` - Table structure extraction

### Debug Mode

When `DEBUG=true` in config:
- SVG files saved to `data/drawing.svg`
- Converted images saved to `data/converted.png`
- Bounding boxes drawn on images via `utils.draw_debug_bounding_boxes`
- Detailed logging to console

## Code Style & Standards

This project follows strict Python best practices defined in `.cursor/rules/project.mdc`:

### Type Annotations
- **Required** for all functions, methods, and class members
- Use the most specific types possible from the `typing` module
- mypy strict mode is enabled (see `pyproject.toml`)

### Documentation
- **Google-style docstrings required** for all public functions, methods, and classes
- Include: purpose, parameters, return values, and exceptions raised
- Provide usage examples where helpful

### Code Formatting
- **Ruff** is the primary linter and formatter (replaces black, isort, flake8)
- Line length: 88 characters
- PEP 8 compliant
- Convention: Google docstring style

### Testing
- Use `pytest` with `pytest-asyncio` for async tests
- Target 90%+ code coverage
- Test both common cases and edge cases
- Tests excluded from mypy and docstring requirements

### Error Handling
- Use specific exception types
- Provide informative error messages
- Implement custom exception classes when needed
- Avoid bare `except` clauses

### Async Programming
- Prefer `async`/`await` for I/O-bound operations
- FastAPI endpoints should be async when calling async services

## Common Patterns

### Service Initialization
Services typically follow this pattern:
```python
class MyService:
    def __init__(self):
        self.client = SomeClient(api_key=settings.API_KEY)

    def execute(self, input: SomeType) -> ResultType:
        # Processing logic
        pass
```

### API Route Structure
```python
@router.post("/endpoint", response_model=ResponseModel)
async def endpoint_handler(
    request: RequestModel = Body(...),
) -> ResponseModel:
    return await controller.process(request)
```

### Configuration Access
Always access config through the proxy:
```python
from app.core.config import settings

api_key = settings.GEMINI_API_KEY  # Lazy-loaded, validated
```

## Dependencies

Managed via **Poetry** (`pyproject.toml`):

Core ML/AI:
- `torch` - PyTorch for ML models
- `transformers` - Hugging Face models (TrOCR, ViT)
- `google-genai` / `langchain-google-genai` - Gemini AI integration
- `opencv-python` - Image processing

API/Web:
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `pydantic` / `pydantic-settings` - Data validation

Utilities:
- `pdf2image` - PDF processing
- `cairosvg` - SVG to image conversion
- `pymongo` - MongoDB client (optional)

Dev Tools:
- `pytest`, `pytest-asyncio`, `pytest-cov`, `pytest-mock`
- `ruff` - Linting and formatting
- `mypy` - Type checking
- `pre-commit` - Git hooks

## Testing Notes

Test configuration in `pyproject.toml`:
- Test discovery: `tests/test_*.py` and `*_test.py`
- Async mode: auto (via pytest-asyncio)
- Markers: `unit`, `integration`, `slow`
- Coverage source: `app/` directory

## Known Patterns

1. **SVG Processing**: The app converts SVG drawings to PIL Images, then processes them with ML models. The conversion preserves aspect ratio and dimensions.

2. **Coordinate Systems**: Gemini returns normalized coordinates (0-1000), which are converted to pixel coordinates based on original image dimensions.

3. **Model Loading**: ML models (TrOCR, ViT) support multiple variants (handwritten/printed, small/base/large) configured at initialization.

4. **Query Processing**: Natural language queries are parsed by `QueryParser` before being sent to segmentation models.
