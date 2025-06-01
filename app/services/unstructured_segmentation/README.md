# Unstructured Document Segmentation Service

This service uses the [Unstructured](https://github.com/Unstructured-IO/unstructured) library to segment documents into different types of elements (paragraphs, tables, images, etc.) and creates highlighted PDF outputs showing the segmentation results.

## Features

- **Document Segmentation**: Automatically identifies and segments different document elements
- **PDF Highlighting**: Creates highlighted PDFs showing segmentation boundaries
- **Multiple Formats**: Supports PDF, DOCX, HTML, images, and more
- **REST API**: FastAPI-based REST endpoints for easy integration
- **Data Export**: Export segmentation data in JSON or CSV format

## Supported Element Types

- **Title**: Document titles and headings
- **NarrativeText**: Main body text and paragraphs  
- **Table**: Tables and structured data
- **ListItem**: Bulleted and numbered lists
- **Image/Figure**: Images and figures
- **Header/Footer**: Page headers and footers
- **FigureCaption**: Image and figure captions
- **Address/EmailAddress**: Contact information

## Usage

### Python API

```python
from services.unstructured_segmentation.main import (
    process_pdf_with_highlights,
    segment_document_unstructured
)

# Basic segmentation
elements, stats = segment_document_unstructured("document.pdf")
print(f"Found {stats['total_elements']} elements")

# Create highlighted PDF
success = process_pdf_with_highlights(
    "input.pdf",
    "output_highlighted.pdf", 
    "segmentation_data.json"
)
```

### REST API

Start the FastAPI server:
```bash
cd app
uvicorn main:app --reload
```

#### Segment Document
```bash
curl -X POST "http://localhost:8000/unstructured-segmentation/segment" \
     -F "file=@document.pdf" \
     -F "strategy=hi_res"
```

#### Create Highlighted PDF
```bash
curl -X POST "http://localhost:8000/unstructured-segmentation/segment-with-highlights" \
     -F "file=@document.pdf" \
     -F "strategy=hi_res" \
     --output highlighted.pdf
```

## Configuration

### Strategies

- **`fast`**: Quick processing, lower accuracy
- **`hi_res`**: High-resolution processing with OCR
- **`auto`**: Automatically choose best strategy

### Supported File Formats

- **Documents**: PDF, DOCX, DOC, PPTX, PPT, HTML, TXT, MD, RTF
- **Spreadsheets**: XLSX, XLS  
- **Images**: PNG, JPG, JPEG, TIFF, BMP

## Color Coding

Each element type is highlighted with a different color:

- **Red**: Titles
- **Green**: Text/Paragraphs
- **Blue**: List Items
- **Yellow**: Tables
- **Magenta**: Images/Figures
- **Orange**: Headers
- **Purple**: Footers
- **Cyan**: Figure Captions

## Dependencies

The service requires these additional packages:
```bash
pip install unstructured[pdf] pdf2image reportlab pillow
```

## Examples

See `test_unstructured_service.py` for usage examples.

## API Endpoints

- `POST /unstructured-segmentation/segment`: Segment document and return data
- `POST /unstructured-segmentation/segment-with-highlights`: Create highlighted PDF
- `GET /unstructured-segmentation/element-types`: Get supported element types
- `GET /unstructured-segmentation/supported-formats`: Get supported file formats 