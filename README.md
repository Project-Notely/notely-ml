# Notely ML

Machine learning components for the Notely application.

## Setup

1. Create a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install paddlepaddle paddleocr opencv-python rapidfuzz
   ```

## Scripts

### Control-F for Images (scripts/ctrl-f.py)

This script implements a "Ctrl+F" like search functionality for images containing text. It uses OCR to detect text within images and then fuzzy-matches a search query against the detected text, highlighting matches.

#### Usage

```
python scripts/ctrl-f.py <image_path> <search_query> <output_path> [segment_size]
```

Example:
```
python scripts/ctrl-f.py handwritten_note.jpg "Hello" output.jpg
```

Advanced example with custom segmentation:
```
python scripts/ctrl-f.py large_document.jpg "Important" output.jpg 400
```

#### Parameters

- `image_path`: Path to the input image containing handwritten or printed text
- `search_query`: Text to search for within the image
- `output_path`: Where to save the resulting image with highlights
- `segment_size`: (Optional) Size of image segments in pixels for improved OCR (default: 300)

#### Features

- **Image Segmentation**: Divides large images into smaller segments for more accurate OCR processing
- **Handwriting Recognition**: Uses advanced preprocessing techniques and optimized OCR parameters to improve handwritten text detection
- **Fuzzy Matching**: Employs fuzzy string matching to find approximate matches, useful for handwriting where the OCR might not be perfect
- **Visual Highlighting**: Detected matches are highlighted with a semi-transparent green fill and red border
- **Match Confidence**: Shows the match percentage next to each highlighted region
- **No Matches Notification**: If no matches are found, a message is displayed on the output image

#### Notes on Handwritten Text

For handwritten text, the OCR technology may not be 100% accurate. The script uses the following strategies to improve results:

1. Image segmentation to process small chunks of the image for better text detection
2. Image preprocessing to enhance text visibility
3. Multiple OCR passes with different parameters
4. Fuzzy matching to account for OCR errors (e.g., "Melle" might match with "Hello")

#### Advanced Configuration

- **Segment Size**: For large or dense documents, try increasing the segment size (e.g., 400-500px). For small notes with fine handwriting, try smaller segments (e.g., 150-200px).
- **Fuzzy Threshold**: Adjust the `fuzzy_threshold` parameter in the code (default: 30) to control match sensitivity. Lower values will match more text but may include false positives.
- **Detection Threshold**: Adjust the `detection_threshold` parameter in the code (default: 0.2) to control OCR confidence threshold. Lower values will detect more text but might include poorly recognized text.