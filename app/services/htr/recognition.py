import easyocr
import numpy as np

# global reader instance
reader = None


def initialize_reader():
    """Init the EasyOCR reader"""
    global reader
    if reader is None:
        print("Initializing EasyOCR (this may take a moment the first time)...")
        reader = easyocr.Reader(["en"])
    return reader


def perform_ocr(img: np.ndarray) -> list[tuple[tuple, str, float]]:
    """Perform OCR on an image"""
    try:
        reader = initialize_reader()
        results = reader.readtext(img)
        print(f"EasyOCR found {len(results)} text regions")
        return results
    except Exception as e:
        print(f"Error during OCR: {e}")
        return []


def convert_to_recognition_results(
    ocr_results: list[tuple[tuple, str, float]],
) -> list[tuple[str, tuple, float]]:
    """Convert OCR results to our format"""
    recognition_results = []

    for bbox, text, confidence in ocr_results:
        if not text.strip():
            continue

        # convert ocr bbox [x1, y1, x2, y2] to our format [x, y, w, h]
        x1, y1 = int(bbox[0][0]), int(bbox[0][1])
        x2, y2 = int(bbox[2][0]), int(bbox[2][1])

        box = (x1, y1, x2 - x1, y2 - y1)
        print(f"Text: '{text}', Confidence: {confidence:.2f}, Box: {box}")
        recognition_results.append((text, box, confidence))

    return recognition_results
