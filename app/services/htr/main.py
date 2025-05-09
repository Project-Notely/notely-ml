import cv2
import numpy as np
import re
import easyocr

IMAGE_PATH = "app/services/img3.png"
reader = None  # Will initialize globally once


def initialize_reader():
    """Initialize the EasyOCR reader once"""
    global reader
    if reader is None:
        print("Initializing EasyOCR (this may take a moment the first time)...")
        reader = easyocr.Reader(["en"])
    return reader


def preprocess_image(image_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load and preprocess image for HTR"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image at {image_path}")
            return None, None

        # Convert to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Enhance contrast with CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_img)

        # Simple denoising
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)

        print("Preprocessing complete")
        return img, denoised
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None, None


def perform_ocr(img):
    """Perform OCR on the whole image using EasyOCR"""
    try:
        reader = initialize_reader()

        # EasyOCR returns results as: [[bbox, text, confidence]]
        # bbox is [x1, y1, x2, y2] where (x1, y1) is top-left, (x2, y2) is bottom-right
        results = reader.readtext(img)

        print(f"EasyOCR found {len(results)} text regions")
        return results
    except Exception as e:
        print(f"Error during OCR: {e}")
        return []


def convert_to_recognition_results(ocr_results):
    """Convert EasyOCR results to our format"""
    recognition_results = []

    for bbox, text, confidence in ocr_results:
        if not text.strip():
            continue

        # Convert EasyOCR bbox [x1, y1, x2, y2] to our format (x, y, w, h)
        x1, y1 = int(bbox[0][0]), int(bbox[0][1])
        x2, y2 = int(bbox[2][0]), int(bbox[2][1])

        x = x1
        y = y1
        w = x2 - x1
        h = y2 - y1

        box = (x, y, w, h)

        print(f"Text: '{text}', Confidence: {confidence:.2f}, Box: {box}")
        recognition_results.append((text, box, confidence))

    return recognition_results


def index_text(text_results):
    """Creates a more flexible in-memory index mapping words to their locations."""
    index = {}  # word -> list of (page_num, bounding_box, confidence, original_text)
    page_num = 1  # assuming single page for now

    for text, box, confidence in text_results:
        # Store original text for exact matching
        original_text = text

        # Clean and normalize text for indexing
        text = text.lower().strip()

        # Store both full phrases and individual words
        # First, add the complete phrase
        if text:
            if text not in index:
                index[text] = []
            index[text].append((page_num, box, confidence, original_text))

        # Then add individual words
        words = re.findall(r"\b\w+\b", text.lower())
        for word in words:
            if word and len(word) > 1:  # Only index words with 2+ characters
                if word not in index:
                    index[word] = []
                # Using the same box for the word as for the whole phrase
                index[word].append((page_num, box, confidence, original_text))

    print(f"Indexing complete. Indexed {len(index)} unique words/phrases.")
    return index


def search_text(query, index):
    """Improved search with various matching strategies including partial word matching"""
    query = query.lower().strip()
    results = []
    min_confidence = 0.6  # Minimum confidence threshold (60%)

    # Strategy 1: Exact phrase match (highest priority)
    if query in index:
        for match in index[query]:
            page_num, box, confidence, original_text = match
            if confidence >= min_confidence:
                results.append(
                    (page_num, box, confidence, original_text, 1.0)
                )  # 1.0 = highest match score

    # Strategy 2: All words in query appear in the indexed text (medium priority)
    if not results:
        query_words = set(re.findall(r"\b\w+\b", query.lower()))
        for indexed_text in index.keys():
            if len(indexed_text) > 3:  # Only consider longer indexed texts
                indexed_words = set(re.findall(r"\b\w+\b", indexed_text.lower()))
                # If all query words are in the indexed text
                if query_words.issubset(indexed_words):
                    for match in index[indexed_text]:
                        page_num, box, confidence, original_text = match
                        if confidence >= min_confidence:
                            # Score based on how much of the indexed text matches the query
                            match_score = len(query_words) / len(indexed_words)
                            results.append(
                                (
                                    page_num,
                                    box,
                                    confidence,
                                    original_text,
                                    0.9 * match_score,
                                )
                            )

    # Strategy 3: Individual word matches (lower priority)
    if not results:
        query_words = re.findall(r"\b\w+\b", query.lower())
        for word in query_words:
            if (
                len(word) > 2 and word in index
            ):  # Only search for words with 3+ characters
                for match in index[word]:
                    page_num, box, confidence, original_text = match
                    if confidence >= min_confidence:
                        # Lower match score for single word matches
                        match_score = 0.7 * (len(word) / len(query))
                        results.append(
                            (page_num, box, confidence, original_text, match_score)
                        )

    # Strategy 4: Partial word matching (substring within words - lowest priority but handles "bis" in "bisection")
    if not results:
        min_chars = 2  # Minimum characters to consider for partial matching
        if len(query) >= min_chars:
            for indexed_word in index.keys():
                # Check if query is a substring of the indexed word
                if query in indexed_word:
                    for match in index[indexed_word]:
                        page_num, box, confidence, original_text = match
                        if confidence >= min_confidence:
                            # Score based on how much of the indexed word the query covers
                            match_score = 0.5 * (len(query) / len(indexed_word))
                            results.append(
                                (page_num, box, confidence, original_text, match_score)
                            )

    # Strategy 5: Check individual words in indexed phrases for partial matches
    if not results and len(query) >= min_chars:
        for indexed_phrase in index.keys():
            if len(indexed_phrase) > len(query):  # Only phrases longer than query
                words_in_phrase = re.findall(r"\b\w+\b", indexed_phrase.lower())
                for word in words_in_phrase:
                    if query in word and len(word) >= len(query):
                        for match in index[indexed_phrase]:
                            page_num, box, confidence, original_text = match
                            if confidence >= min_confidence:
                                # Lower match score for partial matches
                                match_score = 0.4 * (len(query) / len(word))
                                results.append(
                                    (
                                        page_num,
                                        box,
                                        confidence,
                                        original_text,
                                        match_score,
                                    )
                                )

    if results:
        # Remove duplicates based on bounding box (same box should not be highlighted twice)
        # But allow different boxes with the same text to be highlighted
        unique_results = []
        seen_boxes = set()  # Track boxes we've seen

        for result in results:
            page_num, box, confidence, original_text, score = result
            box_key = (box[0], box[1], box[2], box[3])  # x, y, w, h as a hashable tuple

            if box_key not in seen_boxes:
                unique_results.append(result)
                seen_boxes.add(box_key)

        # Sort by match score first, then by confidence
        unique_results.sort(key=lambda x: (x[4], x[2]), reverse=True)
        print(
            f"Found {len(unique_results)} matches for '{query}' with confidence >= {min_confidence:.2f}"
        )

        # Show match details for debugging
        for i, r in enumerate(unique_results):
            page_num, box, confidence, original_text, score = r
            print(
                f"Match {i+1}: '{original_text}' (Score: {score:.2f}, Conf: {confidence:.2f})"
            )

        # Return ALL unique results
        return [(r[0], r[1], r[2]) for r in unique_results]
    else:
        print(f"No matches found for '{query}' with confidence >= {min_confidence:.2f}")
        return []


def visualize_results(original_image, all_text_regions, search_results, query):
    """
    Draws bounding boxes for all text regions and search results on the original image.
    - all_text_regions: all detected text regions
    - search_results: specific search matches
    - query: the search term
    """
    output_image = original_image.copy()

    # Color for all detected text regions
    all_text_color = (0, 100, 0)  # Dark green
    all_text_thickness = 1

    # Color for search results
    highlight_colour = (0, 165, 255)  # Bright orange
    highlight_thickness = 2

    # First, highlight all text regions with lighter color
    print(f"Highlighting {len(all_text_regions)} text regions found by OCR...")
    for text, box, confidence in all_text_regions:
        x, y, w, h = box
        # Draw the rectangle for all text
        cv2.rectangle(
            output_image, (x, y), (x + w, y + h), all_text_color, all_text_thickness
        )

        # Add label with text and confidence for all regions
        label = f"'{text}' ({confidence:.2f})"
        # Add a background to text for better visibility
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            output_image, (x, y - text_h - 10), (x + text_w, y), (0, 0, 0), -1
        )
        cv2.putText(
            output_image,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),  # White text
            1,
        )

    # Then, highlight search results with brighter color
    if not search_results:
        print(f"No matches found for '{query}'.")
    else:
        print(f"Visualizing {len(search_results)} results for query '{query}'...")
        for page_num, box, confidence in search_results:
            x, y, w, h = box
            # Draw the rectangle for search results
            cv2.rectangle(
                output_image,
                (x, y),
                (x + w, y + h),
                highlight_colour,
                highlight_thickness,
            )
            # Add search query label (on top of the existing text label, but offset)
            label = f"MATCH: {query} ({confidence:.2f})"
            # Add a background to text for better visibility
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            cv2.rectangle(
                output_image,
                (x, y + h + 5),
                (x + text_w, y + h + text_h + 15),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                output_image,
                label,
                (x, y + h + text_h + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                highlight_colour,  # Match highlight color
                1,
            )

    return output_image


if __name__ == "__main__":
    original_img, preprocessed_img = preprocess_image(IMAGE_PATH)

    if original_img is not None and preprocessed_img is not None:
        # Save preprocessed image for debugging
        cv2.imwrite("preprocessed.png", preprocessed_img)

        # Perform OCR with EasyOCR (much better for handwritten text)
        ocr_results = perform_ocr(preprocessed_img)

        # Convert EasyOCR results to our format
        recognition_results = convert_to_recognition_results(ocr_results)

        # Index the results
        text_index = index_text(recognition_results)

        # Example search
        search_query = "what is this"
        print(f"\nSearching for: '{search_query}'")
        found_locations = search_text(search_query, text_index)

        # Visualize the results, highlighting all text regions and search matches
        final_image = visualize_results(
            original_img, recognition_results, found_locations, search_query
        )

        # Display or save the output
        output_path = "htr_output_with_highlights.png"
        cv2.imwrite(output_path, final_image)
        print(f"Output image saved to {output_path}")

        cv2.imshow(f"Results for '{search_query}'", final_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to preprocess image")
