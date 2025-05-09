import cv2
import numpy as np
import re

import pytesseract

IMAGE_PATH = "app/services/image.png"


def preprocess_image(image_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load and preprocess image for HTR"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image at {image_path}")
            return None

        # convert to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # apply adaptive thresholding for binarization
        # adjust block size and C value based on image characteristics
        binary = cv2.adaptiveThreshold(
            gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # noise reduction
        binary = cv2.medianBlur(binary, 3)

        print("Preprocessing complete")
        return img, binary
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None, None


def segment_lines(binary_img: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Simplified line segmentation based on horizontal projection.
    Returns bounding boxes for potential text lines.
    NOTE: This is a basic approach and may fail on complex layouts.
    Cloud services often provide much better segmentation.
    """
    try:
        # find contours - might find letters or connected components
        contours, hierarchy = cv2.findContours(
            binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # basic approach: asusme contours roughly correspond to words/lines
        # more robust: use horizontal projection, or deep learning models
        bounding_boxes = []
        min_contour_area = 50  # filter out small noise contours
        for contour in contours:
            if cv2.contourArea(contour) > min_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                # simple heuristic: filter boxes that are too wide or too tall to be text lines/words
                # this needs turning based on expected text size
                aspect_ratio = w / float(h)
                if 0.1 < aspect_ratio < 20:  # example filter
                    bounding_boxes.append((x, y, w, h))

        bounding_boxes.sort(key=lambda box: (box[1], box[0]))

        # optional: merge nearby boxes that likely form a single line (more complex)

        print(
            f"Found {len(bounding_boxes)} potential text regions (simplified segmentation)."
        )
        return bounding_boxes
    except Exception as e:
        print(f"Error during segmentation: {e}")
        return []


def perform_htr(image_segment: np.ndarray, method: str = "pytesseract") -> str:
    """Takes an image segment (e.g., a line or words) and returns recognized text."""
    if method == "pytesseract":
        try:
            # configure Tesseract for handwritten text if possible (e.g., specific language models)
            # '--psm 7' assumes a single text line. Adjust as needed.
            custom_config = r"--oem 1 --psm 7"  # LSTM engine, treat as single line
            text = pytesseract.image_to_string(
                image_segment, config=custom_config, lang="eng"
            )  # Specify language
            # Basic cleaning
            text = re.sub(r"[\n\x0c]", "", text).strip()
            # Tesseract doesn't easily provide confidence scores per word/line without more work
            # You might use image_to_data for more detail including confidences
            # data = pytesseract.image_to_data(image_segment, output_type=pytesseract.Output.DICT, config=custom_config, lang='eng')
            # conf = calculate_average_confidence(data['conf']) # Implement this helper
            confidence = 0.8  # Placeholder confidence
            print(f"Tesseract recognized: '{text}'")
            return text, confidence
        except Exception as e:
            print(f"Error using Tesseract: {e}")
            return "", 0.0
    elif method == "google_vision":
        pass
        client = vision.ImageAnnotatorClient()
        content = cv2.imencode(".png", image_segment)[1].tobytes()
        image = vision.Image(content=content)
        response = client.document_text_detection(image=image)
        # Parse response to get text and bounding boxes (relative to the segment)
        # The cloud API often does segmentation and HTR in one go on the full page image
        text = response.full_text_annotation.text
        confidence = 0.9  # Cloud APIs usually provide confidence scores
        print(f"Cloud API recognized: '{text}' (Simulated)")
        return text, confidence
    else:
        raise ValueError(f"Unsupported HTR method: {method}")


def index_text(text_results):
    """
    Creates a simple in-memory index mapping words to their locations.
    A real app would use a db or search engine
    """
    index = {}  # word -> list of (page_num, bounding_box)
    page_num = 1  # assuming single page for now

    for text, box, confidence in text_results:
        # basic word tokenization (split by space)
        words = text.lower().split()
        for word in words:
            # clean the word (remove punctuation, etc.) - adjust as needed
            cleaned_word = re.sub(r"[^\w]", "", word)
            if cleaned_word:
                if cleaned_word not in index:
                    index[cleaned_word] = []

                # store the bounding box of the entire recognized segment for this word
                # more advanced: estimate word-level boxes if HTR provides them
                index[cleaned_word].append((page_num, box, confidence))
        print(f"Indexing complete. Indexed {len(index)} unique words.")
        return index


def search_text(query, index):
    """Searches the index for the query string (case-insensitive)."""
    query = query.lower()
    # basic search: exact match on cleaned words
    cleaned_query = re.sub(r"[^\w]", "", query)

    if cleaned_query in index:
        results = index[cleaned_query]
        print(f"Found {len(results)} matches for '{query}'.")
        # sort results by confidence (descending) if desired
        results.sort(key=lambda item: item[2], reverse=True)
        return results
    else:
        print(f"No matches found for '{query}'.")
        return []


def visualize_results(original_image, search_results, query):
    """Draws bounding boxes for search results on the original image."""
    output_image = original_image.copy()
    highlight_colour = (255, 0, 0)  # BGR for blue
    thickness = 2

    if not search_results:
        print(f"No matches found for '{query}'.")
        return output_image

    print(f"Visualizing {len(search_results)} results for query '{query}'...")
    for page_num, box, confidence in search_results:
        x, y, w, h = box
        # draw the rectangle
        cv2.rectangle(output_image, (x, y), (x + w, y + h), highlight_colour, thickness)
        # put text label (optional)
        label = f"{query} ({confidence:.2f})"
        cv2.putText(
            output_image,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            highlight_colour,
            1,
        )

    return output_image


if __name__ == "__main__":
    original_img, binary_img = preprocess_image(IMAGE_PATH)

    if original_img is not None and binary_img is not None:
        # perform segmentation
        # NOTE: For cloud APIs, you might send the whole preprocessed image
        # and get back text with boxes directly, skipping manual segmentation
        bounding_boxes = segment_lines(binary_img)

        # perform HTR on each segment
        recognition_results = []
        for box in bounding_boxes:
            x, y, w, h = box
            # extract the segment from the *binary* or *grayscale* image
            # using binary might be better for some HTR engines
            segment = binary_img[y : y + h, x : x + w]

            # pad the segment slightly if needed by the HTR engine
            segment = cv2.copyMakeBorder(
                segment, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0]
            )

            if segment.size > 0:  # ensure segment is not empty
                text, conf = perform_htr(segment, method="pytesseract")
                if text:  # only add if text was recognized
                    recognition_results.append((text, box, conf))
            else:
                print(f"Skipping empty segment at box {box}")

        # index the results
        text_index = index_text(recognition_results)

        # example search
        search_query = "bisection"
        print(f"\nSearching for: '{search_query}'")
        found_locations = search_text(search_query, text_index)

        # visualize the results
        final_image = visualize_results(original_img, found_locations, search_query)

        # display or save the output
        output_path = "htr_output_with_highlights.png"
        cv2.imwrite(output_path, final_image)
        print(f"Output image saved to {output_path}")

        cv2.imshow(f"Results for '{search_query}'", final_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to preprocess image")
