from .config import IMAGE_PATH, ALL_TEXT_COLOUR, HIGHLIGHT_COLOUR
from .recognition import perform_ocr, convert_to_recognition_results
from .indexing import index_text, search_text
from services.common import preprocess_image, draw_bounding_boxes, save_output_image


def visualize_htr_results(original_image, all_text_regions, search_results, query):
    """HTR-specific visualization that extends the common visualization"""
    # draw all text regions
    output_image = draw_bounding_boxes(
        original_image, all_text_regions, colour=ALL_TEXT_COLOUR, thickness=1
    )

    # highlight search results if any
    if search_results:
        search_regions = []
        for page_num, box, confidence in search_results:
            # create label that includes the search query
            label = f"MATCH: {query}"
            search_regions.append((label, box, confidence))

        output_image = draw_bounding_boxes(
            output_image,
            search_regions,
            colour=HIGHLIGHT_COLOUR,
            thickness=2,
            add_labels=True,
            text_colour=(0, 0, 0),
            text_bg_colour=(255, 255, 255),
        )

    return output_image


def process_handwritten_text(image_path=IMAGE_PATH, search_query=None):
    """Run the HTR pipeline on an image and optionally search for text"""
    # Preprocess the image
    original_img, preprocessed_img = preprocess_image(
        image_path, use_clahe=True, denoise=True
    )

    if original_img is None or preprocessed_img is None:
        return None, [], []

    # OCR processing
    ocr_results = perform_ocr(preprocessed_img)
    recognition_results = convert_to_recognition_results(ocr_results)

    # Index the text
    text_index = index_text(recognition_results)

    # Search if query provided
    search_results = []
    if search_query:
        print(f"Searching for: '{search_query}'")
        search_results = search_text(search_query, text_index)

    return original_img, recognition_results, search_results


def run_htr_demo(image_path=IMAGE_PATH, search_query="test"):
    """Run a demonstration of the HTR pipeline"""
    # Process the image
    original_img, recognition_results, search_results = process_handwritten_text(
        image_path, search_query
    )

    if original_img is None:
        print("Failed to process image")
        return []

    # Visualize the results
    final_image = visualize_htr_results(
        original_img, recognition_results, search_results, search_query
    )

    # Save and display the output
    output_path = "htr_output.png"
    save_output_image(final_image, output_path, show=True)

    return search_results


if __name__ == "__main__":
    run_htr_demo()
