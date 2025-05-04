import cv2
import numpy as np
import sys
import os
from paddleocr import PaddleOCR
from rapidfuzz import process, fuzz
import math

def highlight_matches_on_image(
    image_path: str,
    search_query: str,
    output_path: str,
    detection_threshold: float = 0.2,
    fuzzy_threshold: int = 30,
    segment_size: int = 300  # Default segment size in pixels
):
    """
    1. Detects and recognizes text in the input image using PaddleOCR.
    2. Performs a fuzzy match search for `search_query`.
    3. Draws bounding boxes on all text regions that approximately match the query.
    4. Saves the highlighted image to `output_path`.
    
    Required dependencies:
    - paddle: `pip install paddlepaddle`
    - paddleocr: `pip install paddleocr`
    - opencv-python: `pip install opencv-python`
    - rapidfuzz: `pip install rapidfuzz`
    
    :param image_path: Path to the input image (handwritten or printed).
    :param search_query: The text query to search for (e.g., "Hello").
    :param output_path: Where to save the highlighted image.
    :param detection_threshold: Confidence threshold for text detection boxes.
    :param fuzzy_threshold: Minimum fuzz ratio score (0-100) for approximate matches.
    :param segment_size: Size of image segments for improved OCR (in pixels).
    """
    # -------------
    # 1) Load and analyze the image
    # -------------
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not open {image_path}")
        return
    
    original_image = image.copy()
    height, width = image.shape[:2]
    print(f"Image dimensions: {width}x{height}")
    
    # -------------
    # 2) Preprocess the entire image
    # -------------
    # Create a copy for preprocessing
    preprocessed_img = image.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to enhance contrast
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Save preprocessed image temporarily
    preproc_path = "temp_preproc.jpg"
    cv2.imwrite(preproc_path, binary)
    
    # -------------
    # 3) Initialize OCR with better parameters for handwriting
    # -------------
    ocr = PaddleOCR(use_angle_cls=True, lang='en', 
                   det_db_box_thresh=0.3,
                   det_db_unclip_ratio=2.0)
    
    # -------------
    # 4) Process the image in segments for better OCR accuracy
    # -------------
    # Calculate the number of segments in each direction
    n_segments_x = max(1, math.ceil(width / segment_size))
    n_segments_y = max(1, math.ceil(height / segment_size))
    
    segment_width = width // n_segments_x
    segment_height = height // n_segments_y
    
    print(f"Dividing image into {n_segments_x}x{n_segments_y} segments")
    
    # Will store OCR results for each segment
    all_ocr_results = []
    
    # Process each segment
    for y in range(n_segments_y):
        for x in range(n_segments_x):
            # Calculate segment boundaries
            x1 = x * segment_width
            y1 = y * segment_height
            x2 = min(width, (x + 1) * segment_width)
            y2 = min(height, (y + 1) * segment_height)
            
            # Add some overlap between segments to avoid cutting text
            x1_overlap = max(0, x1 - 20)
            y1_overlap = max(0, y1 - 20)
            x2_overlap = min(width, x2 + 20)
            y2_overlap = min(height, y2 + 20)
            
            # Extract segment from original and preprocessed images
            segment_orig = original_image[y1_overlap:y2_overlap, x1_overlap:x2_overlap]
            segment_preproc = binary[y1_overlap:y2_overlap, x1_overlap:x2_overlap]
            
            # Skip empty segments
            if segment_orig.size == 0 or segment_preproc.size == 0:
                continue
                
            # Save segments temporarily
            segment_orig_path = f"temp_segment_orig_{x}_{y}.jpg"
            segment_preproc_path = f"temp_segment_preproc_{x}_{y}.jpg"
            
            cv2.imwrite(segment_orig_path, segment_orig)
            cv2.imwrite(segment_preproc_path, segment_preproc)
            
            # Process segments with OCR
            print(f"Processing segment ({x},{y}) of size {segment_orig.shape[1]}x{segment_orig.shape[0]}")
            
            # Run OCR on both versions of the segment
            try:
                results_orig = ocr.ocr(segment_orig_path, cls=True)
                results_preproc = ocr.ocr(segment_preproc_path, cls=True)
                
                # Process OCR results to adjust coordinates back to original image
                if results_orig[0]:
                    for detection in results_orig[0]:
                        coords, (text, conf) = detection
                        # Adjust coordinates relative to the original image
                        adjusted_coords = [[x + x1_overlap, y + y1_overlap] for x, y in coords]
                        all_ocr_results.append((adjusted_coords, (text, conf)))
                
                if results_preproc[0]:
                    for detection in results_preproc[0]:
                        coords, (text, conf) = detection
                        # Adjust coordinates relative to the original image
                        adjusted_coords = [[x + x1_overlap, y + y1_overlap] for x, y in coords]
                        all_ocr_results.append((adjusted_coords, (text, conf)))
            except Exception as e:
                print(f"Error processing segment ({x},{y}): {e}")
            
            # Clean up temporary files
            if os.path.exists(segment_orig_path):
                os.remove(segment_orig_path)
            if os.path.exists(segment_preproc_path):
                os.remove(segment_preproc_path)
    
    # Clean up main preprocessed image
    if os.path.exists(preproc_path):
        os.remove(preproc_path)
    
    # -------------
    # 5) Remove duplicates based on similar bounding box locations
    # -------------
    unique_results = []
    coords_seen = []
    
    for detection in all_ocr_results:
        coords, (text, conf) = detection
        # Skip empty or very short text
        if not text or len(text.strip()) < 2:
            continue
            
        coords_center = np.mean(coords, axis=0)
        
        # Check if we've already seen a detection at similar coordinates
        is_duplicate = False
        for i, prev_coords in enumerate(coords_seen):
            prev_center = np.mean(prev_coords, axis=0)
            distance = np.linalg.norm(coords_center - prev_center)
            
            if distance < 20:  # Arbitrary threshold for "same location"
                # If duplicate, keep the one with higher confidence
                prev_text, prev_conf = unique_results[i][1]
                if conf > prev_conf:
                    unique_results[i] = detection
                    coords_seen[i] = coords
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_results.append(detection)
            coords_seen.append(coords)
    
    # -------------
    # 6) Process recognized text regions
    # -------------
    recognized_entries = []  # Will hold tuples of (detected_text, bounding_pts)

    print("\n=== DETECTED TEXT ===")
    for detection in unique_results:
        # Each detection is ((quad_points), (text, text_confidence))
        coords, (detected_text, conf) = detection
        print(f"Text: '{detected_text}', Confidence: {conf:.2f}")
        
        if conf < detection_threshold:
            # skip low-confidence detection
            print(f"  [Skipped due to low confidence]")
            continue

        # Store recognized text and bounding coordinates
        recognized_entries.append((detected_text, coords))
    
    # -------------
    # 7) Fuzzy Search
    # -------------
    # Prepare two lists for fuzzy search
    all_texts = [entry[0] for entry in recognized_entries]  # all recognized strings
    
    # Use rapidfuzz.process.extract to compare `search_query` against each recognized text
    matches = process.extract(search_query, all_texts, scorer=fuzz.ratio)

    print("\n=== FUZZY MATCHES ===")
    print(f"Search query: '{search_query}'")
    
    # Build a set of indexes of lines that are "close enough"
    matched_indexes = set()
    for match_str, score, idx in matches:
        print(f"Match: '{match_str}', Score: {score}")
        if score >= fuzzy_threshold:
            matched_indexes.add(idx)
            print(f"  [MATCHED - Score >= {fuzzy_threshold}]")
    
    # -------------
    # 8) Highlight all matches
    # -------------
    # For each recognized text line that we found is a match, draw the bounding box.
    for i, (detected_text, coords) in enumerate(recognized_entries):
        if i not in matched_indexes:
            continue
        # coords is a list of four [x, y] points that define the quadrilateral
        # Convert to int for drawing
        pts = np.array(coords).astype(int)
        
        # Draw a filled polygon with semi-transparency
        overlay = image.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 0), lineType=cv2.LINE_AA)  # Green fill
        
        # Apply the overlay with transparency
        alpha = 0.4  # Transparency factor
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        # Add a thicker outline
        cv2.polylines(image, [pts], True, (0, 0, 255), 3)  # Red outline, thicker
        
        # Add match info
        match_text = f"{search_query} ({process.extractOne(search_query, [detected_text])[1]}%)"
        # Find position for text (above the bounding box)
        text_x = np.min(pts[:,0])
        text_y = np.min(pts[:,1]) - 10
        if text_y < 10:  # If too close to top
            text_y = np.max(pts[:,1]) + 20  # Put below the box
        
        # Add background for text
        text_size = cv2.getTextSize(match_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(image, 
                     (text_x - 2, text_y - text_size[1] - 2), 
                     (text_x + text_size[0] + 2, text_y + 2), 
                     (255, 255, 255), -1)
        # Add text
        cv2.putText(image, match_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # -------------
    # 9) If no matches found, show a message on the image
    # -------------
    if len(matched_indexes) == 0:
        print("No matches found for the search query.")
        # Add text to the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"No matches found for: '{search_query}'"
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        
        # Position at the top of the image
        text_x = (image.shape[1] - text_size[0]) // 2
        text_y = 50
        
        # Draw text with background for better visibility
        cv2.rectangle(image, (text_x - 10, text_y - 30), 
                     (text_x + text_size[0] + 10, text_y + 10), 
                     (0, 0, 0), -1)
        cv2.putText(image, text, (text_x, text_y), 
                   font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # -------------
    # 10) Save the result
    # -------------
    cv2.imwrite(output_path, image)
    print(f"\nHighlighted image saved to {output_path}")
    print(f"Found {len(matched_indexes)} matches for '{search_query}'")


def main():
    # Basic CLI usage: python script.py input_image.jpg "Hello" output.jpg
    if len(sys.argv) < 4:
        print("Usage: python script.py <image_path> <search_query> <output_path> [segment_size]")
        print("  segment_size: Optional size of image segments in pixels (default: 300)")
        sys.exit(1)
    
    image_path = sys.argv[1]
    search_query = sys.argv[2]
    output_path = sys.argv[3]
    
    # Optional segment size parameter
    segment_size = 300  # Default
    if len(sys.argv) >= 5:
        try:
            segment_size = int(sys.argv[4])
        except ValueError:
            print("Warning: Invalid segment size, using default of 300 pixels")
    
    highlight_matches_on_image(
        image_path=image_path,
        search_query=search_query,
        output_path=output_path,
        detection_threshold=0.2,
        fuzzy_threshold=30,
        segment_size=segment_size
    )

if __name__ == "__main__":
    main()
