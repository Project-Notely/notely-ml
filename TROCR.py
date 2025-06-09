#!/usr/bin/env python3
"""
TrOCR - Simple OCR processor for text reading and word highlighting
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image
import torch
from transformers import TrOCRProcessor as HFTrOCRProcessor, VisionEncoderDecoderModel
from app.services.page_analyzer.models.models import TextBox, OCRResult, ProcessingResult


def get_highlight_color(confidence: float) -> Tuple[int, int, int]:
    """Get highlight color based on confidence level (BGR format)"""
    if confidence >= 80:
        return (0, 255, 0)      # Green - high confidence
    elif confidence >= 60:
        return (0, 255, 255)    # Yellow - medium confidence  
    elif confidence >= 40:
        return (0, 165, 255)    # Orange - low confidence
    else:
        return (0, 0, 255)      # Red - very low confidence

def create_highlighted_image(
    image: np.ndarray, 
    text_boxes: List[TextBox], 
    output_path: Path,
    alpha: float = 0.3
) -> bool:
    """Create highlighted image with individual word boxes"""
    try:
        # Create copy of image
        highlighted = image.copy()
        overlay = image.copy()
        
        # Draw highlights for each word
        for text_box in text_boxes:
            x, y, w, h = text_box.bbox
            color = get_highlight_color(text_box.confidence)
            
            # Draw filled rectangle on overlay
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
            
            # Draw border
            cv2.rectangle(highlighted, (x, y), (x + w, y + h), color, 2)
            
            # Add text label (smaller font for individual words)
            label = f"{text_box.text}"
            font_scale = 0.4
            font_thickness = 1
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            
            # Background for text
            cv2.rectangle(highlighted, (x, y - 15), (x + label_size[0], y), color, -1)
            cv2.putText(highlighted, label, (x, y - 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        
        # Blend overlay with original
        cv2.addWeighted(overlay, alpha, highlighted, 1 - alpha, 0, highlighted)
        
        # Save image
        success = cv2.imwrite(str(output_path), highlighted)
        return success
        
    except Exception as e:
        print(f"Error creating highlighted image: {e}")
        return False

def search_text_in_image(text_boxes: List[TextBox], search_term: str) -> List[TextBox]:
    """Search for a term in the detected text boxes (ctrl-f functionality)"""
    search_term = search_term.lower()
    matching_boxes = []
    
    for text_box in text_boxes:
        if search_term in text_box.text.lower():
            matching_boxes.append(text_box)
    
    return matching_boxes

def create_search_highlighted_image(
    image: np.ndarray,
    text_boxes: List[TextBox],
    search_term: str,
    output_path: Path
) -> bool:
    """Create image with search results highlighted"""
    matching_boxes = search_text_in_image(text_boxes, search_term)
    
    if not matching_boxes:
        print(f"No matches found for '{search_term}'")
        return False
    
    try:
        highlighted = image.copy()
        
        # Highlight all text boxes in light gray
        for text_box in text_boxes:
            x, y, w, h = text_box.bbox
            cv2.rectangle(highlighted, (x, y), (x + w, y + h), (200, 200, 200), 1)
        
        # Highlight matching boxes in bright yellow
        for text_box in matching_boxes:
            x, y, w, h = text_box.bbox
            cv2.rectangle(highlighted, (x, y), (x + w, y + h), (0, 255, 255), 3)
            
            # Add search indicator
            cv2.putText(highlighted, "MATCH", (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        success = cv2.imwrite(str(output_path), highlighted)
        print(f"Found {len(matching_boxes)} matches for '{search_term}'")
        return success
        
    except Exception as e:
        print(f"Error creating search highlighted image: {e}")
        return False 