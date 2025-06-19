import sys
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from app.services.page_analyzer.engines.trocr_processor import TrOCRProcessor

def test_detect_lines_simple():
    """Visualize what _detect_lines_simple does"""
    
    # Load image
    image = cv2.imread("data/paragraph_potato.png")
    print(f"Image shape: {image.shape}")
    
    # Create processor and call the method
    processor = TrOCRProcessor()
    lines = processor._detect_lines_simple(image)
    
    print(f"Detected {len(lines)} lines")
    for i, (line_image, line_y, line_height) in enumerate(lines):
        print(f"  Line {i+1}: y={line_y}, height={line_height}")
    
    # Show original vs lines
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # With detected lines
    result = image.copy()
    for i, (line_image, line_y, line_height) in enumerate(lines):
        cv2.rectangle(result, (0, line_y), (image.shape[1], line_y + line_height), (255, 0, 0), 2)
        cv2.putText(result, f'{i+1}', (10, line_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    ax2.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    ax2.set_title(f'Detected {len(lines)} Lines')
    ax2.axis('off')
    
    plt.show()

if __name__ == "__main__":
    test_detect_lines_simple()
