import sys
from pathlib import Path
from PIL import Image
import cv2
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from app.services.page_analyzer.engines.gemini_processor import GeminiProcessor
from app.services.page_analyzer.utils.highlighting import (
    create_highlighted_image,
    save_highlighted_image,
    create_text_summary
)

def main():
    print("🔥 Gemini + Word Highlighting Test")
    print("=" * 50)

    # Initialize processor
    gemini_processor = GeminiProcessor()
    if not gemini_processor.initialize():
        print("❌ Failed to initialize processor")
        return

    # Load and process image
    image_path = "data/paragraph_potato.png"
    print(f"Processing image: {image_path}")
    
    image = Image.open(image_path)
    result = gemini_processor.process_text_region(image)
    
    if not result.success:
        print(f"❌ Processing failed: {result.error}")
        return

    ocr_result = result.result
    print(f"✅ SUCCESS! Extracted {len(ocr_result.text_boxes)} words")
    print(f"📝 Text: {ocr_result.full_text[:100]}...")
    print(f"🎯 Average Confidence: {sum(box.confidence for box in ocr_result.text_boxes) / len(ocr_result.text_boxes):.1f}%")

    # Convert PIL to numpy for highlighting
    image_np = np.array(image)
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    else:
        image_cv = image_np

    # Test 1: Highlight all words
    highlighted_basic = create_highlighted_image(image_cv, ocr_result.text_boxes)
    save_highlighted_image(highlighted_basic, "output/highlighted_basic.jpg")
    print("✅ Saved highlighted image to output/highlighted_basic.jpg")

    # Test 2: Highlight specific words with different colors for verification
    test_words = ["potato", "Chile", "studies", "cultivated", "origin"]
    print(f"\n🎯 Testing specific word highlighting: {test_words}")
    
    # Create highlighted image with specific words in red
    highlighted_specific = image_cv.copy()
    found_words = []
    
    for box in ocr_result.text_boxes:
        if box.text in test_words:
            x, y, w, h = box.bbox
            # Draw red rectangle for test words
            cv2.rectangle(highlighted_specific, (x, y), (x + w, y + h), (0, 0, 255), 3)
            # Add text label
            cv2.putText(highlighted_specific, f"{box.text}", (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            found_words.append(box.text)
            print(f"  Found '{box.text}' at [{x}, {y}, {w}, {h}]")
    
    save_highlighted_image(highlighted_specific, "output/highlighted_specific_words.jpg")
    print(f"✅ Found {len(found_words)}/{len(test_words)} test words")
    print("✅ Saved specific word highlighting to output/highlighted_specific_words.jpg")

    # Save text summary
    summary = create_text_summary(ocr_result)
    with open("output/text_summary.txt", "w") as f:
        f.write(summary)
    print("✅ Saved text summary to output/text_summary.txt")

    print("\n" + "=" * 50)
    print("🎉 All done! Check the output folder for results.")

    # Show first few detected words for debugging
    print(f"\n📋 First 5 detected words:")
    for i, box in enumerate(ocr_result.text_boxes[:5]):
        x, y, w, h = box.bbox
        print(f"  {i+1}. '{box.text}' at ({x}, {y}) - {box.confidence:.1f}%")

if __name__ == "__main__":
    main()