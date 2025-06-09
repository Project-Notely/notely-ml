#!/usr/bin/env python3
"""
Test TrOCR with real image and search functionality
"""

import cv2
import json
from pathlib import Path
from TROCR import (
    create_highlighted_image,
    create_search_highlighted_image,
)
from app.services.page_analyzer.engines.trocr_processor import TrOCRProcessor


def main():
    print("🔥 TrOCR Real Image Test with Search")
    print("=" * 50)

    # Setup paths
    data_dir = Path("data")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Clear old results
    print("🧹 Clearing old results...")
    for f in output_dir.glob("*"):
        if f.is_file():
            f.unlink()

    # Load the real image
    image_path = data_dir / "paragraph_potato.png"
    if not image_path.exists():
        print(f"❌ ERROR: Image not found at {image_path}")
        return

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ ERROR: Failed to load image from {image_path}")
        return

    print(f"📸 Loaded image: {image_path}")
    print(f"   📏 Size: {image.shape}")

    # Test the best performing model for text reading
    print(f"\n{'=' * 50}")
    print(f"🔄 Processing with TrOCR (printed_base)")
    print(f"{'=' * 50}")

    # Initialize processor with printed model (better for paragraph text)
    processor = TrOCRProcessor(model_type="printed_base", device="cpu")

    if not processor.initialize():
        print(f"❌ Failed to initialize TrOCR")
        return

    # Process the image
    print("⚡ Processing image...")
    result = processor.process_text_region(image)

    if not result.success:
        print(f"❌ Processing failed: {result.error}")
        processor.cleanup()
        return

    # Extract results
    confidence = result.result.average_confidence
    text = result.result.full_text
    word_count = result.result.total_words

    print(f"✅ SUCCESS!")
    print(f"   🎯 Average Confidence: {confidence:.1f}%")
    print(f"   📝 Full Text: '{text}'")
    print(f"   📊 Total Words: {word_count}")
    print(f"   📦 Individual Word Boxes: {len(result.result.text_boxes)}")

    # Show each detected word
    print(f"\n📝 Detected Words:")
    for i, text_box in enumerate(result.result.text_boxes):
        print(
            f"   {i+1:2d}. '{text_box.text}' (confidence: {text_box.confidence:.1f}%)"
        )

    # Save detailed results to JSON
    results_file = output_dir / "word_detection_results.json"
    print(f"\n💾 Saving results: {results_file}")

    with open(results_file, "w") as f:
        json.dump(
            {
                "full_text": text,
                "average_confidence": confidence,
                "total_words": word_count,
                "individual_words": [
                    {"word": tb.text, "confidence": tb.confidence, "bbox": tb.bbox}
                    for tb in result.result.text_boxes
                ],
            },
            f,
            indent=2,
        )

    # Create highlighted image with all words
    highlighted_path = output_dir / "all_words_highlighted.png"
    print(f"🎨 Creating word-highlighted image: {highlighted_path}")

    success = create_highlighted_image(
        image, result.result.text_boxes, highlighted_path
    )

    if success:
        print(f"✅ Word-highlighted image saved!")
    else:
        print(f"❌ Failed to create highlighted image")

    # Test search functionality
    print(f"\n🔍 Testing Search Functionality (Ctrl-F style)")
    print("=" * 50)

    # Test searches for common words
    search_terms = [
        "the",
        "and",
        "in",
        "of",
        "potato",
    ]  # Common words that might appear

    for search_term in search_terms:
        print(f"\n🔎 Searching for: '{search_term}'")

        # Create search highlighted image
        search_output_path = output_dir / f"search_{search_term}.png"
        success = create_search_highlighted_image(
            image, result.result.text_boxes, search_term, search_output_path
        )

        if success:
            print(f"   ✅ Search results saved to: {search_output_path}")
        else:
            print(f"   ❌ No matches found or error occurred")

    # Interactive search (if running interactively)
    try:
        print(f"\n🎮 Interactive Search (press Enter to skip):")
        user_search = input("Enter a word to search for: ").strip()

        if user_search:
            search_output_path = output_dir / f"search_user_{user_search}.png"
            success = create_search_highlighted_image(
                image, result.result.text_boxes, user_search, search_output_path
            )

            if success:
                print(
                    f"✅ Search results for '{user_search}' saved to: {search_output_path}"
                )
            else:
                print(f"❌ No matches found for '{user_search}'")
    except:
        print("   (Skipping interactive search)")

    # Show summary
    print(f"\n📁 FILES CREATED IN {output_dir}:")
    for f in sorted(output_dir.glob("*")):
        size_kb = f.stat().st_size / 1024
        print(f"   📄 {f.name} ({size_kb:.1f} KB)")

    # Cleanup
    processor.cleanup()
    print(f"🧹 Cleaned up resources")

    print(f"\n✅ DONE! Check the highlighted images to see:")
    print(f"   🎨 all_words_highlighted.png - All detected words with boxes")
    print(f"   🔍 search_*.png - Search results for specific words")
    print(f"   📊 word_detection_results.json - Complete detection data")


if __name__ == "__main__":
    main()
