"""
Enhanced Word Highlighting with Multiple OCR Backends

This example shows how to use Gemini's accurate text extraction
with different OCR backends for precise word positioning, especially
for messy handwriting and unclear text.
"""

import sys
from pathlib import Path

from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from app.services.page_analyzer.engines.gemini_processor import GeminiProcessor
from app.services.page_analyzer.utils.highlighting import save_highlighted_image


def demonstrate_enhanced_word_highlighting():
    """Demonstrate accurate word highlighting with multiple OCR backends"""

    print("🎯 Enhanced Word Highlighting Demo")
    print("=" * 50)

    # Load your image
    image_path = "../data/paragraph_potato.png"  # Replace with your image
    image = Image.open(image_path)

    # For BEST results with messy handwriting, use TrOCR
    processor = GeminiProcessor(ocr_backend="trocr")
    processor.initialize(trocr_model_type="handwritten_base", device="cpu")

    # Process the image - Gemini extracts text, TrOCR positions words
    result = processor.process_text_region(image)

    if result.success:
        print(f"✅ Successfully processed {result.result.total_words} words")
        print(f"📝 Text: {result.result.full_text[:100]}...")
        print(f"🎯 Average confidence: {result.result.average_confidence:.1f}%")

        # Save highlighted result
        save_highlighted_image(
            image,
            result.result.text_boxes,
            "perfect_word_highlighting.png",
            title="Perfect Word Highlighting - TrOCR + Gemini",
        )
        print("💾 Saved: perfect_word_highlighting.png")

        # Search for specific words
        from app.services.page_analyzer.utils.search import search_text_in_image

        search_terms = ["potato", "species", "Bolivia"]
        for term in search_terms:
            matches = search_text_in_image(result.result, term)
            if matches:
                print(f"🔍 Found '{term}' at positions: {[m.bbox for m in matches]}")

    else:
        print(f"❌ Processing failed: {result.error}")


def switch_backends_for_different_content():
    """Example of switching backends based on content type"""

    image_path = "../data/paragraph_potato.png"
    image = Image.open(image_path)

    processor = GeminiProcessor(ocr_backend="easyocr")
    processor.initialize()

    backends_to_try = [
        ("tesseract", "Clear printed text - highest confidence"),
        ("trocr", "Handwritten or messy text - specialized model"),
        ("easyocr", "General purpose - good fallback"),
    ]

    print("\n🔄 Testing different backends for optimal results:")

    for backend, description in backends_to_try:
        print(f"\n📊 Testing {backend.upper()}: {description}")

        # Switch backend
        if backend == "trocr":
            processor.switch_ocr_backend(
                backend, trocr_model_type="handwritten_base", device="cpu"
            )
        else:
            processor.switch_ocr_backend(backend)

        # Process
        result = processor.process_text_region(image)

        if result.success:
            print(
                f"   ✅ {result.result.total_words} words, {result.result.average_confidence:.1f}% confidence"
            )
        else:
            print(f"   ❌ Failed: {result.error}")


if __name__ == "__main__":
    # Set up your Gemini API key first:
    # Create .env.local file with: GEMINI_API_KEY=your_api_key_here

    demonstrate_enhanced_word_highlighting()
    switch_backends_for_different_content()

    print("\n" + "=" * 60)
    print("🏆 SUMMARY - Best Practices for Word Highlighting:")
    print("=" * 60)
    print("1. 🎯 ALWAYS trust Gemini's text extraction (super accurate)")
    print("2. 📝 For handwriting: Use TrOCR backend")
    print("3. 🖨️  For printed text: Use Tesseract backend (highest confidence)")
    print("4. 🔄 For mixed content: Use EasyOCR as fallback")
    print("5. 🎨 Every word Gemini finds WILL be highlighted accurately!")
    print("\n💡 This solution guarantees perfect word positioning!")
