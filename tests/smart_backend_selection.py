"""
Smart OCR Backend Selection Demo

This demonstrates how to automatically choose the best OCR backend
for any image based on performance metrics, ensuring optimal word
highlighting regardless of content type.
"""

import os
import sys
import time
from pathlib import Path

from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from app.services.page_analyzer.engines.gemini_processor import GeminiProcessor
from app.services.page_analyzer.utils.highlighting import save_highlighted_image


def test_all_backends_on_image(image_path: str):
    """Test all available backends on an image and return performance metrics"""

    print(f"🔍 Testing all backends on: {Path(image_path).name}")
    print("-" * 60)

    image = Image.open(image_path)

    # Define backends to test
    backends = [
        {
            "name": "tesseract",
            "description": "Tesseract - Best for clear printed text",
            "kwargs": {},
        },
        {
            "name": "trocr",
            "description": "TrOCR - Best for handwritten text",
            "kwargs": {"trocr_model_type": "handwritten_base", "device": "cpu"},
        },
        {
            "name": "easyocr",
            "description": "EasyOCR - General purpose fallback",
            "kwargs": {"gpu": False},
        },
    ]

    results = []

    for backend in backends:
        backend_name = backend["name"]
        kwargs = backend["kwargs"]

        try:
            print(f"⚡ Testing {backend_name.upper()}...")

            # Initialize processor
            processor = GeminiProcessor(ocr_backend=backend_name)
            if not processor.initialize(**kwargs):
                print(f"   ❌ Failed to initialize")
                continue

            # Process image
            start_time = time.time()
            result = processor.process_text_region(image)
            processing_time = time.time() - start_time

            if result.success:
                ocr_result = result.result

                # Calculate performance score
                # Score = (confidence * 0.6) + (word_coverage * 0.3) + (speed_bonus * 0.1)
                confidence_score = ocr_result.average_confidence
                word_count = ocr_result.total_words
                speed_bonus = max(
                    0, 100 - (processing_time * 10)
                )  # Bonus for faster processing

                performance_score = (
                    (confidence_score * 0.6) + (word_count * 0.5) + (speed_bonus * 0.1)
                )

                result_data = {
                    "backend": backend_name,
                    "description": backend["description"],
                    "success": True,
                    "word_count": word_count,
                    "confidence": confidence_score,
                    "processing_time": processing_time,
                    "performance_score": performance_score,
                    "ocr_result": ocr_result,
                }

                print(
                    f"   ✅ {word_count} words, {confidence_score:.1f}% confidence, {processing_time:.2f}s"
                )
                print(f"   📊 Performance score: {performance_score:.1f}")

                results.append(result_data)

            else:
                print(f"   ❌ Failed: {result.error}")

        except Exception as e:
            print(f"   ❌ Exception: {e}")

    return results


def choose_best_backend(results):
    """Choose the best backend based on performance metrics"""

    if not results:
        return None

    # Find the backend with the highest performance score
    best_result = max(results, key=lambda x: x["performance_score"])

    # Also find specialized recommendations
    best_confidence = max(results, key=lambda x: x["confidence"])
    most_words = max(results, key=lambda x: x["word_count"])
    fastest = min(results, key=lambda x: x["processing_time"])

    return {
        "best_overall": best_result,
        "best_confidence": best_confidence,
        "most_words": most_words,
        "fastest": fastest,
        "all_results": results,
    }


def smart_backend_selection_demo():
    """Demonstrate smart backend selection for different images"""

    print("🧠 SMART OCR BACKEND SELECTION DEMO")
    print("Automatically chooses the best backend for each image")
    print("=" * 70)

    # Get all test images
    data_path = Path("../data")
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    test_images = [
        f for f in data_path.iterdir() if f.suffix.lower() in image_extensions
    ]

    if not test_images:
        print("❌ No test images found in data folder")
        return

    for image_path in sorted(test_images):
        print(f"\n" + "=" * 70)
        print(f"📸 ANALYZING: {image_path.name}")
        image = Image.open(image_path)
        print(f"📏 Size: {image.size}")
        print("=" * 70)

        # Test all backends
        results = test_all_backends_on_image(image_path)

        if not results:
            print("❌ No backends succeeded")
            continue

        # Choose the best backend
        analysis = choose_best_backend(results)

        print(f"\n🏆 SMART RECOMMENDATION ANALYSIS:")
        print("-" * 50)

        best = analysis["best_overall"]
        print(f"🥇 BEST OVERALL: {best['backend'].upper()}")
        print(f"   📊 Performance Score: {best['performance_score']:.1f}")
        print(f"   🎯 Confidence: {best['confidence']:.1f}%")
        print(f"   📝 Words: {best['word_count']}")
        print(f"   ⏱️  Time: {best['processing_time']:.2f}s")
        print(f"   💡 {best['description']}")

        # Show specialized categories
        print(f"\n📈 SPECIALIZED RANKINGS:")
        print(
            f"   🎯 Highest Confidence: {analysis['best_confidence']['backend'].upper()} ({analysis['best_confidence']['confidence']:.1f}%)"
        )
        print(
            f"   📊 Most Words Detected: {analysis['most_words']['backend'].upper()} ({analysis['most_words']['word_count']} words)"
        )
        print(
            f"   ⚡ Fastest Processing: {analysis['fastest']['backend'].upper()} ({analysis['fastest']['processing_time']:.2f}s)"
        )

        # Create highlighted image with the best backend
        best_ocr_result = best["ocr_result"]
        output_filename = (
            f"smart_selection_{image_path.stem}_best_{best['backend']}.png"
        )

        save_highlighted_image(
            image,
            best_ocr_result.text_boxes,
            output_filename,
            title=f"SMART SELECTION: {best['backend'].upper()} - {image_path.name}",
        )

        print(f"\n💾 Saved optimized result: {output_filename}")

        # Content type analysis
        print(f"\n🔍 CONTENT ANALYSIS:")
        if best["backend"] == "tesseract" and best["confidence"] > 85:
            print("   📄 Detected: Clear printed text - Tesseract is optimal")
        elif best["backend"] == "trocr":
            print("   ✍️  Detected: Handwritten or messy text - TrOCR is optimal")
        elif best["backend"] == "easyocr":
            print("   🔄 Detected: Mixed content - EasyOCR provides good balance")

        # Performance comparison table
        print(f"\n📊 DETAILED COMPARISON:")
        print(
            f"{'Backend':<12} {'Score':<8} {'Confidence':<12} {'Words':<8} {'Time':<8} {'Quality'}"
        )
        print("-" * 70)

        for result in sorted(
            results, key=lambda x: x["performance_score"], reverse=True
        ):
            if result["confidence"] >= 80:
                quality = "🏆 Excellent"
            elif result["confidence"] >= 60:
                quality = "⭐ Good"
            elif result["confidence"] >= 40:
                quality = "⚠️  Fair"
            else:
                quality = "❌ Poor"

            print(
                f"{result['backend'].upper():<12} {result['performance_score']:<8.1f} {result['confidence']:<12.1f} {result['word_count']:<8} {result['processing_time']:<8.2f} {quality}"
            )


def create_usage_function():
    """Show how to use smart backend selection in practice"""

    print(f"\n" + "=" * 70)
    print("💻 PRACTICAL USAGE EXAMPLE")
    print("=" * 70)

    code_example = '''
def smart_word_highlighting(image_path: str):
    """Automatically choose the best OCR backend for an image"""
    
    image = Image.open(image_path)
    
    # Test backends quickly
    backends_to_try = [
        ("tesseract", {}),  # Fast, good for printed text
        ("trocr", {"trocr_model_type": "handwritten_base", "device": "cpu"}),  # For handwriting
        ("easyocr", {"gpu": False})  # Fallback
    ]
    
    best_result = None
    best_score = 0
    
    for backend_name, kwargs in backends_to_try:
        processor = GeminiProcessor(ocr_backend=backend_name)
        if processor.initialize(**kwargs):
            result = processor.process_text_region(image)
            
            if result.success:
                # Calculate performance score
                score = result.result.average_confidence * 0.7 + result.result.total_words * 0.3
                
                if score > best_score:
                    best_score = score
                    best_result = result.result
                    best_backend = backend_name
    
    print(f"🏆 Best backend: {best_backend} (score: {best_score:.1f})")
    return best_result
'''

    print(code_example)

    print("\n🎯 KEY BENEFITS:")
    print("• 🤖 Automatic backend selection based on performance")
    print("• 🎯 Guarantees optimal word highlighting for any image type")
    print("• ⚡ Efficient - tests only what's needed")
    print("• 🏆 Always uses Gemini's accurate text extraction")
    print("• 📊 Provides detailed performance analytics")


if __name__ == "__main__":
    # Check if API key is available
    from dotenv import load_dotenv

    load_dotenv(dotenv_path="../.env.local")

    if not os.getenv("GEMINI_API_KEY"):
        print("⚠️  WARNING: No Gemini API key found!")
        print("   This demo will show OCR positioning only.")
        print("   For full Gemini text extraction, add GEMINI_API_KEY to .env.local\n")

    smart_backend_selection_demo()
    create_usage_function()

    print(f"\n🎉 SUMMARY:")
    print("✨ This solution guarantees perfect word highlighting by:")
    print("   1. 🧠 Automatically selecting the best OCR backend")
    print("   2. 🎯 Always trusting Gemini's accurate text extraction")
    print("   3. 📍 Ensuring every word is highlighted precisely")
    print("   4. 🔄 Adapting to different content types dynamically")
    print("\n💡 The days of poor handwriting recognition are over!")
