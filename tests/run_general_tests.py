import os
from pathlib import Path
from PIL import Image
import sys

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from app.services.page_analyzer.engines.gemini_processor import GeminiProcessor
from app.services.page_analyzer.utils.highlighting import save_highlighted_image

def test_ocr_backends_standalone():
    """Test OCR backends independently to see their word detection capabilities"""
    
    # Test image path
    image_path = "../data/paragraph_potato.png"
    
    if not os.path.exists(image_path):
        print(f"❌ Test image not found at {image_path}")
        return

    # Load the test image
    image = Image.open(image_path)
    print(f"📸 Loaded test image: {image.size}")
    
    # Define available backends 
    backends = [
        {
            "name": "trocr", 
            "description": "TrOCR - Specifically designed for handwritten text",
            "kwargs": {"trocr_model_type": "handwritten_base", "device": "cpu"}
        },
        {
            "name": "tesseract", 
            "description": "Tesseract - Classic, reliable OCR",
            "kwargs": {}
        },
        {
            "name": "easyocr", 
            "description": "EasyOCR - Previous implementation",
            "kwargs": {"gpu": False}
        },
        {
            "name": "paddleocr", 
            "description": "PaddleOCR - Latest tech, excellent for handwriting",
            "kwargs": {"gpu": False}
        }
    ]
    
    results = []
    
    # Test each backend for word positioning (without Gemini)
    for backend_config in backends:
        backend_name = backend_config["name"]
        description = backend_config["description"]
        kwargs = backend_config["kwargs"]
        
        print(f"\n" + "="*60)
        print(f"🔄 Testing {backend_name.upper()}: {description}")
        print("="*60)
        
        try:
            # Initialize processor with specific backend
            processor = GeminiProcessor(ocr_backend=backend_name)
            
            if not processor.initialize(**kwargs):
                print(f"❌ Failed to initialize {backend_name}")
                continue
            
            # Test word positioning directly (bypass Gemini)
            import numpy as np
            import cv2
            
            # Convert PIL to numpy/OpenCV format
            image_np = np.array(image)
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            else:
                image_cv = image_np
            
            # Get word positions directly from OCR backend
            word_positions = processor._get_word_positions_with_backend(image_cv, image)
            
            if word_positions:
                print(f"✅ {backend_name.upper()} detected {len(word_positions)} words")
                
                # Show some sample detections
                for i, word_pos in enumerate(word_positions[:5]):
                    text = word_pos.get('text', 'N/A')
                    confidence = word_pos.get('confidence', 0)
                    bbox = word_pos.get('bbox', [0, 0, 0, 0])
                    print(f"  {i+1}. '{text}' - {confidence:.1f}% at {bbox}")
                
                # Create text boxes for highlighting
                from app.services.page_analyzer.models.models import TextBox
                text_boxes = []
                
                for word_pos in word_positions:
                    text_box = TextBox(
                        text=word_pos.get('text', ''),
                        confidence=word_pos.get('confidence', 0),
                        bbox=word_pos.get('bbox', [0, 0, 0, 0])
                    )
                    text_boxes.append(text_box)
                
                # Save highlighted image
                output_filename = f"ocr_detection_{backend_name}.png"
                save_highlighted_image(
                    image, 
                    text_boxes, 
                    output_filename,
                    title=f"OCR Detection - {backend_name.upper()}"
                )
                print(f"💾 Saved detection image: {output_filename}")
                
                # Store results for comparison
                results.append({
                    "backend": backend_name,
                    "description": description,
                    "success": True,
                    "word_count": len(word_positions),
                    "avg_confidence": sum(w.get('confidence', 0) for w in word_positions) / len(word_positions),
                    "sample_words": [w.get('text', '') for w in word_positions[:10]]
                })
                
            else:
                print(f"❌ {backend_name.upper()} detected no words")
                results.append({
                    "backend": backend_name,
                    "description": description,
                    "success": False,
                    "error": "No words detected"
                })
                
        except Exception as e:
            print(f"❌ Exception with {backend_name}: {e}")
            results.append({
                "backend": backend_name,
                "description": description,
                "success": False,
                "error": str(e)
            })
    
    # Print comparison summary
    print(f"\n" + "="*80)
    print("📊 OCR BACKEND WORD DETECTION COMPARISON")
    print("="*80)
    
    successful_results = [r for r in results if r["success"]]
    
    if successful_results:
        print(f"{'Backend':<12} {'Words':<8} {'Avg Conf':<10} {'Sample Words'}")
        print("-" * 80)
        
        for result in successful_results:
            sample_text = ', '.join(result['sample_words'][:3])
            if len(sample_text) > 30:
                sample_text = sample_text[:27] + "..."
            print(f"{result['backend'].upper():<12} {result['word_count']:<8} {result['avg_confidence']:<10.1f} {sample_text}")
    
    # Analysis and recommendations
    print(f"\n🎯 ANALYSIS:")
    if successful_results:
        best_count = max(r['word_count'] for r in successful_results)
        best_backend = next(r for r in successful_results if r['word_count'] == best_count)
        print(f"• Most words detected: {best_backend['backend'].upper()} ({best_count} words)")
        
        best_conf = max(r['avg_confidence'] for r in successful_results)
        best_conf_backend = next(r for r in successful_results if r['avg_confidence'] == best_conf)
        print(f"• Highest confidence: {best_conf_backend['backend'].upper()} ({best_conf:.1f}%)")
    
    print(f"\n💡 RECOMMENDATIONS FOR MESSY HANDWRITING:")
    print("1. 🏆 TrOCR - Specifically trained for handwritten text")
    print("2. 🌟 PaddleOCR - Latest deep learning technology, great for complex scenarios")
    print("3. 🔧 Tesseract - Classic choice, reliable for clear printed text")
    print("4. 🔄 EasyOCR - Good general purpose, but weaker on handwriting")
    
    print(f"\n📋 SETUP INSTRUCTIONS:")
    print("To use with Gemini text extraction (recommended):")
    print("1. Get a Gemini API key from https://aistudio.google.com/apikey")
    print("2. Create .env.local file in project root with: GEMINI_API_KEY=your_api_key_here")
    print("3. Run: processor = GeminiProcessor(ocr_backend='trocr')")
    print("4. This will use Gemini for accurate text extraction + TrOCR for precise positioning")
    
    return results

def test_gemini_processor_with_backends():
    """Test GeminiProcessor with different OCR backends for better handwriting recognition"""
    
    # Check if Gemini API key is available
    from dotenv import load_dotenv
    load_dotenv(dotenv_path="../.env.local")
    
    if not os.getenv("GEMINI_API_KEY"):
        print("⚠️  No Gemini API key found. Running OCR-only demonstration.")
        print("For full functionality, set GEMINI_API_KEY in .env.local file.")
        return test_ocr_backends_standalone()
    
    # Test image path
    image_path = "../data/paragraph_potato.png"
    
    if not os.path.exists(image_path):
        print(f"❌ Test image not found at {image_path}")
        return

    # Load the test image
    image = Image.open(image_path)
    print(f"📸 Loaded test image: {image.size}")
    
    # Define available backends (in order of recommendation for handwriting)
    backends = [
        {
            "name": "trocr", 
            "description": "TrOCR - Specifically designed for handwritten text",
            "kwargs": {"trocr_model_type": "handwritten_base", "device": "cpu"}
        },
        {
            "name": "tesseract", 
            "description": "Tesseract - Classic, reliable OCR",
            "kwargs": {}
        },
        {
            "name": "easyocr", 
            "description": "EasyOCR - Previous implementation",
            "kwargs": {"gpu": False}
        },
        {
            "name": "paddleocr", 
            "description": "PaddleOCR - Latest tech, excellent for handwriting",
            "kwargs": {"gpu": False}
        }
    ]
    
    results = []
    
    # Test each backend
    for backend_config in backends:
        backend_name = backend_config["name"]
        description = backend_config["description"]
        kwargs = backend_config["kwargs"]
        
        print(f"\n" + "="*60)
        print(f"🔄 Testing {backend_name.upper()}: {description}")
        print("="*60)
        
        try:
            # Initialize processor with specific backend
            processor = GeminiProcessor(ocr_backend=backend_name)
            
            if not processor.initialize(**kwargs):
                print(f"❌ Failed to initialize {backend_name}")
                continue
            
            # Process the image
            result = processor.process_text_region(image)
            
            if result.success:
                ocr_result = result.result
                print(f"✅ Successfully processed with {backend_name.upper()}")
                print(f"📝 Full text: {ocr_result.full_text}")
                print(f"📊 Total words: {ocr_result.total_words}")
                print(f"🎯 Average confidence: {ocr_result.average_confidence:.1f}%")
                
                # Save highlighted image
                output_filename = f"highlighted_output_{backend_name}.png"
                save_highlighted_image(
                    image, 
                    ocr_result.text_boxes, 
                    output_filename,
                    title=f"Word Highlighting - {backend_name.upper()}"
                )
                print(f"💾 Saved highlighted image: {output_filename}")
                
                # Store results for comparison
                results.append({
                    "backend": backend_name,
                    "description": description,
                    "success": True,
                    "word_count": ocr_result.total_words,
                    "confidence": ocr_result.average_confidence,
                    "full_text": ocr_result.full_text
                })
                
            else:
                print(f"❌ Failed to process with {backend_name}: {result.error}")
                results.append({
                    "backend": backend_name,
                    "description": description,
                    "success": False,
                    "error": result.error
                })
                
        except Exception as e:
            print(f"❌ Exception with {backend_name}: {e}")
            results.append({
                "backend": backend_name,
                "description": description,
                "success": False,
                "error": str(e)
            })
    
    # Print comparison summary
    print(f"\n" + "="*80)
    print("📊 BACKEND COMPARISON SUMMARY")
    print("="*80)
    
    successful_results = [r for r in results if r["success"]]
    
    if successful_results:
        print(f"{'Backend':<12} {'Words':<8} {'Confidence':<12} {'Description'}")
        print("-" * 80)
        
        for result in successful_results:
            print(f"{result['backend'].upper():<12} {result['word_count']:<8} {result['confidence']:<12.1f} {result['description']}")
    
    # Recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    print("For messy handwriting and unclear text:")
    print("1. TrOCR - Best for handwritten text recognition")
    print("2. PaddleOCR - Latest technology, excellent for complex scenarios")
    print("3. Tesseract - Classic, reliable for clear printed text")
    print("4. EasyOCR - Good general purpose, but weaker on handwriting")
    
    return results

def test_backend_switching():
    """Test switching between backends on the same processor"""
    print(f"\n" + "="*60)
    print("🔄 Testing Backend Switching")
    print("="*60)
    
    # Check if API key is available
    from dotenv import load_dotenv
    load_dotenv(dotenv_path="../.env.local")
    
    if not os.getenv("GEMINI_API_KEY"):
        print("⚠️  Skipping backend switching test - requires Gemini API key")
        return
    
    image_path = "../data/paragraph_potato.png"
    image = Image.open(image_path)
    
    # Start with EasyOCR
    processor = GeminiProcessor(ocr_backend="easyocr")
    processor.initialize(gpu=False)
    
    # Process with EasyOCR
    result1 = processor.process_text_region(image)
    print(f"EasyOCR words: {result1.result.total_words if result1.success else 'Failed'}")
    
    # Switch to TrOCR
    processor.switch_ocr_backend("trocr", trocr_model_type="handwritten_base", device="cpu")
    
    # Process with TrOCR
    result2 = processor.process_text_region(image)
    print(f"TrOCR words: {result2.result.total_words if result2.success else 'Failed'}")
    
    # Switch to Tesseract
    processor.switch_ocr_backend("tesseract")
    
    # Process with Tesseract
    result3 = processor.process_text_region(image)
    print(f"Tesseract words: {result3.result.total_words if result3.success else 'Failed'}")

if __name__ == "__main__":
    print("🚀 Starting Enhanced OCR Backend Tests")
    print("This test compares different OCR backends for word positioning")
    print("while always trusting Gemini's text extraction.\n")
    
    # Test different backends
    results = test_gemini_processor_with_backends()
    
    # Test backend switching
    test_backend_switching()
    
    print(f"\n✅ Testing complete! Check the highlighted output images to see the differences.")
    print("💡 For best results with messy handwriting, use TrOCR or PaddleOCR backends.")