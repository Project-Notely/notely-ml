#!/usr/bin/env python3
"""
Test script for the unstructured document segmentation service using img3.PNG
"""

import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.append("app")

from app.services.unstructured_segmentation.main import (
    segment_document_unstructured,
    UnstructuredSegmentationService
)


def test_img3_segmentation():
    """Test segmentation with img3.PNG"""
    print("🧪 Testing Unstructured Document Segmentation with img3.PNG\n")
    
    # Path to your image
    image_path = "data/notability.png"
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return False
    
    print(f"📄 Processing image: {image_path}")
    
    try:
        # Test basic segmentation
        print("🔍 Segmenting document...")
        elements, stats = segment_document_unstructured(
            image_path,
            strategy="hi_res",  # Use high-resolution for images
            extract_images=True
        )
        
        if not elements:
            print("❌ No elements found in the image")
            return False
            
        print(f"✅ Successfully segmented image!")
        print(f"   📊 Total elements: {stats['total_elements']}")
        print(f"   📄 Total pages: {stats['total_pages']}")
        print(f"   📝 Total text length: {stats['total_text_length']} characters")
        print(f"   🏷️  Element types found: {list(stats['element_types'].keys())}")
        
        # Show breakdown by element type
        print(f"\n📈 Element type breakdown:")
        for element_type, count in stats['element_types'].items():
            print(f"   - {element_type}: {count}")
        
        # Show first few elements with details
        print(f"\n📝 First {min(5, len(elements))} elements:")
        for i, elem in enumerate(elements[:5]):
            print(f"   {i+1}. Type: {elem.element_type}")
            if elem.text:
                text_preview = elem.text.replace('\n', ' ').strip()
                if len(text_preview) > 100:
                    text_preview = text_preview[:100] + "..."
                print(f"      Text: {text_preview}")
            else:
                print(f"      Text: [No text content]")
            print(f"      Page: {elem.page_number}")
            if elem.coordinates:
                print(f"      Coordinates: {elem.coordinates}")
            print()
        
        # Test export functionality
        print("💾 Testing data export...")
        service = UnstructuredSegmentationService()
        
        # Export as JSON
        json_output = "output/notability_segmentation.json"
        json_success = service.export_segmentation_data(elements, json_output, "json")
        if json_success:
            print(f"   ✅ JSON export successful: {json_output}")
        else:
            print(f"   ❌ JSON export failed")
        
        return True
        
    except Exception as e:
        print(f"❌ Segmentation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_service_methods():
    """Test service initialization and methods"""
    print("🔧 Testing service methods...")
    
    try:
        service = UnstructuredSegmentationService()
        
        # Test color mapping
        print(f"   🎨 Available element colors: {len(service.ELEMENT_COLORS)}")
        print(f"   🏷️  Element types supported: {list(service.ELEMENT_COLORS.keys())}")
        
        print("   ✅ Service initialization successful!")
        return True
        
    except Exception as e:
        print(f"   ❌ Service test failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Starting Unstructured Document Segmentation Test\n")
    
    # Test service methods first
    service_ok = test_service_methods()
    print()
    
    if service_ok:
        # Test with img3.PNG
        success = test_img3_segmentation()
        
        if success:
            print("\n🎉 All tests completed successfully!")
            print("\n📋 Next steps:")
            print("   1. Check the generated img3_segmentation.json file")
            print("   2. Try the REST API endpoints")
            print("   3. Test with other document types")
        else:
            print("\n💥 Some tests failed!")
    else:
        print("\n💥 Service initialization failed!") 