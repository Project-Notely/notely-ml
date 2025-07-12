"""
Demonstration script for running segmentation service on real data
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from PIL import Image, ImageDraw, ImageFont

from app.models.segmentation_models import DocumentSegment
from app.services.segmentation_service import SegmentationService


class SegmentationDemo:
    """Demo class for running segmentation on real data"""

    def __init__(self):
        self.service = SegmentationService()
        self.data_dir = Path(__file__).parent.parent / "data"
        self.output_dir = Path(__file__).parent.parent / "output" / "segmentation_demo"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def run_demo(self):
        """Run the complete demonstration"""
        print("🚀 Starting Segmentation Service Demo")
        print("=" * 60)

        # Find all images in data directory
        image_files = self.find_images()

        if not image_files:
            print("❌ No images found in data directory")
            return

        print(f"📁 Found {len(image_files)} images to process:")
        for img_file in image_files:
            print(f"   • {img_file.name}")
        print()

        # Process each image
        results = []
        for img_file in image_files:
            print(f"🔍 Processing: {img_file.name}")
            result = await self.process_image(img_file)
            results.append(
                {
                    "filename": img_file.name,
                    "result": result,
                    "file_path": str(img_file),
                }
            )
            print()

        # Generate summary report
        self.generate_summary_report(results)

        # Create visualizations
        await self.create_visualizations(results)

        print("✅ Demo completed! Check the output directory for results.")
        print(f"📂 Results saved to: {self.output_dir}")

    def find_images(self) -> list[Path]:
        """Find all image files in the data directory"""
        image_extensions = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
        image_files = []

        for file_path in self.data_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_files.append(file_path)

        return sorted(image_files)

    async def process_image(self, image_path: Path) -> dict[str, Any]:
        """Process a single image and return detailed results"""
        print(f"   📊 Analyzing {image_path.name}...")

        start_time = time.time()

        try:
            # Run segmentation
            result = await self.service.segment_image(image_path)

            end_time = time.time()
            processing_time = end_time - start_time

            # Get image info
            with Image.open(image_path) as img:
                image_info = {
                    "width": img.width,
                    "height": img.height,
                    "format": img.format,
                    "mode": img.mode,
                }

            # Compile detailed results
            detailed_result = {
                "success": result.success,
                "processing_time": processing_time,
                "image_info": image_info,
                "segments_found": len(result.segments),
                "segments": [self.segment_to_dict(seg) for seg in result.segments],
                "statistics": result.statistics,
                "strategy_used": result.strategy_used,
                "error": result.error,
            }

            # Print summary
            if result.success:
                print(f"   ✅ Success! Found {len(result.segments)} segments")
                print(f"   ⏱️  Processing time: {processing_time:.2f}s")

                # Print segment summary
                segment_types = {}
                for segment in result.segments:
                    seg_type = segment.segment_type.value
                    segment_types[seg_type] = segment_types.get(seg_type, 0) + 1

                print("   📋 Segment types found:")
                for seg_type, count in segment_types.items():
                    print(f"      • {seg_type}: {count}")
            else:
                print(f"   ❌ Failed: {result.error}")

            return detailed_result

        except Exception as e:
            print(f"   💥 Exception: {e!s}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time,
                "image_info": None,
                "segments_found": 0,
                "segments": [],
            }

    def segment_to_dict(self, segment: DocumentSegment) -> dict[str, Any]:
        """Convert a DocumentSegment to a dictionary for JSON serialization"""
        return {
            "text": segment.text,
            "segment_type": segment.segment_type.value,
            "confidence": segment.confidence,
            "bbox": (
                {
                    "x": segment.bbox.x,
                    "y": segment.bbox.y,
                    "width": segment.bbox.width,
                    "height": segment.bbox.height,
                }
                if segment.bbox
                else None
            ),
            "metadata": segment.metadata,
        }

    def generate_summary_report(self, results: list[dict[str, Any]]):
        """Generate a comprehensive summary report"""
        print("📊 SEGMENTATION DEMO SUMMARY")
        print("=" * 60)

        total_images = len(results)
        successful_images = sum(1 for r in results if r["result"]["success"])
        total_segments = sum(r["result"]["segments_found"] for r in results)
        total_processing_time = sum(r["result"]["processing_time"] for r in results)

        print(f"Total Images Processed: {total_images}")
        print(f"Successful Segmentations: {successful_images}/{total_images}")
        print(f"Total Segments Found: {total_segments}")
        print(f"Total Processing Time: {total_processing_time:.2f}s")
        print(f"Average Time per Image: {total_processing_time / total_images:.2f}s")
        print()

        # Detailed results per image
        for i, result_data in enumerate(results, 1):
            filename = result_data["filename"]
            result = result_data["result"]

            print(f"{i}. {filename}")
            print(f"   Status: {'✅ Success' if result['success'] else '❌ Failed'}")

            if result["success"]:
                print(f"   Segments: {result['segments_found']}")
                print(f"   Processing time: {result['processing_time']:.2f}s")

                if result.get("image_info"):
                    img_info = result["image_info"]
                    print(
                        f"   Image: {img_info['width']}x{img_info['height']} {img_info['format']}"
                    )

                # Show segment breakdown
                if result["segments"]:
                    segment_types = {}
                    for segment in result["segments"]:
                        seg_type = segment["segment_type"]
                        segment_types[seg_type] = segment_types.get(seg_type, 0) + 1

                    type_summary = ", ".join(
                        [f"{k}: {v}" for k, v in segment_types.items()]
                    )
                    print(f"   Types: {type_summary}")
            else:
                print(f"   Error: {result.get('error', 'Unknown error')}")
            print()

        # Save detailed JSON report
        report_file = self.output_dir / "segmentation_report.json"
        with open(report_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"📄 Detailed report saved to: {report_file}")

    async def create_visualizations(self, results: list[dict[str, Any]]):
        """Create visual outputs showing detected segments"""
        print("\n🎨 Creating visualizations...")

        for result_data in results:
            if not result_data["result"]["success"]:
                continue

            filename = result_data["filename"]
            file_path = Path(result_data["file_path"])
            segments = result_data["result"]["segments"]

            if not segments:
                continue

            print(f"   🖼️  Creating visualization for {filename}")

            try:
                # Load original image
                with Image.open(file_path) as img:
                    # Create a copy for drawing
                    viz_img = img.copy()
                    if viz_img.mode != "RGB":
                        viz_img = viz_img.convert("RGB")

                    draw = ImageDraw.Draw(viz_img)

                    # Define colors for different segment types
                    colors = {
                        "title": "red",
                        "paragraph": "blue",
                        "table": "green",
                        "list_item": "orange",
                        "image": "purple",
                        "figure": "pink",
                        "header": "brown",
                        "footer": "gray",
                        "caption": "cyan",
                        "text": "yellow",
                    }

                    # Draw bounding boxes for each segment
                    for i, segment in enumerate(segments):
                        if segment["bbox"]:
                            bbox = segment["bbox"]
                            x1, y1 = bbox["x"], bbox["y"]
                            x2, y2 = x1 + bbox["width"], y1 + bbox["height"]

                            seg_type = segment["segment_type"]
                            color = colors.get(seg_type, "black")

                            # Draw rectangle
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

                            # Draw label
                            label = f"{i + 1}. {seg_type} ({segment['confidence']:.2f})"

                            # Try to use a font, fallback to default
                            try:
                                font = ImageFont.truetype("Arial.ttf", 16)
                            except:
                                font = ImageFont.load_default()

                            # Draw text background
                            text_bbox = draw.textbbox((x1, y1 - 20), label, font=font)
                            draw.rectangle(text_bbox, fill=color)
                            draw.text((x1, y1 - 20), label, fill="white", font=font)

                    # Save visualization
                    viz_filename = f"{file_path.stem}_segmented.png"
                    viz_path = self.output_dir / viz_filename
                    viz_img.save(viz_path)
                    print(f"      💾 Saved: {viz_filename}")

                    # Create text report for this image
                    self.create_text_report(filename, segments)

            except Exception as e:
                print(f"      ❌ Failed to create visualization: {e}")

    def create_text_report(self, filename: str, segments: list[dict[str, Any]]):
        """Create a detailed text report for an image"""
        report_filename = f"{Path(filename).stem}_segments.txt"
        report_path = self.output_dir / report_filename

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"SEGMENTATION REPORT: {filename}\n")
            f.write("=" * 60 + "\n\n")

            for i, segment in enumerate(segments, 1):
                f.write(f"SEGMENT {i}\n")
                f.write(f"Type: {segment['segment_type']}\n")
                f.write(f"Confidence: {segment['confidence']:.3f}\n")

                if segment["bbox"]:
                    bbox = segment["bbox"]
                    f.write(f"Position: ({bbox['x']}, {bbox['y']}) ")
                    f.write(f"Size: {bbox['width']}x{bbox['height']}\n")

                f.write(f"Text: {segment['text'][:200]}")
                if len(segment["text"]) > 200:
                    f.write("...")
                f.write("\n")

                if segment["metadata"]:
                    f.write(f"Metadata: {segment['metadata']}\n")

                f.write("-" * 40 + "\n\n")


async def main():
    """Main function to run the demo"""
    demo = SegmentationDemo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())
