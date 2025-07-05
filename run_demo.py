#!/usr/bin/env python3
"""
Simple runner script for the segmentation demo
"""
import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

from tests.demo_segmentation_on_real_data import main

if __name__ == "__main__":
    print("🚀 Starting Segmentation Demo on Real Data")
    print("This will process images from your data directory")
    print("Results will be saved to output/segmentation_demo/")
    print("-" * 50)
    
    asyncio.run(main()) 