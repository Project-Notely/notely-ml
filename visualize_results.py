#!/usr/bin/env python3
"""
Visualize the segmentation results from img3.PNG
"""

import json
import sys
from PIL import Image, ImageDraw, ImageFont

# Load the segmentation results
with open('output/notability_segmentation.json', 'r') as f:
    elements = json.load(f)

# Load the original image
image = Image.open('data/notability.png')
print(f"Original image size: {image.size}")

# Create a copy for drawing
result_image = image.copy()
draw = ImageDraw.Draw(result_image, 'RGBA')

# Color mapping
colors = {
    'Title': (255, 0, 0, 100),        # Red
    'Formula': (0, 255, 0, 100),      # Green
    'Image': (255, 0, 255, 100),      # Magenta
    'PageBreak': (128, 128, 128, 100), # Gray
}

# Draw bounding boxes for each element
for i, element in enumerate(elements):
    element_type = element['element_type']
    text = element['text']
    
    if element['metadata'].get('coordinates'):
        coords = element['metadata']['coordinates']
        if 'points' in coords:
            points = coords['points']
            if len(points) >= 2:
                # Get bounding box from points
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                x1, y1 = min(x_coords), min(y_coords)
                x2, y2 = max(x_coords), max(y_coords)
                
                color = colors.get(element_type, (128, 128, 128, 100))
                
                # Draw rectangle
                draw.rectangle([x1, y1, x2, y2], fill=color, outline=color[:3] + (255,), width=3)
                
                # Draw label
                try:
                    font = ImageFont.truetype("Arial.ttf", 24)
                except:
                    font = ImageFont.load_default()
                
                label = f"{element_type}: {text[:20]}..." if text else element_type
                
                # Label background
                text_bbox = draw.textbbox((0, 0), label, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                label_bg = (x1, y1 - text_height - 8, x1 + text_width + 16, y1)
                draw.rectangle(label_bg, fill=color[:3] + (200,))
                
                # Label text
                draw.text((x1 + 8, y1 - text_height - 4), label, fill=(0, 0, 0, 255), font=font)
                
                print(f"{i+1}. {element_type}: '{text}' at ({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")

# Save the result
result_image.save('output/notability_segmented_visualization.png')
print(f"\nVisualization saved as: notability_segmented_visualization.png")

# Print summary
print(f"\nSummary:")
print(f"- Found {len(elements)} elements")
element_counts = {}
for elem in elements:
    element_type = elem['element_type']
    element_counts[element_type] = element_counts.get(element_type, 0) + 1

for element_type, count in element_counts.items():
    print(f"- {element_type}: {count}") 