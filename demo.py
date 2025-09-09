#!/usr/bin/env python3
"""Simple demo script for character attribute extraction."""

import sys
from pathlib import Path
from PIL import Image
import json

from character_pipeline import create_pipeline

def main():
    print("🎭 Character Attribute Extraction Demo")
    print("=" * 50)
    
    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = create_pipeline()
    print("✅ Pipeline initialized!")
    
    # Test with specified image
    image_path = "./continued/sensitive/danbooru_1380555_f9c05b66378137705fb63e010d6259d8.png"
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"\n📸 Processing image: {Path(image_path).name}")
    
    # Extract attributes
    try:
        attributes = pipeline.extract_from_image(image_path)
        
        print("\n🎯 Extracted Attributes:")
        print("-" * 30)
        
        result_dict = attributes.to_dict()
        
        for key, value in result_dict.items():
            if value:
                print(f"• {key}: {value}")
        
        print(f"\n📊 Confidence Score: {attributes.confidence_score:.2f}" if attributes.confidence_score else "\n📊 Confidence Score: N/A")
        
        print("\n📋 JSON Output:")
        print(json.dumps(result_dict, indent=2))
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()