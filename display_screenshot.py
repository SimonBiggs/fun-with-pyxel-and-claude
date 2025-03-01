#!/usr/bin/env python3
import os
import sys
import base64
from PIL import Image
from io import BytesIO
from pathlib import Path

def compress_and_encode_image(filepath, format="JPEG", quality=85, max_size=(640, 480)):
    """Compress and encode an image to base64 for displaying in the terminal"""
    try:
        # Open and resize the image if needed
        img = Image.open(filepath)
        img.thumbnail(max_size, Image.LANCZOS)
        
        # Save to BytesIO in specified format with compression
        buffered = BytesIO()
        img.save(buffered, format=format, quality=quality)
        
        # Get base64 encoded string
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Return data URL
        return f"data:image/{format.lower()};base64,{img_str}"
    except Exception as e:
        return f"Error processing image: {e}"

def list_screenshots():
    """List all screenshots in the directory"""
    screenshot_dir = os.path.expanduser("~/.cache/fun-with-pyxel")
    
    try:
        screenshots = sorted(Path(screenshot_dir).glob("*.png"), 
                            key=os.path.getmtime, reverse=True)
        
        if not screenshots:
            print("No screenshots found.")
            return []
            
        print("Available screenshots:")
        for i, screenshot in enumerate(screenshots):
            print(f"{i+1}. {screenshot.name}")
            
        return screenshots
    except Exception as e:
        print(f"Error listing screenshots: {e}")
        return []

def main():
    screenshots = list_screenshots()
    
    if not screenshots:
        return
    
    if len(sys.argv) > 1:
        try:
            idx = int(sys.argv[1]) - 1
        except ValueError:
            print(f"Error: '{sys.argv[1]}' is not a valid number. Please provide a screenshot number.")
            return
    else:
        # Default to the first (most recent) screenshot
        idx = 0
    
    if 0 <= idx < len(screenshots):
        filepath = screenshots[idx]
        print(f"\nDisplaying: {filepath.name}")
        data_url = compress_and_encode_image(filepath)
        print(data_url)
        print("\n(The image should display above if your terminal supports it)")
    else:
        print(f"Error: No screenshot found at index {idx+1}. Valid range is 1-{len(screenshots)}.")

if __name__ == "__main__":
    main()