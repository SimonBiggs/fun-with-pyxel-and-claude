#!/usr/bin/env python3
import sys
import base64
from PIL import Image
from io import BytesIO

def encode_image(filepath, format="JPEG", quality=85, max_size=(640, 480)):
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
        return f"Error: {e}"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python show_image.py <path_to_image>")
        sys.exit(1)
        
    filepath = sys.argv[1]
    print(encode_image(filepath))