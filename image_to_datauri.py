#!/usr/bin/env python3
import sys
import os
import base64
import imghdr
from PIL import Image
import io

def image_to_datauri(image_path, quality=85):
    """Convert any image file to a JPEG data URI with compression"""
    try:
        # Open the image with PIL
        img = Image.open(image_path)
        
        # Convert to RGB if necessary (for PNG with transparency, etc.)
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            # Create a white background
            background = Image.new('RGB', img.size, (255, 255, 255))
            # Paste the image on the background
            if img.mode == 'RGBA':
                background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
            elif img.mode == 'LA':
                background.paste(img, mask=img.split()[1])  # Use alpha channel as mask
            elif img.mode == 'P':
                background.paste(img, mask=img.convert('RGBA').split()[3])
            img = background
        
        # For non-RGB images without alpha (like grayscale), just convert to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Create a BytesIO buffer to store the compressed image
        buffer = io.BytesIO()
        # Save the image to the buffer with compression as JPEG
        img.save(buffer, format='JPEG', quality=quality, optimize=True)
        # Get the compressed image data
        buffer.seek(0)
        img_data = buffer.read()
                
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        return f"data:image/jpeg;base64,{img_base64}"
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return ""

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python image_to_datauri.py <image_path> [quality]", file=sys.stderr)
        sys.exit(1)
        
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Get quality parameter if provided
    quality = 85  # Default quality
    if len(sys.argv) == 3:
        try:
            quality = int(sys.argv[2])
            if quality < 1 or quality > 100:
                print("Quality must be between 1-100. Using default 85.", file=sys.stderr)
                quality = 85
        except ValueError:
            print("Quality must be a number between 1-100. Using default 85.", file=sys.stderr)
            quality = 85
        
    datauri = image_to_datauri(image_path, quality)
    if datauri:
        # Print only the data URI with no other text
        print(datauri)

if __name__ == "__main__":
    main()