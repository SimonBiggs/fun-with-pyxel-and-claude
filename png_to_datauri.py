#!/usr/bin/env python3
import sys
import os
import base64
from PIL import Image
from io import BytesIO

def convert_png_to_datauri(input_file):
    try:
        # Load file using PIL
        with open(input_file, 'rb') as f:
            file_content = f.read()
            
        # Try to figure out what kind of file it is
        if file_content.startswith(b'PK\x03\x04'):
            print(f"File {input_file} appears to be a ZIP file, not a PNG.")
            # For ZIP files, we might want to extract the image if it's inside
            # But for now, let's just return an error
            return "Error: File is a ZIP file, not a PNG"
            
        # Try to open it as a regular image
        try:
            with Image.open(BytesIO(file_content)) as img:
                # Compress to JPEG
                output = BytesIO()
                
                # Convert to RGB mode if needed (for JPEG)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    
                # Save as JPEG
                img.save(output, format='JPEG', quality=85)
                
                # Convert to base64
                encoded = base64.b64encode(output.getvalue()).decode('ascii')
                
                # Return data URI
                return f"data:image/jpeg;base64,{encoded}"
        except Exception as e:
            # If we can't open as image, try to extract info
            print(f"Error opening {input_file} as image: {str(e)}")
            
            # Output first few bytes for debugging
            print(f"First 20 bytes: {file_content[:20].hex()}")
            
            return f"Error: Failed to process image - {str(e)}"
    
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python png_to_datauri.py <png_file>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} does not exist")
        sys.exit(1)
        
    datauri = convert_png_to_datauri(input_file)
    print(datauri)