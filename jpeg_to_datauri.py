#!/usr/bin/env python3
import sys
import os
import base64

def jpeg_to_datauri(jpeg_path):
    """Convert a JPEG file to a data URI"""
    try:
        with open(jpeg_path, 'rb') as f:
            img_data = f.read()
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            return f"data:image/jpeg;base64,{img_base64}"
    except Exception:
        return ""

def main():
    if len(sys.argv) != 2:
        # Just exit silently if no argument provided
        sys.exit(1)
        
    jpeg_path = sys.argv[1]
    if not os.path.exists(jpeg_path):
        # Exit silently if file doesn't exist
        sys.exit(1)
        
    datauri = jpeg_to_datauri(jpeg_path)
    if datauri:
        # Print only the data URI with no other text
        print(datauri)

if __name__ == "__main__":
    main()