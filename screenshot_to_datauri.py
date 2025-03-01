#!/usr/bin/env python3
import sys
import os
import zipfile
import tempfile
import base64
from io import BytesIO
from PIL import Image

def extract_pyxel_screenshot(screenshot_path):
    """Extract image data from a Pyxel screenshot and return as data URI"""
    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract the ZIP file
            with zipfile.ZipFile(screenshot_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
                
                # Look for the resource files
                for root, dirs, files in os.walk(tmpdir):
                    for filename in files:
                        if filename.endswith('.pyxres'):
                            file_path = os.path.join(root, filename)
                            with open(file_path, 'rb') as f:
                                img_data = f.read()
                                img_base64 = base64.b64encode(img_data).decode('utf-8')
                                return f"data:application/x-pyxel-resource;base64,{img_base64}"
                
                # If no .pyxres files found, use the pyxel_resource.toml file (it contains the image data)
                resource_file = os.path.join(tmpdir, "pyxel_resource.toml")
                if os.path.exists(resource_file):
                    with open(resource_file, 'rb') as f:
                        toml_data = f.read()
                        toml_base64 = base64.b64encode(toml_data).decode('utf-8')
                        return f"data:application/toml;base64,{toml_base64}"
                        
                # If we get here, no useful data was found
                return ""
    except Exception:
        return ""

def main():
    if len(sys.argv) != 2:
        # Just exit silently if no argument provided
        sys.exit(1)
        
    screenshot_path = sys.argv[1]
    if not os.path.exists(screenshot_path):
        # Exit silently if file doesn't exist
        sys.exit(1)
        
    datauri = extract_pyxel_screenshot(screenshot_path)
    if datauri:
        # Print only the data URI with no other text
        print(datauri)

if __name__ == "__main__":
    main()