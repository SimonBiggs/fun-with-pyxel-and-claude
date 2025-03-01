#!/usr/bin/env python3
import sys
import os
import zipfile
import tempfile
import base64
from PIL import Image
from io import BytesIO

def extract_and_convert_pyxel_save(input_file):
    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract the ZIP file
            with zipfile.ZipFile(input_file, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
                
                # List extracted files for debugging
                print(f"Files in Pyxel save:")
                for root, dirs, files in os.walk(tmpdir):
                    for file in files:
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, tmpdir)
                        size = os.path.getsize(full_path)
                        print(f"  {rel_path} ({size} bytes)")
                
                # Look for the screen data or any images
                image_paths = []
                for root, dirs, files in os.walk(tmpdir):
                    for file in files:
                        if file.endswith(('.png', '.jpg', '.jpeg')) or file == 'screen':
                            image_paths.append(os.path.join(root, file))
                
                if not image_paths:
                    return "Error: No image data found in the Pyxel save file"
                
                # Try to convert each potential image to data URI
                for img_path in image_paths:
                    try:
                        with open(img_path, 'rb') as f:
                            data = f.read()
                            
                        # Try to open as an image
                        img = Image.open(BytesIO(data))
                        
                        # Convert to RGB for JPEG
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Compress to JPEG
                        output = BytesIO()
                        img.save(output, format='JPEG', quality=85)
                        
                        # Convert to base64
                        encoded = base64.b64encode(output.getvalue()).decode('ascii')
                        
                        # Return data URI
                        return f"data:image/jpeg;base64,{encoded}"
                    except Exception as e:
                        print(f"Failed to process {img_path}: {e}")
                
                return "Error: Could not process any images in the Pyxel save file"
    
    except zipfile.BadZipFile:
        return "Error: Not a valid ZIP file"
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_pyxel_save.py <pyxel_save_file>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} does not exist")
        sys.exit(1)
        
    datauri = extract_and_convert_pyxel_save(input_file)
    print(datauri)