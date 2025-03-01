#!/usr/bin/env python3
import os
import sys
import zipfile
import tempfile
import toml
import pyxel
from pathlib import Path

SCREENSHOT_DIR = os.path.expanduser("~/.cache/fun-with-pyxel")

def list_screenshots():
    """List all screenshots in the directory"""
    try:
        screenshots = sorted(Path(SCREENSHOT_DIR).glob("*.png"), 
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

def extract_pyxel_resource(pyxel_file):
    """Extract the pyxel resource file from the zip archive"""
    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract the ZIP file
            with zipfile.ZipFile(pyxel_file, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
                
                # Check for the resource file
                resource_file = os.path.join(tmpdir, "pyxel_resource.toml")
                if not os.path.exists(resource_file):
                    print(f"No resource file found in {pyxel_file}")
                    return None
                
                # Load the TOML file to get resource details
                with open(resource_file, 'r') as f:
                    resource_data = toml.load(f)
                    print(f"Resource data: {resource_data.keys()}")
                    
                    # If resource has image data, return the path to use with pyxel.load
                    return tmpdir
    except Exception as e:
        print(f"Error extracting resource: {e}")
        return None

def display_screenshot(screenshot_path):
    """Display a screenshot using Pyxel"""
    try:
        resource_dir = extract_pyxel_resource(screenshot_path)
        if not resource_dir:
            print("Failed to extract resource data")
            return
        
        # Load the resource
        resource_file = os.path.join(resource_dir, "pyxel_resource.toml")
        
        # Define a simple app to display the screenshot
        class ScreenshotApp:
            def __init__(self):
                self.filename = os.path.basename(screenshot_path)
                pyxel.init(160, 120, title=f"Screenshot: {self.filename}")
                pyxel.load(resource_file)
                pyxel.run(self.update, self.draw)
                
            def update(self):
                if pyxel.btnp(pyxel.KEY_Q) or pyxel.btnp(pyxel.KEY_ESCAPE):
                    pyxel.quit()
                    
            def draw(self):
                pyxel.cls(0)
                
                # The game screen should be in image bank 0
                pyxel.bltm(0, 0, 0, 0, 0, 160, 120)
                
                # Draw text
                pyxel.text(4, 4, f"Screenshot: {self.filename}", 7)
                pyxel.text(4, 12, "Press Q or ESC to quit", 7)
                
        # Start the app
        ScreenshotApp()
    except Exception as e:
        print(f"Error displaying screenshot: {e}")

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
        display_screenshot(filepath)
    else:
        print(f"Error: No screenshot found at index {idx+1}. Valid range is 1-{len(screenshots)}.")

if __name__ == "__main__":
    main()