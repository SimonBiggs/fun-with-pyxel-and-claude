#!/usr/bin/env python3
import os
from pathlib import Path

SCREENSHOT_DIR = os.path.expanduser("~/.cache/fun-with-pyxel")

def list_screenshots():
    """List all screenshots in the directory with their details"""
    try:
        screenshots = sorted(Path(SCREENSHOT_DIR).glob("*.png"), 
                            key=os.path.getmtime, reverse=True)
        
        if not screenshots:
            print("No screenshots found.")
            return
            
        print("Available screenshots:")
        print(f"{'#':<3} {'Filename':<40} {'Size':<10} {'Path'}")
        print("-" * 80)
        
        for i, screenshot in enumerate(screenshots):
            size = os.path.getsize(screenshot)
            print(f"{i+1:<3} {screenshot.name:<40} {size:<10} {screenshot}")
            
    except Exception as e:
        print(f"Error listing screenshots: {e}")

if __name__ == "__main__":
    list_screenshots()