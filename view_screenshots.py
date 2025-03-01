#!/usr/bin/env python3
import os
import sys
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from pathlib import Path

SCREENSHOT_DIR = os.path.expanduser("~/.cache/fun-with-pyxel")

class ScreenshotViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Pyxel Game Screenshots Viewer")
        self.root.geometry("800x600")
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create list frame on the left
        self.list_frame = ttk.Frame(self.main_frame, width=200)
        self.list_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Create listbox with scrollbar
        self.listbox_label = ttk.Label(self.list_frame, text="Screenshots:")
        self.listbox_label.pack(anchor="w")
        
        self.listbox = tk.Listbox(self.list_frame, width=40, height=30)
        self.listbox.pack(fill=tk.BOTH, expand=True)
        self.listbox.bind('<<ListboxSelect>>', self.on_screenshot_select)
        
        # Create image frame on the right
        self.image_frame = ttk.Frame(self.main_frame)
        self.image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.image_label = ttk.Label(self.image_frame, text="Screenshot will appear here")
        self.image_label.pack(pady=10)
        
        self.image_display = ttk.Label(self.image_frame)
        self.image_display.pack(fill=tk.BOTH, expand=True)
        
        # Load screenshots
        self.screenshots = []
        self.load_screenshots()
        
    def load_screenshots(self):
        """Load screenshot list from directory"""
        self.listbox.delete(0, tk.END)
        
        try:
            screenshots = sorted(Path(SCREENSHOT_DIR).glob("*.png"), 
                                key=os.path.getmtime, reverse=True)
            
            for i, screenshot in enumerate(screenshots):
                self.screenshots.append(str(screenshot))
                name = screenshot.name
                self.listbox.insert(tk.END, name)
                
            if self.screenshots:
                self.listbox.selection_set(0)
                self.on_screenshot_select(None)
        except Exception as e:
            self.image_label.config(text=f"Error loading screenshots: {e}")
            
    def on_screenshot_select(self, event):
        """Handle screenshot selection"""
        selection = self.listbox.curselection()
        if not selection:
            return
            
        index = selection[0]
        if 0 <= index < len(self.screenshots):
            self.display_screenshot(self.screenshots[index])
            
    def display_screenshot(self, filepath):
        """Display the selected screenshot"""
        try:
            # Get just the filename for display
            filename = os.path.basename(filepath)
            
            # Update screenshot name label
            self.image_label.config(text=filename)
            
            # Load and display image
            image = Image.open(filepath)
            # Scale the image up by 3x for better visibility
            width, height = image.size
            image = image.resize((width * 3, height * 3), Image.NEAREST)
            
            photo = ImageTk.PhotoImage(image)
            self.image_display.config(image=photo)
            self.image_display.image = photo  # Keep a reference to prevent garbage collection
            
        except Exception as e:
            self.image_label.config(text=f"Error displaying image: {e}")

def main():
    root = tk.Tk()
    app = ScreenshotViewer(root)
    root.mainloop()

if __name__ == "__main__":
    main()