import pyxel
import random
import noise
import time
import logging
import os
import sys
import datetime
import json
import socket
import threading
import argparse
import math
from pathlib import Path

# Set up logging
log_file = "/home/david/code/fun-with-pyxel-and-claude/game_logs.txt"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Screenshot directory
SCREENSHOT_DIR = os.path.expanduser("~/.cache/fun-with-pyxel")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# Game constants - MADE LARGER for better screenshots
WIDTH = 675  # Half of 1350 
HEIGHT = 700  # Slightly reduced from 720 for better compatibility
TILE_SIZE = 14  # Larger tiles for better visibility
HOT_RELOAD_ENABLED = True  # Flag to indicate hot reload is active

# Colors - Stardew Valley inspired palette
PLAYER_COLOR = 7      # Lighter skin/face tone
WALL_COLOR = 5        # Dark brown for walls
GOAL_COLOR = 10       # Gold/yellow for treasures
ENEMY_COLOR = 8       # Red for enemies
POTION_COLOR = 11     # Light green for potions
GRASS_COLOR = 11      # Bright green for grass
WATER_COLOR = 12      # Bright blue for water
SAND_COLOR = 9        # Light brown/tan for sand/soil
MOUNTAIN_COLOR = 13   # Gray for mountains/stones
TREE_COLOR = 3        # Dark green for trees
PATH_COLOR = 4        # Brown for paths/dirt
CASTLE_COLOR = 6      # Light gray for buildings

# Terrain types
WATER = 0
SAND = 1
GRASS = 2
FOREST = 3
MOUNTAIN = 4
WALL = 5
POTION = 6
GOAL = 7
PATH = 8
CASTLE = 9

class Enemy:
    def __init__(self, name, hp, attack, defense):
        self.name = name
        self.hp = hp
        self.max_hp = hp
        self.attack = attack
        self.defense = defense
        logging.info(f"Created {name} enemy with {hp} HP, {attack} ATK, {defense} DEF")

class ScreenshotManager:
    def __init__(self, directory):
        self.directory = directory
        self.session_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.screenshot_count = 0
        
    def take_screenshot(self, event_name):
        """Take a screenshot and save it with the event name in the filename as a standard PNG
        
        Returns the filepath to the screenshot.
        """
        filename = f"{self.session_id}_{self.screenshot_count:03d}_{event_name}.png"
        filepath = os.path.join(self.directory, filename)
        
        # First save using Pyxel's built-in method to a temp file
        temp_filepath = os.path.join(self.directory, f"temp_{self.session_id}_{self.screenshot_count}.png")
        pyxel.save(temp_filepath)
        
        # Get the screen data directly from Pyxel's screen buffer
        width, height = WIDTH, HEIGHT
        
        # Create a new PIL Image and copy the screen data
        from PIL import Image
        img = Image.new("RGB", (width, height))
        pixels = []
        
        # Iterate through the screen pixels and get the colors
        for y in range(height):
            for x in range(width):
                # Get the color index from Pyxel's screen
                col = pyxel.pget(x, y)
                
                # Convert Pyxel's color palette to RGB
                # Pyxel's default 16-color palette
                palette = [
                    (0, 0, 0),       # 0: Black
                    (29, 43, 83),    # 1: Dark Blue
                    (126, 37, 83),   # 2: Purple
                    (0, 135, 81),    # 3: Green
                    (171, 82, 54),   # 4: Brown
                    (95, 87, 79),    # 5: Dark Gray
                    (194, 195, 199), # 6: Light Gray
                    (255, 241, 232), # 7: White
                    (255, 0, 77),    # 8: Red
                    (255, 163, 0),   # 9: Orange
                    (255, 236, 39),  # 10: Yellow
                    (0, 228, 54),    # 11: Light Green
                    (41, 173, 255),  # 12: Light Blue
                    (131, 118, 156), # 13: Lavender
                    (255, 119, 168), # 14: Pink
                    (255, 204, 170)  # 15: Light Peach
                ]
                
                # Get the RGB color from palette
                rgb = palette[col] if 0 <= col < len(palette) else (0, 0, 0)
                pixels.append(rgb)
        
        # Put the pixels into the image
        img.putdata(pixels)
        
        # Save the image as a standard PNG
        img.save(filepath, format="PNG")
        
        # Remove the temporary file
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
            
        logging.info(f"Screenshot saved as standard PNG: {filepath}")
        self.screenshot_count += 1
        
        # Return the filepath for CLI commands
        return filepath

class WorldGenerator:
    def __init__(self, width, height, seed=None):
        self.width = width
        self.height = height
        self.seed = seed or random.randint(0, 999999)
        logging.info(f"Generating world with seed: {self.seed}")
        if not hasattr(self, 'headless') or not self.headless:
            print("Hot reload is active! Edit the game code and see changes immediately.")
        
    def generate_terrain(self):
        # Initialize empty world grid
        world = [[0 for _ in range(self.width)] for _ in range(self.height)]
        
        # Generate terrain using Perlin noise with improved parameters
        scale = 12.0  # Increased scale for more varied terrain
        
        # Add multiple noise layers for more detailed terrain
        for y in range(self.height):
            for x in range(self.width):
                nx = x / self.width - 0.5
                ny = y / self.height - 0.5
                
                # Base elevation using Perlin noise
                elevation = noise.pnoise2(nx * scale, ny * scale, 
                                       octaves=8,  # Increased octaves for more detail
                                       persistence=0.6,  # Slightly higher persistence
                                       lacunarity=2.2,  # Increased lacunarity for more variation
                                       base=self.seed)
                
                # Add secondary elevation features for more realism
                elevation_detail = noise.pnoise2(nx * scale * 2, ny * scale * 2,
                                      octaves=4,
                                      persistence=0.3,
                                      lacunarity=2.5,
                                      base=self.seed + 2) * 0.2
                                      
                # Combine elevation layers
                elevation = elevation + elevation_detail
                
                # Moisture using different noise settings
                moisture = noise.pnoise2(nx * scale + 5, ny * scale + 5,
                                      octaves=5,  # More octaves for moisture
                                      persistence=0.55,
                                      lacunarity=2.1,
                                      base=self.seed + 1)
                
                # River channels using different noise
                river_value = noise.pnoise2(nx * scale * 3, ny * scale * 3,
                                        octaves=3,
                                        persistence=0.4,
                                        lacunarity=2.0,
                                        base=self.seed + 3)
                
                # River probability - higher near existing water
                river_probability = 0.97
                is_river = abs(river_value) > river_probability and elevation < 0.1
                
                # Determine terrain type based on elevation and moisture
                if is_river or elevation < -0.25:
                    world[y][x] = WATER  # Water or rivers
                elif elevation < -0.08:
                    world[y][x] = SAND   # Sand/Beach
                elif elevation < 0.15:
                    if moisture > 0.2:
                        world[y][x] = FOREST  # Forest (increased forest density)
                    else:
                        world[y][x] = GRASS   # Grass/Plains
                elif elevation < 0.35:
                    if moisture > 0.1:
                        world[y][x] = FOREST  # Forest at higher elevations
                    else:
                        world[y][x] = GRASS   # More varied grass/plains
                else:
                    world[y][x] = MOUNTAIN  # Mountains
        
        # Apply terrain smoothing to prevent isolated tiles
        world = self.smooth_terrain(world)
        
        # Add border walls
        for x in range(self.width):
            world[0][x] = WALL
            world[self.height-1][x] = WALL
            
        for y in range(self.height):
            world[y][0] = WALL
            world[y][self.width-1] = WALL
            
        # Add medieval themed elements - castles and paths
        self.add_castles_and_paths(world)
        
        return world
        
    def smooth_terrain(self, world):
        """Apply smoothing to reduce isolated terrain tiles"""
        smoothed = [row[:] for row in world]  # Create a copy
        
        for y in range(1, self.height-1):
            for x in range(1, self.width-1):
                # Check for isolated tiles
                terrain_type = world[y][x]
                
                # Skip walls and special features
                if terrain_type in [WALL, CASTLE, GOAL, POTION]:
                    continue
                
                # Count neighboring tiles of same type
                neighbor_count = 0
                neighbor_types = {}
                
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                            
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            neighbor_type = world[ny][nx]
                            if neighbor_type == terrain_type:
                                neighbor_count += 1
                            
                            if neighbor_type not in neighbor_types:
                                neighbor_types[neighbor_type] = 0
                            neighbor_types[neighbor_type] += 1
                
                # If this tile has few neighbors of same type, change it to most common neighbor
                if neighbor_count < 2:  # Isolated or only one neighbor of same type
                    most_common = terrain_type
                    max_count = 0
                    
                    for ntype, count in neighbor_types.items():
                        if count > max_count and ntype not in [WALL, CASTLE, GOAL, POTION]:
                            max_count = count
                            most_common = ntype
                    
                    smoothed[y][x] = most_common
        
        return smoothed
    
    def add_castles_and_paths(self, world):
        """Add medieval themed elements to the world"""
        # Add 3-5 castles for a more populated world
        num_castles = random.randint(3, 5)
        castle_positions = []
        
        # Find suitable castle positions (flat areas)
        for _ in range(num_castles):
            attempts = 0
            while attempts < 50:  # Limit attempts
                x = random.randint(5, self.width - 6)
                y = random.randint(5, self.height - 6)
                
                # Check if the area is flat (grass or forest)
                flat_area = True
                for cy in range(y-3, y+4):  # Larger area check for better castle placement
                    for cx in range(x-3, x+4):
                        if not (0 <= cx < self.width and 0 <= cy < self.height):
                            flat_area = False
                            break
                        if world[cy][cx] not in [GRASS, FOREST]:
                            flat_area = False
                            break
                
                # If we found a flat area and it's far enough from other castles
                if flat_area and all(abs(x-cx) + abs(y-cy) > 15 for cx, cy in castle_positions):
                    # Place castle (larger 4x4 area for more imposing castles)
                    for cy in range(y-2, y+2):
                        for cx in range(x-2, x+2):
                            world[cy][cx] = CASTLE
                    
                    # Add outer wall/courtyard around some castles
                    if random.random() < 0.6:  # 60% chance for castle to have outer walls
                        # Top and bottom walls
                        for cx in range(x-3, x+3):
                            if 0 <= cx < self.width and 0 <= y-3 < self.height and world[y-3][cx] not in [CASTLE, WALL]:
                                world[y-3][cx] = WALL
                            if 0 <= cx < self.width and 0 <= y+2 < self.height and world[y+2][cx] not in [CASTLE, WALL]:
                                world[y+2][cx] = WALL
                        
                        # Left and right walls
                        for cy in range(y-3, y+3):
                            if 0 <= cy < self.height and 0 <= x-3 < self.width and world[cy][x-3] not in [CASTLE, WALL]:
                                world[cy][x-3] = WALL
                            if 0 <= cy < self.height and 0 <= x+2 < self.width and world[cy][x+2] not in [CASTLE, WALL]:
                                world[cy][x+2] = WALL
                        
                        # Add gate (open space in walls)
                        gate_side = random.choice(['top', 'right', 'bottom', 'left'])
                        if gate_side == 'top' and 0 <= y-3 < self.height and 0 <= x < self.width:
                            world[y-3][x] = PATH
                        elif gate_side == 'right' and 0 <= y < self.height and 0 <= x+2 < self.width:
                            world[y][x+2] = PATH
                        elif gate_side == 'bottom' and 0 <= y+2 < self.height and 0 <= x < self.width:
                            world[y+2][x] = PATH
                        elif gate_side == 'left' and 0 <= y < self.height and 0 <= x-3 < self.width:
                            world[y][x-3] = PATH
                    
                    castle_positions.append((x, y))
                    logging.info(f"Placed castle at ({x}, {y})")
                    break
                    
                attempts += 1
        
        # Add village or settlement hubs around some castles
        for castle_x, castle_y in castle_positions:
            if random.random() < 0.7:  # 70% chance for castle to have a village
                self.add_village(world, castle_x, castle_y)
        
        # Add paths between castles and other interesting areas
        path_endpoints = castle_positions.copy()
        
        # Add some natural features as path endpoints
        for _ in range(5):  # More natural features
            x = random.randint(5, self.width - 6)
            y = random.randint(5, self.height - 6)
            # Make sure it's not too close to existing endpoints
            if all(abs(x-ex) + abs(y-ey) > 10 for ex, ey in path_endpoints):
                # Create small landmarks at some endpoints (small shrines, etc)
                if random.random() < 0.4:
                    world[y][x] = CASTLE  # Using castle tile for landmarks
                    # Add some surrounding decoration
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            nx, ny = x + dx, y + dy
                            if (0 <= nx < self.width and 0 <= ny < self.height and 
                                world[ny][nx] not in [CASTLE, WALL, WATER]):
                                world[ny][nx] = PATH  # Create shrine surroundings
                
                path_endpoints.append((x, y))
        
        # Generate paths between endpoints using better pathfinding
        for i in range(len(path_endpoints)):
            for j in range(i+1, len(path_endpoints)):
                start_x, start_y = path_endpoints[i]
                end_x, end_y = path_endpoints[j]
                
                # Only connect if not too far away (create regional connections)
                if abs(start_x - end_x) + abs(start_y - end_y) < min(self.width, self.height) // 2:
                    # Generate path using A* like approach
                    self.generate_path(world, start_x, start_y, end_x, end_y)
        
        # Create a few long distance trade routes that cross the map
        for _ in range(2):
            # Choose random points on edges
            side1 = random.choice(['top', 'right', 'bottom', 'left'])
            side2 = random.choice(['top', 'right', 'bottom', 'left'])
            while side1 == side2:
                side2 = random.choice(['top', 'right', 'bottom', 'left'])
                
            if side1 == 'top':
                x1 = random.randint(5, self.width-6)
                y1 = 2  # Near top edge
            elif side1 == 'right':
                x1 = self.width-3  # Near right edge
                y1 = random.randint(5, self.height-6)
            elif side1 == 'bottom':
                x1 = random.randint(5, self.width-6)
                y1 = self.height-3  # Near bottom edge
            else:  # left
                x1 = 2  # Near left edge
                y1 = random.randint(5, self.height-6)
                
            if side2 == 'top':
                x2 = random.randint(5, self.width-6)
                y2 = 2  # Near top edge
            elif side2 == 'right':
                x2 = self.width-3  # Near right edge
                y2 = random.randint(5, self.height-6)
            elif side2 == 'bottom':
                x2 = random.randint(5, self.width-6)
                y2 = self.height-3  # Near bottom edge
            else:  # left
                x2 = 2  # Near left edge
                y2 = random.randint(5, self.height-6)
                
            # Create a cross-map trade route
            self.generate_path(world, x1, y1, x2, y2, is_major_road=True)
    
    def add_village(self, world, castle_x, castle_y):
        """Add a small village around a castle"""
        # Determine village size
        village_size = random.randint(3, 7)
        
        # Place buildings around castle at varying distances
        for _ in range(village_size):
            # Random offset from castle
            offset_distance = random.randint(4, 8)
            angle = random.uniform(0, 2 * 3.14159)  # Random angle
            
            # Calculate position
            bx = int(castle_x + offset_distance * math.cos(angle))
            by = int(castle_y + offset_distance * math.sin(angle))
            
            # Ensure within bounds
            if not (0 <= bx < self.width and 0 <= by < self.height):
                continue
            
            # Place a small building if area is suitable
            if world[by][bx] in [GRASS, PATH]:
                world[by][bx] = CASTLE  # Use castle tile for buildings
                
                # Add small paths connecting buildings to castle
                self.generate_path(world, bx, by, castle_x, castle_y, max_winding=0.2)
    
    def generate_path(self, world, start_x, start_y, end_x, end_y, is_major_road=False, max_winding=0.3):
        """Generate a winding path between two points
        
        Args:
            world: The world grid to modify
            start_x, start_y: Starting coordinates
            end_x, end_y: Ending coordinates
            is_major_road: If True, creates a wider, more prominent path
            max_winding: Maximum winding factor (0-1) - lower means straighter paths
        """
        current_x, current_y = start_x, start_y
        path_length = abs(end_x - start_x) + abs(end_y - start_y)
        
        # Track visited positions to avoid loops
        visited = set([(current_x, current_y)])
        
        # Create a winding path, moving roughly toward the destination
        for _ in range(path_length * 3):  # Allow extra steps for winding
            # Mark current position as path if it's not already a special tile
            if world[current_y][current_x] not in [CASTLE, WALL]:
                world[current_y][current_x] = PATH
                
            # For major roads, make the path wider
            if is_major_road:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = current_x + dx, current_y + dy
                        if (0 <= nx < self.width and 0 <= ny < self.height and 
                            world[ny][nx] not in [CASTLE, WALL, WATER]):
                            # 50% chance to widen road at each point
                            if random.random() < 0.5:
                                world[ny][nx] = PATH
            
            # If we've reached the end, stop
            if current_x == end_x and current_y == end_y:
                break
                
            # Decide which direction to move (with controlled randomness for winding)
            if random.random() < (1.0 - max_winding):  # Chance to move toward destination
                # Move in the direction of the endpoint with improved pathfinding
                directions = []
                
                # Calculate distances for each possible move
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                    nx, ny = current_x + dx, current_y + dy
                    
                    # Check if the new position is valid
                    if (1 <= nx < self.width-1 and 1 <= ny < self.height-1 and 
                        (nx, ny) not in visited and
                        world[ny][nx] not in [WALL, WATER]):
                        
                        # Calculate Manhattan distance to goal
                        dist = abs(nx - end_x) + abs(ny - end_y)
                        
                        # Add terrain cost
                        terrain_cost = 0
                        if world[ny][nx] == MOUNTAIN:
                            terrain_cost = 5  # Very high cost for mountains
                        elif world[ny][nx] == FOREST:
                            terrain_cost = 2  # Medium cost for forests
                        
                        # Calculate total cost (distance + terrain)
                        total_cost = dist + terrain_cost
                        
                        # Add direction with its cost
                        directions.append((dx, dy, total_cost))
                
                if directions:
                    # Sort by cost (lowest first)
                    directions.sort(key=lambda x: x[2])
                    
                    # Choose one of the best directions (some randomness in selection)
                    if len(directions) > 1 and random.random() < 0.3:
                        # 30% chance to take the second-best path for some variation
                        dx, dy, _ = directions[1]
                    else:
                        # Otherwise take the best path
                        dx, dy, _ = directions[0]
                    
                    new_x = current_x + dx
                    new_y = current_y + dy
                    current_x, current_y = new_x, new_y
                    visited.add((current_x, current_y))
                    continue
                    
            # If we get here, either random movement or no valid paths toward goal
            # Try random direction
            shuffled_dirs = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
            random.shuffle(shuffled_dirs)
            
            moved = False
            for dx, dy in shuffled_dirs:
                new_x = current_x + dx
                new_y = current_y + dy
                
                if (1 <= new_x < self.width-1 and 
                    1 <= new_y < self.height-1 and
                    (new_x, new_y) not in visited and
                    world[new_y][new_x] not in [WALL, WATER]):
                    current_x, current_y = new_x, new_y
                    visited.add((current_x, current_y))
                    moved = True
                    break
                    
            # If completely stuck, break
            if not moved:
                break
    
    def place_potions(self, world, count):
        """Place potions strategically on the map (on walkable tiles)"""
        # First, identify different regions for potion placement
        walkable_positions = []
        forest_positions = []
        path_positions = []
        grass_positions = []
        
        for y in range(self.height):
            for x in range(self.width):
                # Find walkable spaces (not water, mountain or wall)
                if world[y][x] not in [WATER, MOUNTAIN, WALL, CASTLE]:
                    walkable_positions.append((x, y))
                    
                    # Categorize by terrain type for strategic placement
                    if world[y][x] == FOREST:
                        forest_positions.append((x, y))
                    elif world[y][x] == PATH:
                        path_positions.append((x, y))
                    elif world[y][x] == GRASS:
                        grass_positions.append((x, y))
        
        # Calculate how many potions to place in each terrain type
        # More potions on paths (easy to find) and in forests (rewarding exploration)
        total_potions = min(count, len(walkable_positions))
        path_potion_count = min(int(total_potions * 0.4), len(path_positions))  # 40% on paths
        forest_potion_count = min(int(total_potions * 0.4), len(forest_positions))  # 40% in forests
        grass_potion_count = min(total_potions - path_potion_count - forest_potion_count, len(grass_positions))
        
        # Place potions on paths (easy to find)
        potion_positions = []
        if path_positions:
            # Place evenly along paths 
            path_indices = sorted(random.sample(range(len(path_positions)), path_potion_count))
            for idx in path_indices:
                x, y = path_positions[idx]
                world[y][x] = POTION
                potion_positions.append((x, y))
        
        # Place potions in forests (hidden treasures)
        if forest_positions:
            # Try to place in deep forests (positions with many forest neighbors)
            forest_scores = []
            for x, y in forest_positions:
                if (x, y) in potion_positions:
                    continue
                    
                # Count forest neighbors to find "deep forest"
                forest_neighbors = 0
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            if world[ny][nx] == FOREST:
                                forest_neighbors += 1
                
                forest_scores.append((x, y, forest_neighbors))
            
            # Sort by score (higher is deeper forest)
            forest_scores.sort(key=lambda x: x[2], reverse=True)
            
            # Take the top forest positions
            for i in range(min(forest_potion_count, len(forest_scores))):
                x, y, _ = forest_scores[i]
                world[y][x] = POTION
                potion_positions.append((x, y))
        
        # Place remaining potions in grass areas, focusing on interesting locations
        if grass_positions:
            grass_positions = [(x, y) for x, y in grass_positions if (x, y) not in potion_positions]
            
            # Create groups of potions in certain areas
            remaining_potions = grass_potion_count
            cluster_count = min(3, remaining_potions // 2)  # Up to 3 clusters
            
            for _ in range(cluster_count):
                if not grass_positions or remaining_potions <= 0:
                    break
                    
                # Pick a random starting point for the cluster
                start_idx = random.randrange(len(grass_positions))
                start_x, start_y = grass_positions[start_idx]
                
                # Place a potion at the start point
                world[start_y][start_x] = POTION
                potion_positions.append((start_x, start_y))
                grass_positions.remove((start_x, start_y))
                remaining_potions -= 1
                
                # Place 1-3 more potions nearby
                cluster_size = min(random.randint(1, 3), remaining_potions, len(grass_positions))
                neighbor_positions = []
                
                # Find nearby grass positions
                for y in range(max(0, start_y-3), min(self.height, start_y+4)):
                    for x in range(max(0, start_x-3), min(self.width, start_x+4)):
                        if (x, y) in grass_positions:
                            dist = abs(x - start_x) + abs(y - start_y)
                            neighbor_positions.append((x, y, dist))
                
                if neighbor_positions:
                    # Sort by distance
                    neighbor_positions.sort(key=lambda x: x[2])
                    
                    # Place cluster potions
                    for i in range(min(cluster_size, len(neighbor_positions))):
                        x, y, _ = neighbor_positions[i]
                        world[y][x] = POTION
                        potion_positions.append((x, y))
                        if (x, y) in grass_positions:
                            grass_positions.remove((x, y))
                        remaining_potions -= 1
            
            # Place any remaining potions randomly in grass
            if remaining_potions > 0 and grass_positions:
                random_positions = random.sample(grass_positions, min(remaining_potions, len(grass_positions)))
                for x, y in random_positions:
                    world[y][x] = POTION
                    potion_positions.append((x, y))
        
        # If we still have potions to place, use random selection
        placed_count = len(potion_positions)
        remaining_count = total_potions - placed_count
        
        if remaining_count > 0:
            # Get positions that don't already have potions
            available_positions = [(x, y) for x, y in walkable_positions 
                                  if (x, y) not in potion_positions]
            
            if available_positions:
                random_positions = random.sample(available_positions, 
                                              min(remaining_count, len(available_positions)))
                for x, y in random_positions:
                    world[y][x] = POTION
                    potion_positions.append((x, y))
            
        logging.info(f"Placed {len(potion_positions)} potions strategically in the world")
        return world
    
    def find_player_start(self, world):
        """Find a good starting position for the player"""
        # Look for path or grass tiles away from the edge
        candidates = []
        for y in range(2, self.height-2):
            for x in range(2, self.width-2):
                if world[y][x] in [PATH, GRASS]:
                    candidates.append((x, y))
        
        if candidates:
            return random.choice(candidates)
        else:
            # Fallback to any walkable position
            for y in range(2, self.height-2):
                for x in range(2, self.width-2):
                    if world[y][x] not in [WALL, WATER, MOUNTAIN]:
                        return (x, y)
        
        # Last resort
        return (self.width // 4, self.height // 4)
    
    def find_goal_position(self, world, player_start):
        """Find a goal position that's far from the player start"""
        px, py = player_start
        
        # Try to place the goal in or near a castle
        castle_positions = []
        for y in range(2, self.height-2):
            for x in range(2, self.width-2):
                if world[y][x] == CASTLE:
                    # Check surrounding tiles for a suitable position
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            nx, ny = x + dx, y + dy
                            if (0 <= nx < self.width and 0 <= ny < self.height and 
                                abs(nx - px) + abs(ny - py) > (self.width + self.height) // 4):
                                castle_positions.append((nx, ny))
        
        if castle_positions:
            x, y = random.choice(castle_positions)
            return (x, y)
        
        # Fallback - find a distant position if no castles are suitable
        candidates = []
        for y in range(2, self.height-2):
            for x in range(2, self.width-2):
                if world[y][x] in [GRASS, FOREST, PATH]:
                    # Manhattan distance
                    dist = abs(x - px) + abs(y - py)
                    if dist > (self.width + self.height) // 3:  # At least 1/3 of the map size away
                        candidates.append((x, y, dist))
        
        # Sort by distance and pick one of the farthest
        if candidates:
            candidates.sort(key=lambda x: x[2], reverse=True)
            # Pick from the top 10 or fewer
            top_n = min(10, len(candidates))
            x, y, _ = random.choice(candidates[:top_n])
            return (x, y)
        
        # Fallback - opposite corner
        return (self.width - 4, self.height - 4)

class MazeGame:
    def __init__(self, headless=False, socket_path=None):
        self.world_width = 50  # Larger world size
        self.world_height = 40
        self.headless = headless
        
        # Command queue for CLI commands
        self.command_queue = []
        self.scheduled_quit = False
        
        # Initialize Pyxel
        if not headless:
            pyxel.init(WIDTH, HEIGHT, title="Medieval Fantasy RPG - Co-op Mode")
        else:
            # Headless mode - just init normally but don't show window 
            # (Pyxel doesn't have true headless mode, but we can still control via socket)
            pyxel.init(WIDTH, HEIGHT, title="Medieval Fantasy RPG - Headless Mode")
        
        # Initialize screenshot manager
        self.screenshot_mgr = ScreenshotManager(SCREENSHOT_DIR)
        
        # Socket server for CLI control
        self.socket_server = None
        if socket_path:
            self.socket_server = SocketServer(self, socket_path)
        
        # Initialize game
        logging.info("Starting new co-op game")
        
        # Game states
        self.EXPLORING = 0
        self.BATTLE = 1
        self.GAME_OVER = 2
        self.WIN = 3
        self.PLAYER_SELECT = 4  # New state for selecting active player
        
        self.game_state = self.EXPLORING
        
        # Generate world
        self.generate_world()
        
        # Human player stats (knight)
        self.player_hp = 25
        self.player_max_hp = 25
        self.player_attack = 6
        self.player_defense = 3
        
        # AI companion (Claude) stats - wizard character
        self.companion_hp = 20
        self.companion_max_hp = 20
        self.companion_attack = 8     # Higher attack (spells)
        self.companion_defense = 2    # Lower defense
        self.companion_x = 0          # Will be set after player position
        self.companion_y = 0
        self.companion_frame = 0      # For animation
        self.companion_spells = 5     # Special ability - limited use spells
        
        # Shared inventory
        self.potions = 4  # Start with 4 potions for co-op mode
        
        # Active player (0 for human player, 1 for AI companion)
        self.active_player = 0
        
        # AI companion logic variables
        self.companion_move_timer = 0
        self.companion_target_x = 0
        self.companion_target_y = 0
        self.companion_path = []
        
        # New system: Single character controlled by both AI and human simultaneously
        self.single_character = True  # Using a single character instead of party
        self.active_player = 0  # Knight is the only character
        # Both AI and human can control the knight simultaneously
        
        # Camera position (for scrolling)
        self.camera_x = max(0, self.companion_x - WIDTH // (2 * TILE_SIZE))
        self.camera_y = max(0, self.companion_y - HEIGHT // (2 * TILE_SIZE))
        
        # Animation frames for player
        self.player_frame = 0
        self.frame_counter = 0
        
        # Battle system
        self.battle_enemy = None
        self.battle_text = ""
        self.battle_options = ["Attack", "Potion", "Run"]
        self.selected_option = 0
        self.battle_cooldown = 0
        self.enemy_types = [
            Enemy("Goblin", 8, 3, 1),
            Enemy("Skeleton", 10, 4, 2),
            Enemy("Orc", 14, 5, 3),
            Enemy("Troll", 18, 6, 4),
            Enemy("Dragon", 25, 8, 5)
        ]
        
        # Encounter system (steps between encounters more variable)
        self.steps_since_encounter = 0
        self.steps_until_next_encounter = self.generate_encounter_steps()
        
        # Game timing
        self.start_time = time.time()
        
        logging.info(f"Player starting at ({self.player_x}, {self.player_y})")
        logging.info(f"Goal at ({self.goal_x}, {self.goal_y})")
        
        # Take a screenshot of the initial game state
        self.draw()
        self.screenshot_mgr.take_screenshot("game_start")
        
        # Start socket server if needed
        if self.socket_server:
            self.socket_server.start()
            
        # Start the game
        try:
            pyxel.run(self.update, self.draw)
        finally:
            # Clean up socket server
            if self.socket_server:
                self.socket_server.stop()
    
    def generate_world(self):
        """Generate a new world"""
        generator = WorldGenerator(self.world_width, self.world_height)
        self.world = generator.generate_terrain()
        self.world = generator.place_potions(self.world, 25)  # More potions for improved experience
        
        # Set player and goal positions
        self.player_x, self.player_y = generator.find_player_start(self.world)
        self.goal_x, self.goal_y = generator.find_goal_position(self.world, (self.player_x, self.player_y))
        
        # In single character mode, we don't need companion positioning
        # We'll keep companion_x and companion_y variables to avoid breaking other code
        # but they won't be used
        self.companion_x, self.companion_y = self.player_x, self.player_y
            
        logging.info(f"Single character (Knight) starting at ({self.player_x}, {self.player_y})")
        
        # Mark goal position
        self.world[self.goal_y][self.goal_x] = GOAL  # Special value for goal
    
    def generate_encounter_steps(self):
        """Generate a random number of steps until the next encounter"""
        # More variable - between 8 and 20 steps
        return random.randint(8, 20)
    
    def update(self):
        # Check for scheduled quit
        if hasattr(self, 'scheduled_quit') and self.scheduled_quit:
            logging.info("Game quit by scheduled command")
            # Take final screenshot before quitting
            self.draw()
            self.screenshot_mgr.take_screenshot("game_quit")
            pyxel.quit()
            return
        
        # Process any CLI commands in the queue
        if hasattr(self, 'command_queue'):
            self.process_command_queue()
        
        # Handle keyboard input if not headless
        if not hasattr(self, 'headless') or not self.headless:
            if pyxel.btnp(pyxel.KEY_Q):
                logging.info("Game quit by player")
                # Take final screenshot before quitting
                self.draw()
                self.screenshot_mgr.take_screenshot("game_quit")
                pyxel.quit()
                return
        
        # Update animation counters
        self.frame_counter += 1
        if self.frame_counter >= 15:  # Change frame every 15 frames
            self.frame_counter = 0
            self.player_frame = (self.player_frame + 1) % 2
            self.companion_frame = (self.companion_frame + 1) % 2
            
        # TAB key no longer needed since there's only one character
        if pyxel.btnp(pyxel.KEY_TAB) and self.game_state == self.EXPLORING:
            logging.info("Only one character in use (Knight)")
            # Take screenshot to show the single character
            self.draw()
            self.screenshot_mgr.take_screenshot("single_character")
        
        # C key now just takes a screenshot to show both AI and human are controlling
        if pyxel.btnp(pyxel.KEY_C) and self.game_state == self.EXPLORING:
            logging.info("Both AI and human are controlling the character simultaneously")
            self.draw()
            self.screenshot_mgr.take_screenshot("simultaneous_control")
            
        # Update game state
        if self.game_state == self.EXPLORING:
            self.update_exploring()
            
            # Update companion AI behavior when not active
            if self.active_player == 0:
                self.update_companion_ai()
                
        elif self.game_state == self.BATTLE:
            self.update_battle()
        elif self.game_state == self.GAME_OVER or self.game_state == self.WIN:
            # Just wait for Q key in these states
            pass
            
        # Log periodic updates
        if pyxel.frame_count % 60 == 0:  # Once per second
            logging.info(f"Player at ({self.player_x}, {self.player_y}), HP: {self.player_hp}/{self.player_max_hp}, Companion at ({self.companion_x}, {self.companion_y}), HP: {self.companion_hp}/{self.companion_max_hp}, Potions: {self.potions}")
            
        # Take screenshots on key presses for debugging
        if pyxel.btnp(pyxel.KEY_S):
            self.screenshot_mgr.take_screenshot("manual")
    
    def toggle_party_control(self):
        """Not needed in new simultaneous control system - but kept for CLI compatibility"""
        # In the new system, both AI and human control the character simultaneously
        logging.info("Both AI and human control the character simultaneously")
        
        # Take screenshot when this function is called
        self.draw()
        self.screenshot_mgr.take_screenshot("simultaneous_control")
    
    def process_command_queue(self):
        """Process commands from the command queue"""
        # Process up to 5 commands per frame to avoid blocking
        for _ in range(min(5, len(self.command_queue))):
            if not self.command_queue:
                break
                
            # Get next command
            cmd_type, cmd_data = self.command_queue.pop(0)
            
            # Process based on command type
            if cmd_type == "move":
                self.handle_move_command(cmd_data)
            elif cmd_type == "action":
                self.handle_action_command(cmd_data)
            elif cmd_type == "switch":
                self.active_player = cmd_data  # Just switch the visual focus
            elif cmd_type == "toggle_party_control":
                self.toggle_party_control()
    
    def handle_move_command(self, direction):
        """Handle move command from CLI"""
        if self.game_state != self.EXPLORING:
            return
            
        # In single character mode, we only move the knight
        # Get current position
        knight_x, knight_y = self.player_x, self.player_y
        
        # Determine direction delta
        dx, dy = 0, 0
        if direction == "up":
            dy = -1
        elif direction == "down":
            dy = 1
        elif direction == "left":
            dx = -1
        elif direction == "right":
            dx = 1
        
        # Calculate new position
        new_knight_x = knight_x + dx
        new_knight_y = knight_y + dy
        
        # Check if the move is valid
        knight_can_move = self.is_walkable(new_knight_x, new_knight_y)
        
        # Move if possible
        if knight_can_move:
            # Check for potions
            self.check_potion_at(new_knight_x, new_knight_y, "Knight")
            
            # Move character
            self.player_x, self.player_y = new_knight_x, new_knight_y
            
            # Update camera position
            self.update_camera_position()
            
            # Check for goal and encounters
            self.check_goal_reached()
            self.steps_since_encounter += 1
            if self.steps_since_encounter >= self.steps_until_next_encounter:
                logging.info(f"CLI: Encounter triggered after {self.steps_since_encounter} steps")
                self.start_battle()
                self.steps_since_encounter = 0
                self.steps_until_next_encounter = self.generate_encounter_steps()
                
            logging.info(f"CLI: Knight moved {direction}")
        else:
            logging.info(f"CLI: Knight could not move {direction}")
    
    def handle_action_command(self, action_type):
        """Handle action command from CLI"""
        if action_type == "force_battle" and self.game_state == self.EXPLORING:
            # Force a battle for debugging
            logging.info("Force battle command received - starting battle")
            self.start_battle()
            return
        elif action_type == "attack" and self.game_state == self.BATTLE:
            # Select and execute Attack option - modify to directly execute attack
            self.selected_option = 0
            logging.info("CLI Attack command received - executing attack")
            self.execute_battle_action("Attack")
        elif action_type == "potion":
            if self.game_state == self.EXPLORING:
                # Use potion outside battle
                self.use_potion_exploring()
            elif self.game_state == self.BATTLE:
                # Select and execute Potion option
                self.selected_option = 1
                logging.info("CLI Potion command received - executing potion use")
                self.execute_battle_action("Potion")
        elif action_type == "run" and self.game_state == self.BATTLE:
            # Select and execute Run option
            self.selected_option = 2
            logging.info("CLI Run command received - executing run")
            self.execute_battle_action("Run")
    
    def use_potion_exploring(self):
        """Use a potion outside of battle"""
        if self.potions <= 0:
            logging.info("Tried to use potion but had none")
            return
            
        # Determine who needs healing
        knight_missing_hp = self.player_max_hp - self.player_hp
        wizard_missing_hp = self.companion_max_hp - self.companion_hp
        
        # If both at full health, don't use
        if knight_missing_hp <= 0 and wizard_missing_hp <= 0:
            logging.info("Tried to use potion but both characters at full HP")
            return
            
        # Use potion
        self.potions -= 1
        
        # Heal both characters
        knight_heal_amount = int(self.player_max_hp * 0.3)  # 30% for knight
        wizard_heal_amount = int(self.companion_max_hp * 0.3)  # 30% for wizard
        
        # Apply healing
        old_knight_hp = self.player_hp
        self.player_hp = min(self.player_max_hp, self.player_hp + knight_heal_amount)
        knight_actual_heal = self.player_hp - old_knight_hp
        
        old_wizard_hp = self.companion_hp
        self.companion_hp = min(self.companion_max_hp, self.companion_hp + wizard_heal_amount)
        wizard_actual_heal = self.companion_hp - old_wizard_hp
        
        logging.info(f"Used a potion outside battle. Knight +{knight_actual_heal} HP, Wizard +{wizard_actual_heal} HP. Potions left: {self.potions}")
        
        # Take screenshot
        self.draw()
        self.screenshot_mgr.take_screenshot("used_potion_outside_battle")
    
    def move_active_player(self, dx, dy):
        """Move the currently active player by the given delta"""
        if self.active_player == 0:
            # Move knight
            new_x = self.player_x + dx
            new_y = self.player_y + dy
            
            # Check if the new position is valid (walkable)
            if self.is_walkable(new_x, new_y) and not (new_x == self.companion_x and new_y == self.companion_y):
                # Check for potion
                if self.world[new_y][new_x] == POTION:
                    self.potions += 1
                    logging.info(f"Player found a potion at ({new_x}, {new_y}). Total: {self.potions}")
                    self.world[new_y][new_x] = GRASS  # Remove potion from map
                    
                    # Take screenshot when finding a potion
                    self.draw()
                    self.screenshot_mgr.take_screenshot("player_found_potion")
                
                self.player_x, self.player_y = new_x, new_y
                
                # Update camera position to keep player centered
                self.update_camera_position()
                
                # Check if either character reached the goal
                self.check_goal_reached()
                
                # Check for random encounters if moved
                self.steps_since_encounter += 1
                if self.steps_since_encounter >= self.steps_until_next_encounter:
                    logging.info(f"Encounter triggered after {self.steps_since_encounter} steps")
                    self.start_battle()
                    self.steps_since_encounter = 0
                    self.steps_until_next_encounter = self.generate_encounter_steps()
        else:
            # Move wizard (if not AI controlled)
            if not self.ai_controlled:
                new_x = self.companion_x + dx
                new_y = self.companion_y + dy
                
                # Check if the new position is valid (walkable)
                if self.is_walkable(new_x, new_y) and not (new_x == self.player_x and new_y == self.player_y):
                    # Check for potion
                    if self.world[new_y][new_x] == POTION:
                        self.potions += 1
                        logging.info(f"Companion found a potion at ({new_x}, {new_y}). Total: {self.potions}")
                        self.world[new_y][new_x] = GRASS  # Remove potion from map
                        
                        # Take screenshot when finding a potion
                        self.draw()
                        self.screenshot_mgr.take_screenshot("companion_found_potion")
                    
                    self.companion_x, self.companion_y = new_x, new_y
                    
                    # Update camera position to keep active character centered
                    self.update_camera_position()
                    
                    # Check if either character reached the goal
                    self.check_goal_reached()
                    
                    # Check for random encounters if moved
                    self.steps_since_encounter += 1
                    if self.steps_since_encounter >= self.steps_until_next_encounter:
                        logging.info(f"Encounter triggered after {self.steps_since_encounter} steps")
                        self.start_battle()
                        self.steps_since_encounter = 0
                        self.steps_until_next_encounter = self.generate_encounter_steps()
    
    def switch_to_player(self, player_idx):
        """Switch to specific player (0=knight, 1=wizard)"""
        if player_idx == self.active_player:
            return  # Already active
            
        self.active_player = player_idx
        
        # When switching to wizard, enable AI by default
        if player_idx == 1:
            self.ai_controlled = True
            logging.info("Switched to Claude AI controlling the Wizard")
        else:
            # When switching to knight, disable AI
            self.ai_controlled = False
            logging.info("Switched to Human Player controlling the Knight")
            
        # Take screenshot when switching players
        self.draw()
        self.screenshot_mgr.take_screenshot(f"switch_to_{'companion' if self.active_player == 1 else 'player'}")
    
    def toggle_ai_control(self):
        """Toggle AI control for wizard character"""
        self.ai_controlled = not self.ai_controlled
        logging.info(f"AI control is now {'enabled' if self.ai_controlled else 'disabled'} for Claude's character")
        self.draw()
        self.screenshot_mgr.take_screenshot(f"ai_control_{'enabled' if self.ai_controlled else 'disabled'}")
    
    def update_companion_ai(self):
        """Handle AI companion behavior when controlled by Claude"""
        # Only move every few frames to avoid moving too fast
        self.companion_move_timer += 1
        if self.companion_move_timer < 20:  # Move slightly faster (every 1/3 second)
            return
            
        self.companion_move_timer = 0
        
        # Track time since last major move
        time_since_last_move = pyxel.frame_count % 120  # Cycle behavior every 2 seconds
        
        # Get distance to player for coordination
        distance_to_player = abs(self.companion_x - self.player_x) + abs(self.companion_y - self.player_y)
        
        # Look for nearby potions (higher priority)
        potion_found = False
        search_radius = 5  # Larger search radius
        for y in range(max(0, self.companion_y - search_radius), min(self.world_height, self.companion_y + search_radius + 1)):
            for x in range(max(0, self.companion_x - search_radius), min(self.world_width, self.companion_x + search_radius + 1)):
                if self.world[y][x] == POTION:
                    # Log AI decision making
                    logging.info(f"Claude AI: Moving toward potion at ({x}, {y})")
                    
                    # Move toward potion
                    self.move_companion_toward(x, y)
                    potion_found = True
                    break
            if potion_found:
                break
        
        if potion_found:
            return
        
        # Determine AI strategy based on situation
        if distance_to_player > 7:
            # Too far away - move toward player
            logging.info(f"Claude AI: Moving toward player (distance: {distance_to_player})")
            self.move_companion_toward(self.player_x, self.player_y)
            
        elif distance_to_player < 2:
            # Too close - create tactical space
            # Choose direction strategically (toward goal or unexplored area)
            player_to_goal_x = self.goal_x - self.player_x
            player_to_goal_y = self.goal_y - self.player_y
            
            # Try to position based on goal direction
            tactical_x = self.player_x + (1 if player_to_goal_x > 0 else -1)
            tactical_y = self.player_y + (1 if player_to_goal_y > 0 else -1)
            
            logging.info(f"Claude AI: Creating tactical spacing")
            self.move_companion_toward(tactical_x, tactical_y)
            
        else:
            # Good tactical distance - implement different strategies
            
            # Every 10 seconds, try moving toward goal
            if time_since_last_move < 20:
                goal_distance = abs(self.companion_x - self.goal_x) + abs(self.companion_y - self.goal_y)
                if goal_distance > 10:
                    # Move toward goal if far away
                    logging.info(f"Claude AI: Moving toward goal (distance: {goal_distance})")
                    self.move_companion_toward(self.goal_x, self.goal_y)
                else:
                    # When close to goal, scout around for potions
                    logging.info(f"Claude AI: Scouting around goal")
                    self.explore_nearest_unexplored()
            
            # Otherwise, explore methodically based on terrain
            elif time_since_last_move < 60:
                # Check current terrain for strategic advantage
                current_terrain = self.world[self.companion_y][self.companion_x]
                
                if current_terrain == FOREST:
                    # Wizards move better in forests - look for more forest
                    self.find_and_move_to_terrain(FOREST)
                elif current_terrain == PATH:
                    # Paths lead to interesting places
                    self.follow_path()
                else:
                    # Default: explore in spiral pattern around player
                    angle = pyxel.frame_count * 0.05
                    radius = min(5, distance_to_player)
                    target_x = self.player_x + int(radius * pyxel.cos(angle))
                    target_y = self.player_y + int(radius * pyxel.sin(angle))
                    
                    # Keep within bounds
                    target_x = max(1, min(target_x, self.world_width - 2))
                    target_y = max(1, min(target_y, self.world_height - 2))
                    
                    logging.info(f"Claude AI: Patrolling around player")
                    self.move_companion_toward(target_x, target_y)
            else:
                # Sometimes just move randomly to explore
                logging.info(f"Claude AI: Exploring randomly")
                directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
                random.shuffle(directions)
                
                for dx, dy in directions:
                    new_x = self.companion_x + dx
                    new_y = self.companion_y + dy
                    
                    if self.is_walkable(new_x, new_y) and not (new_x == self.player_x and new_y == self.player_y):
                        self.companion_x, self.companion_y = new_x, new_y
                        break
    
    def explore_nearest_unexplored(self):
        """Find and move toward the nearest unexplored area"""
        # Simple implementation: move toward areas we haven't been yet
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)
        
        # Prefer directions we haven't been in
        for dx, dy in directions:
            look_x = self.companion_x + dx * 2
            look_y = self.companion_y + dy * 2
            
            if (0 <= look_x < self.world_width and 
                0 <= look_y < self.world_height and
                self.is_walkable(look_x, look_y)):
                self.move_companion_toward(look_x, look_y)
                return
                
        # If all explored, move randomly
        if self.is_walkable(self.companion_x + directions[0][0], self.companion_y + directions[0][1]):
            self.companion_x += directions[0][0]
            self.companion_y += directions[0][1]
    
    def find_and_move_to_terrain(self, terrain_type):
        """Find nearby terrain of the specified type and move toward it"""
        # Look in a spiral pattern
        for radius in range(1, 6):
            for angle in range(0, 360, 45):
                rad_angle = angle * 3.14159 / 180
                test_x = self.companion_x + int(radius * pyxel.cos(rad_angle))
                test_y = self.companion_y + int(radius * pyxel.sin(rad_angle))
                
                if (0 <= test_x < self.world_width and 
                    0 <= test_y < self.world_height and
                    self.world[test_y][test_x] == terrain_type):
                    self.move_companion_toward(test_x, test_y)
                    return
        
        # If not found, default to normal movement
        self.move_companion_toward(self.player_x, self.player_y)
    
    def follow_path(self):
        """Follow a path if the character is on one"""
        # Check if we're on a path
        if self.world[self.companion_y][self.companion_x] != PATH:
            return False
            
        # Check all directions for continuing path
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        path_directions = []
        
        for dx, dy in directions:
            test_x = self.companion_x + dx
            test_y = self.companion_y + dy
            
            if (0 <= test_x < self.world_width and 
                0 <= test_y < self.world_height and
                self.world[test_y][test_x] == PATH):
                path_directions.append((dx, dy))
        
        if path_directions:
            # Choose a path direction, preferring one we haven't been on
            dx, dy = random.choice(path_directions)
            self.companion_x += dx
            self.companion_y += dy
            return True
            
        return False
    
    def move_companion_toward(self, target_x, target_y):
        """Move companion one step toward the target coordinates"""
        # Simple pathfinding toward target
        dx = 0
        dy = 0
        
        if self.companion_x < target_x:
            dx = 1
        elif self.companion_x > target_x:
            dx = -1
            
        if self.companion_y < target_y:
            dy = 1
        elif self.companion_y > target_y:
            dy = -1
            
        # Try horizontal movement first
        if dx != 0 and self.is_walkable(self.companion_x + dx, self.companion_y):
            self.companion_x += dx
            return
            
        # Try vertical movement
        if dy != 0 and self.is_walkable(self.companion_x, self.companion_y + dy):
            self.companion_y += dy
            return
            
        # Try diagonal as last resort
        if dx != 0 and dy != 0 and self.is_walkable(self.companion_x + dx, self.companion_y + dy):
            self.companion_x += dx
            self.companion_y += dy
    
    def is_walkable(self, x, y):
        """Check if a position is walkable"""
        if not (0 <= x < self.world_width and 0 <= y < self.world_height):
            return False
        
        terrain = self.world[y][x]
        return terrain not in [WATER, MOUNTAIN, WALL]
    
    def update_exploring(self):
        # In the new system, we process both human and AI inputs every frame
        # This allows simultaneous control by both AI and human
        
        # Process human input first
        self.update_human_input()
        
        # Then process AI input (on same character)
        self.update_ai_input()
        
        # Check for encounters regardless of who moved
        if self.steps_since_encounter >= self.steps_until_next_encounter:
            logging.info(f"Encounter triggered after {self.steps_since_encounter} steps")
            self.start_battle()
            self.steps_since_encounter = 0
            self.steps_until_next_encounter = self.generate_encounter_steps()
            
    def update_human_input(self):
        """Process human keyboard input for character movement"""
        new_x, new_y = self.player_x, self.player_y
        moved = False
        
        # Get movement direction from keyboard
        if pyxel.btnp(pyxel.KEY_UP) or pyxel.btnp(pyxel.KEY_W):
            new_y -= 1
            moved = True
        elif pyxel.btnp(pyxel.KEY_DOWN) or pyxel.btnp(pyxel.KEY_S):
            new_y += 1
            moved = True
        elif pyxel.btnp(pyxel.KEY_LEFT) or pyxel.btnp(pyxel.KEY_A):
            new_x -= 1
            moved = True
        elif pyxel.btnp(pyxel.KEY_RIGHT) or pyxel.btnp(pyxel.KEY_D):
            new_x += 1
            moved = True
        
        # Process movement if user pressed a key
        if moved:
            # Check if new position is valid
            if self.is_walkable(new_x, new_y):
                # Check for potions
                self.check_potion_at(new_x, new_y, "Knight")
                
                # Move character
                self.player_x, self.player_y = new_x, new_y
                
                # Keep the companion position in sync (not used but maintained for compatibility)
                self.companion_x, self.companion_y = self.player_x, self.player_y
                
                # Update camera position
                self.update_camera_position()
                
                # Check if reached the goal
                self.check_goal_reached()
                
                # Count step for encounter
                self.steps_since_encounter += 1
                
                logging.info(f"Human moved Knight to ({self.player_x}, {self.player_y})")
            else:
                logging.info(f"Human tried to move Knight to ({new_x}, {new_y}) but path is blocked")
    
    def check_potion_at(self, x, y, character_name):
        """Check if there's a potion at position and collect it"""
        if self.world[y][x] == POTION:
            self.potions += 1
            logging.info(f"{character_name} found a potion at ({x}, {y}). Total: {self.potions}")
            self.world[y][x] = GRASS  # Remove potion from map
            
            # Take screenshot when finding a potion
            self.draw()
            self.screenshot_mgr.take_screenshot(f"{character_name.lower()}_found_potion")
            return True
        return False
            
    def update_ai_input(self):
        """AI provides input for the knight character
        
        This allows the AI to suggest moves independently from
        human input, creating collaborative control.
        """
        # Only update AI movement occasionally to avoid moving too fast
        self.companion_move_timer += 1
        if self.companion_move_timer < 45:  # Slower frequency than human input
            return
            
        self.companion_move_timer = 0
        
        # Determine best goal for AI
        ai_goal_x, ai_goal_y = self.determine_ai_goal()
        
        # AI attempts to move the character toward its chosen goal
        self.attempt_ai_move(ai_goal_x, ai_goal_y)
        
    def determine_ai_goal(self):
        """Determine the best goal for AI movement based on current situation"""
        # Priorities:
        # 1. Nearby potions if health is low
        # 2. The final goal if health is good
        # 3. Unexplored areas
        
        # Check if health is low
        knight_health_pct = self.player_hp / self.player_max_hp
        low_health = knight_health_pct < 0.5
        
        # If health is low, look for potions
        if low_health and self.potions == 0:
            # Search for nearby potions in a larger radius
            search_radius = 8
            for y in range(max(0, self.player_y - search_radius), 
                          min(self.world_height, self.player_y + search_radius + 1)):
                for x in range(max(0, self.player_x - search_radius), 
                              min(self.world_width, self.player_x + search_radius + 1)):
                    if self.world[y][x] == POTION:
                        logging.info(f"AI: Suggesting to move toward potion at ({x}, {y}) due to low health")
                        return x, y
        
        # If health is good or we have potions, head toward the main goal
        potion_count_ok = self.potions > 0
        health_ok = knight_health_pct >= 0.5
        
        if health_ok or potion_count_ok:
            # Head toward the main goal
            logging.info(f"AI: Suggesting to move toward main goal at ({self.goal_x}, {self.goal_y})")
            return self.goal_x, self.goal_y
            
        # Default: explore in a somewhat random pattern
        angle = pyxel.frame_count * 0.05
        radius = 8
        explore_x = self.player_x + int(radius * pyxel.cos(angle))
        explore_y = self.player_y + int(radius * pyxel.sin(angle))
        
        # Keep within bounds
        explore_x = max(1, min(explore_x, self.world_width - 2))
        explore_y = max(1, min(explore_y, self.world_height - 2))
        
        logging.info(f"AI: Suggesting exploration around ({explore_x}, {explore_y})")
        return explore_x, explore_y
    
    def attempt_ai_move(self, target_x, target_y):
        """AI attempts to move the character one step toward the target position
        
        This function allows the AI to suggest a move independently from human input,
        creating a collaborative control system.
        """
        # Determine direction to move
        dx = 0
        dy = 0
        
        if self.player_x < target_x:
            dx = 1
        elif self.player_x > target_x:
            dx = -1
            
        if self.player_y < target_y:
            dy = 1
        elif self.player_y > target_y:
            dy = -1
            
        # If diagonal movement, choose horizontal or vertical randomly
        if dx != 0 and dy != 0:
            if random.random() < 0.5:
                dy = 0
            else:
                dx = 0
                
        # Calculate new position
        new_x = self.player_x + dx
        new_y = self.player_y + dy
        
        # Check if character can move to the new position
        can_move = self.is_walkable(new_x, new_y)
        
        # If can move, attempt to move
        if can_move:
            # Check for potions
            self.check_potion_at(new_x, new_y, "Knight")
            
            # Move character
            self.player_x, self.player_y = new_x, new_y
            
            # Keep companion position in sync (for compatibility)
            self.companion_x, self.companion_y = self.player_x, self.player_y
            
            # Update camera
            self.update_camera_position()
            
            # Check goal
            self.check_goal_reached()
            
            # Count step for encounter
            self.steps_since_encounter += 1
            
            logging.info(f"AI moved Knight to ({self.player_x}, {self.player_y})")
        else:
            # Try a different direction if blocked
            # First try alternative horizontal/vertical move
            if dx != 0 and dy == 0:
                # Try vertical
                if self.player_y < target_y:
                    new_y = self.player_y + 1
                else:
                    new_y = self.player_y - 1
                new_x = self.player_x
            elif dy != 0 and dx == 0:
                # Try horizontal
                if self.player_x < target_x:
                    new_x = self.player_x + 1
                else:
                    new_x = self.player_x - 1
                new_y = self.player_y
                
            # Check if this alternative move is valid
            if (new_x != self.player_x or new_y != self.player_y) and self.is_walkable(new_x, new_y):
                # Check for potions
                self.check_potion_at(new_x, new_y, "Knight")
                
                # Move character
                self.player_x, self.player_y = new_x, new_y
                
                # Keep companion position in sync (for compatibility)
                self.companion_x, self.companion_y = self.player_x, self.player_y
                
                # Update camera
                self.update_camera_position()
                
                # Check goal
                self.check_goal_reached()
                
                # Count step for encounter
                self.steps_since_encounter += 1
                
                logging.info(f"AI found alternative path for Knight to ({self.player_x}, {self.player_y})")
            else:
                logging.info(f"AI couldn't find a valid move toward ({target_x}, {target_y})")
    
    def update_player_movement(self):
        """Handle human player movement for the knight character"""
        new_x, new_y = self.player_x, self.player_y
        moved = False
        
        if pyxel.btnp(pyxel.KEY_UP) or pyxel.btnp(pyxel.KEY_W):
            new_y -= 1
            moved = True
        elif pyxel.btnp(pyxel.KEY_DOWN) or pyxel.btnp(pyxel.KEY_S):
            new_y += 1
            moved = True
        elif pyxel.btnp(pyxel.KEY_LEFT) or pyxel.btnp(pyxel.KEY_A):
            new_x -= 1
            moved = True
        elif pyxel.btnp(pyxel.KEY_RIGHT) or pyxel.btnp(pyxel.KEY_D):
            new_x += 1
            moved = True
            
        # Check if the new position is valid (walkable)
        if moved and self.is_walkable(new_x, new_y) and not (new_x == self.companion_x and new_y == self.companion_y):
            # Check for potion
            if self.world[new_y][new_x] == POTION:
                self.potions += 1
                logging.info(f"Player found a potion at ({new_x}, {new_y}). Total: {self.potions}")
                self.world[new_y][new_x] = GRASS  # Remove potion from map
                
                # Take screenshot when finding a potion
                self.draw()
                self.screenshot_mgr.take_screenshot("player_found_potion")
            
            self.player_x, self.player_y = new_x, new_y
            
            # Update camera position to keep player centered
            self.update_camera_position()
            
            # Check if either character reached the goal
            self.check_goal_reached()
            
            # Check for random encounters if moved
            self.steps_since_encounter += 1
            if self.steps_since_encounter >= self.steps_until_next_encounter:
                logging.info(f"Encounter triggered after {self.steps_since_encounter} steps")
                self.start_battle()
                self.steps_since_encounter = 0
                self.steps_until_next_encounter = self.generate_encounter_steps()
                
    def update_companion_movement(self):
        """Handle human player movement for Claude's wizard character"""
        new_x, new_y = self.companion_x, self.companion_y
        moved = False
        
        if pyxel.btnp(pyxel.KEY_UP) or pyxel.btnp(pyxel.KEY_W):
            new_y -= 1
            moved = True
        elif pyxel.btnp(pyxel.KEY_DOWN) or pyxel.btnp(pyxel.KEY_S):
            new_y += 1
            moved = True
        elif pyxel.btnp(pyxel.KEY_LEFT) or pyxel.btnp(pyxel.KEY_A):
            new_x -= 1
            moved = True
        elif pyxel.btnp(pyxel.KEY_RIGHT) or pyxel.btnp(pyxel.KEY_D):
            new_x += 1
            moved = True
            
        # Check if the new position is valid (walkable)
        if moved and self.is_walkable(new_x, new_y) and not (new_x == self.player_x and new_y == self.player_y):
            # Check for potion
            if self.world[new_y][new_x] == POTION:
                self.potions += 1
                logging.info(f"Companion found a potion at ({new_x}, {new_y}). Total: {self.potions}")
                self.world[new_y][new_x] = GRASS  # Remove potion from map
                
                # Take screenshot when finding a potion
                self.draw()
                self.screenshot_mgr.take_screenshot("companion_found_potion")
            
            self.companion_x, self.companion_y = new_x, new_y
            
            # Update camera position to keep active character centered
            self.update_camera_position()
            
            # Check if either character reached the goal
            self.check_goal_reached()
            
            # Check for random encounters if moved
            self.steps_since_encounter += 1
            if self.steps_since_encounter >= self.steps_until_next_encounter:
                logging.info(f"Encounter triggered after {self.steps_since_encounter} steps")
                self.start_battle()
                self.steps_since_encounter = 0
                self.steps_until_next_encounter = self.generate_encounter_steps()
    
    def update_camera_position(self):
        """Update camera position to keep the character centered"""
        # In single character mode, we only track player_x and player_y
        self.camera_x = max(0, min(self.player_x - WIDTH // (2 * TILE_SIZE), 
                                  self.world_width - WIDTH // TILE_SIZE))
        self.camera_y = max(0, min(self.player_y - HEIGHT // (2 * TILE_SIZE), 
                                  self.world_height - HEIGHT // TILE_SIZE))
    
    def check_goal_reached(self):
        """Check if character has reached the goal"""
        if self.player_x == self.goal_x and self.player_y == self.goal_y:
            logging.info("Goal reached! Game won through human-AI collaboration.")
            self.game_state = self.WIN
            
            # Take screenshot when winning - note the collaboration
            self.draw()
            self.screenshot_mgr.take_screenshot("game_won_collaborative")
            return True
        return False
    
    def start_battle(self):
        # Store previous state to allow for transition effects
        prev_state = self.game_state
        self.game_state = self.BATTLE
        
        # Add transition effect properties
        self.battle_transition_frame = 0
        self.battle_transition_max_frames = 30
        self.battle_flash_count = 0
        self.battle_shake_intensity = 5
        
        # Take a screenshot before battle starts
        self.draw()
        self.screenshot_mgr.take_screenshot("before_battle_transition")
        
        # Pick a random enemy (weighted by current terrain type)
        # Use position of active character to determine terrain
        if self.active_player == 0:
            active_x, active_y = self.player_x, self.player_y
        else:
            active_x, active_y = self.companion_x, self.companion_y
            
        current_terrain = self.world[active_y][active_x]
        
        if current_terrain == FOREST:
            # More likely to encounter Goblin in forests
            weights = [0.5, 0.2, 0.2, 0.1, 0.0]
            terrain_msg = " lurking among the trees!"
        elif current_terrain == MOUNTAIN:
            # More likely to encounter Troll in mountains
            weights = [0.1, 0.2, 0.2, 0.4, 0.1] 
            terrain_msg = " on the rocky mountainside!"
        elif current_terrain == CASTLE:
            # More likely to encounter Dragons near castles
            weights = [0.0, 0.2, 0.2, 0.2, 0.4]
            terrain_msg = " defending the castle!"
        elif current_terrain == SAND:
            # Desert encounters
            weights = [0.2, 0.4, 0.3, 0.1, 0.0]
            terrain_msg = " emerging from the sand!"
        elif current_terrain == PATH:
            # Path encounters - more bandits on roads
            weights = [0.4, 0.3, 0.2, 0.1, 0.0]
            terrain_msg = " blocking your path!"
        else:
            # Default weights
            weights = [0.3, 0.3, 0.2, 0.15, 0.05]
            terrain_msg = " appears!"
            
        # Choose enemy based on weights
        enemy_idx = random.choices(range(len(self.enemy_types)), weights=weights)[0]
        enemy_template = self.enemy_types[enemy_idx]
        
        # Small variation in enemy stats (±10%)
        hp_mod = random.uniform(0.9, 1.1)
        atk_mod = random.uniform(0.9, 1.1)
        def_mod = random.uniform(0.9, 1.1)
        
        self.battle_enemy = Enemy(
            enemy_template.name,
            max(1, int(enemy_template.hp * hp_mod)),
            max(1, int(enemy_template.attack * atk_mod)),
            max(1, int(enemy_template.defense * def_mod))
        )
        
        # In co-op mode, battles are easier since both characters are involved
        self.battle_enemy.hp = int(self.battle_enemy.hp * 0.8)  # Reduce enemy HP slightly
        
        logging.info(f"Battle started with {self.battle_enemy.name} - Player: {self.player_hp}/{self.player_max_hp} HP, Companion: {self.companion_hp}/{self.companion_max_hp} HP")
        self.battle_text = f"A {self.battle_enemy.name}{terrain_msg}"
        self.selected_option = 0
        self.battle_cooldown = 30  # Half a second delay before accepting input
        
        # Take screenshot when battle starts
        self.draw()
        self.screenshot_mgr.take_screenshot(f"battle_start_{self.battle_enemy.name}")
    
    def execute_battle_action(self, option):
        """Execute a battle action directly (for CLI or keyboard input)"""
        if self.battle_cooldown > 0:
            logging.info(f"Battle on cooldown ({self.battle_cooldown} frames remaining), ignoring action")
            return
        
        # Take a screenshot before the action for debugging
        if hasattr(self, 'screenshot_mgr'):
            self.draw()
            self.screenshot_mgr.take_screenshot(f"before_{option.lower()}")
            
        # Process the option
        if option == "Attack":
            self._process_attack()
        elif option == "Potion":
            self._process_potion_use()
        elif option == "Run":
            self._process_run_attempt()
            
        # Take a screenshot after the action
        if hasattr(self, 'screenshot_mgr'):
            self.draw()
            self.screenshot_mgr.take_screenshot(f"after_{option.lower()}")
            
    def update_battle(self):
        if self.battle_cooldown > 0:
            self.battle_cooldown -= 1
            return
            
        # Handle menu navigation
        if pyxel.btnp(pyxel.KEY_UP) or pyxel.btnp(pyxel.KEY_W):
            self.selected_option = (self.selected_option - 1) % len(self.battle_options)
        elif pyxel.btnp(pyxel.KEY_DOWN) or pyxel.btnp(pyxel.KEY_S):
            self.selected_option = (self.selected_option + 1) % len(self.battle_options)
            
        # Handle selection
        if pyxel.btnp(pyxel.KEY_SPACE) or pyxel.btnp(pyxel.KEY_RETURN):
            option = self.battle_options[self.selected_option]
            self.execute_battle_action(option)
            
            # This is now handled by execute_battle_action
            pass
            
    def _process_attack(self):
        """Process an attack action in battle"""
        # In the single-character model, only knight attacks
        primary_attacker = "Knight"
        primary_attack = max(1, self.player_attack - self.battle_enemy.defense)
        
        # Check for critical hit (10% chance)
        is_critical = random.random() < 0.1
        
        if is_critical:
            # Critical hits do 150% damage
            primary_damage = int(primary_attack * 1.5)
            primary_hit_text = f"CRITICAL HIT! Your {primary_attacker} hits for {primary_damage} damage!"
        else:
            # Normal hit with small variation (±20%)
            variation = random.uniform(0.8, 1.2)
            primary_damage = max(1, int(primary_attack * variation))
            primary_hit_text = f"Your {primary_attacker} hits for {primary_damage} damage!"
        
        # Apply damage
        self.battle_enemy.hp -= primary_damage
        
        # Construct battle text
        self.battle_text = primary_hit_text
        logging.info(f"Single-character attack: {primary_attacker} hit for {primary_damage}" + 
                     (" (critical)" if is_critical else ""))
        
        # Take a screenshot of the attack
        self.draw()
        self.screenshot_mgr.take_screenshot("attack")
        
        # Check if enemy is defeated
        if self.battle_enemy.hp <= 0:
            self.battle_text = f"You defeated the {self.battle_enemy.name}!"
            logging.info(f"Knight defeated {self.battle_enemy.name}")
            
            # Take screenshot when defeating an enemy
            self.draw()
            self.screenshot_mgr.take_screenshot(f"defeated_{self.battle_enemy.name}")
            
            # Rewards for victory
            reward_roll = random.random()
            
            # 30% chance to get a potion
            if reward_roll < 0.3:
                self.potions += 1
                logging.info(f"Found a potion from defeated enemy. Total: {self.potions}")
                self.battle_text += "\nFound a potion!"
            # 20% chance for knight to gain +1 attack
            elif reward_roll < 0.5:
                self.player_attack += 1
                logging.info(f"Knight attack increased to {self.player_attack}")
                self.battle_text += "\nKnight's attack increased by 1!"
            # 20% chance for knight to gain +1 defense
            elif reward_roll < 0.7:
                self.player_defense += 1
                logging.info(f"Knight defense increased to {self.player_defense}")
                self.battle_text += "\nKnight's defense increased by 1!"
            # 30% chance to heal character
            else:
                heal_amount = int(self.player_max_hp * 0.3)  # Heal 30% of max HP
                
                old_player_hp = self.player_hp
                self.player_hp = min(self.player_max_hp, self.player_hp + heal_amount)
                player_heal = self.player_hp - old_player_hp
                
                logging.info(f"Knight healed: +{player_heal} HP")
                self.battle_text += f"\nYou feel refreshed! (+{player_heal} HP)"
                
            self.battle_cooldown = 60  # 1 second delay
            self.game_state = self.EXPLORING
            return
            
        # Enemy attacks
        self.enemy_attack()
    
    def _process_potion_use(self):
        """Process a potion use action in battle"""
        # Use potion if available
        if self.potions > 0:
            # Check if at full health
            knight_missing_hp = self.player_max_hp - self.player_hp
            
            # If at full health, show message
            if knight_missing_hp <= 0:
                self.battle_text = "Knight is already at full HP!"
                logging.info("Tried to use potion but Knight is at full HP")
                return
                
            # Otherwise use the potion
            self.potions -= 1
            
            # Calculate healing
            knight_heal_amount = int(self.player_max_hp * 0.3)  # 30% for knight
            
            # Apply healing
            old_knight_hp = self.player_hp
            self.player_hp = min(self.player_max_hp, self.player_hp + knight_heal_amount)
            knight_actual_heal = self.player_hp - old_knight_hp
            
            # Build message
            self.battle_text = f"Used a potion! Knight +{knight_actual_heal} HP."
                
            logging.info(f"Used a potion. Knight HP: {self.player_hp}/{self.player_max_hp}, Potions left: {self.potions}")
            
            # Take screenshot when using a potion
            self.draw()
            self.screenshot_mgr.take_screenshot("used_potion")
            
            # 30% chance to avoid enemy's attack (quick potion)
            if random.random() < 0.3:
                self.battle_text += "\nYou quickly used the potion and dodged the enemy's attack!"
                logging.info("Knight dodged enemy attack after using potion")
            else:
                # Enemy still gets an attack
                self.enemy_attack()
        else:
            # No potions left
            self.battle_text = "No potions left!"
            logging.info("Tried to use potion but had none")
    
    def _process_run_attempt(self):
        """Process a run attempt action in battle"""
        # Escape chance varies based on terrain
        base_escape_chance = 0.6  # 60% base chance
        
        # Forest and PATH terrain make it easier to escape
        if self.world[self.player_y][self.player_x] == FOREST:
            escape_chance = base_escape_chance + 0.2  # 80% in forest
            escape_msg = "You vanish into the trees!"
        elif self.world[self.player_y][self.player_x] == PATH:
            escape_chance = base_escape_chance + 0.1  # 70% on paths
            escape_msg = "You sprint away along the path!"
        else:
            escape_chance = base_escape_chance
            escape_msg = "You escaped!"
        
        # Stronger enemies are harder to escape from
        if self.battle_enemy.name == "Dragon":
            escape_chance -= 0.2  # -20% with dragons
        elif self.battle_enemy.name == "Troll":
            escape_chance -= 0.1  # -10% with trolls
        
        if random.random() < escape_chance:
            self.battle_text = escape_msg
            logging.info(f"Knight successfully escaped from battle (chance: {escape_chance:.0%})")
            
            # Take screenshot when escaping
            self.draw()
            self.screenshot_mgr.take_screenshot("escaped_battle")
            
            self.battle_cooldown = 30  # Half a second delay
            self.game_state = self.EXPLORING
        else:
            # Different failure messages based on enemy
            if self.battle_enemy.name == "Dragon":
                fail_msg = "The Dragon's wings block your escape!"
            elif self.battle_enemy.name == "Troll":
                fail_msg = "The Troll's long arms catch you!"
            else:
                fail_msg = "Couldn't escape!"
            
            self.battle_text = fail_msg
            logging.info(f"Knight failed to escape from battle (chance: {escape_chance:.0%})")
            
            # Enemy gets a free attack
            self.enemy_attack()
    
    def enemy_attack(self):
        """Handle enemy attacking the player character"""
        # In single character mode, the enemy only attacks the Knight
        
        # Check for dodge based on terrain
        knight_terrain = self.world[self.player_y][self.player_x]
        
        # Calculate dodge chance based on terrain
        dodge_chance = 0
        
        # Forest gives advantage in dodging
        if knight_terrain == FOREST:
            dodge_chance = 0.15  # 15% chance to dodge in forest
        # Path gives a small dodge chance
        elif knight_terrain == PATH:
            dodge_chance = 0.1   # 10% chance to dodge on path
        
        # Determine if dodge is successful
        knight_dodged = random.random() < dodge_chance
        
        # Calculate damage with a small random variation
        base_damage = max(1, self.battle_enemy.attack - self.player_defense)
        variation = random.uniform(0.8, 1.2)  # Damage varies by ±20%
        damage = max(1, int(base_damage * variation))
        
        # Apply damage and build text
        battle_message = ""
        
        if knight_dodged:
            battle_message += f"\nKnight dodges the {self.battle_enemy.name}'s attack!"
            logging.info(f"Knight dodged {self.battle_enemy.name}'s attack")
        else:
            self.player_hp = max(0, self.player_hp - damage)
            battle_message += f"\n{self.battle_enemy.name} hits Knight for {damage} damage!"
            logging.info(f"{self.battle_enemy.name} hit Knight for {damage} damage. HP: {self.player_hp}/{self.player_max_hp}")
        
        # Add message to battle text
        self.battle_text += battle_message
        
        # Check if character is defeated
        if self.player_hp <= 0:
            self.player_hp = 0
            self.battle_text = "Knight was defeated!"
            logging.info("Knight was defeated!")
            
            # Take screenshot on defeat
            self.draw()
            self.screenshot_mgr.take_screenshot("knight_defeated")
            
            self.battle_cooldown = 60  # 1 second delay
            self.game_state = self.GAME_OVER
    
    def get_terrain_color(self, terrain_type):
        """Get the color for a terrain type"""
        if terrain_type == WATER:
            return WATER_COLOR
        elif terrain_type == SAND:
            return SAND_COLOR
        elif terrain_type == GRASS:
            return GRASS_COLOR
        elif terrain_type == FOREST:
            return TREE_COLOR
        elif terrain_type == MOUNTAIN:
            return MOUNTAIN_COLOR
        elif terrain_type == WALL:
            return WALL_COLOR
        elif terrain_type == POTION:
            return POTION_COLOR
        elif terrain_type == GOAL:
            return GOAL_COLOR
        elif terrain_type == PATH:
            return PATH_COLOR
        elif terrain_type == CASTLE:
            return CASTLE_COLOR
        return 0
    
    def draw(self):
        # Change background color to deep blue (1) instead of black (0)
        pyxel.cls(1)
        
        if self.game_state == self.EXPLORING:
            self.draw_world()
        elif self.game_state == self.BATTLE:
            self.draw_battle()
        elif self.game_state == self.GAME_OVER:
            self.draw_game_over()
        elif self.game_state == self.WIN:
            self.draw_win()
    
    def draw_world(self):
        # Draw hot reload indicator with animation and a border
        if HOT_RELOAD_ENABLED:
            reload_color = 8 + (pyxel.frame_count // 10) % 8  # Faster cycling colors
            pyxel.rectb(WIDTH - 75, 2, 72, 10, reload_color)  # Adding a border
            pyxel.text(WIDTH - 70, 5, "HOT RELOAD ACTIVE", 7)
        
        # Calculate visible range
        start_x = max(0, self.camera_x)
        end_x = min(self.world_width, start_x + WIDTH // TILE_SIZE + 1)
        start_y = max(0, self.camera_y)
        end_y = min(self.world_height, start_y + HEIGHT // TILE_SIZE + 1)
        
        # First pass: Draw base terrain
        for y in range(start_y, end_y):
            for x in range(start_x, end_x):
                screen_x = (x - self.camera_x) * TILE_SIZE
                screen_y = (y - self.camera_y) * TILE_SIZE
                
                # Skip if outside screen or world boundaries
                if not (0 <= screen_x < WIDTH and 0 <= screen_y < HEIGHT):
                    continue
                
                # Extra boundary check to avoid index errors
                if not (0 <= y < self.world_height and 0 <= x < self.world_width):
                    continue
                    
                # Draw base terrain in Stardew Valley style
                terrain_type = self.world[y][x]
                color = self.get_terrain_color(terrain_type)
                pyxel.rect(screen_x, screen_y, TILE_SIZE, TILE_SIZE, color)
                
                # Add terrain texture variations based on position
                # This creates more natural looking patterns in Stardew style
                if terrain_type == GRASS:
                    # Draw farmland/crop pattern for grass
                    if (x + y) % 4 == 0:  # Regular pattern for tilled soil look
                        # Soil pattern - horizontal rows
                        pyxel.line(screen_x + 2, screen_y + 7, 
                                  screen_x + TILE_SIZE - 2, screen_y + 7, 9)  # Soil line
                        
                        # Random crop/plant sprouts
                        if random.random() < 0.4:  # 40% chance for crops
                            crop_type = (x * 7 + y * 11) % 3  # Deterministic but varied crop types
                            if crop_type == 0:  # Small sprout
                                pyxel.rect(screen_x + 5, screen_y + 3, 1, 4, 3)  # Plant stem
                                pyxel.pset(screen_x + 4, screen_y + 3, 3)  # Leaf
                                pyxel.pset(screen_x + 6, screen_y + 2, 3)  # Leaf
                            elif crop_type == 1:  # Flowered
                                pyxel.rect(screen_x + 8, screen_y + 4, 1, 3, 3)  # Plant stem
                                pyxel.circ(screen_x + 8, screen_y + 3, 1, 8)  # Flower
                            else:  # Taller plant
                                pyxel.line(screen_x + 6, screen_y + 6, 
                                          screen_x + 6, screen_y + 2, 3)  # Tall stem
                                pyxel.pset(screen_x + 5, screen_y + 3, 3)  # Leaf
                                pyxel.pset(screen_x + 7, screen_y + 4, 3)  # Leaf
                
                elif terrain_type == SAND:
                    # Create tilled soil pattern
                    if (x + y) % 3 == 0:
                        # Small soil mounds in Stardew style
                        for i in range(2):
                            dot_x = screen_x + 3 + i*6
                            dot_y = screen_y + 6
                            pyxel.pset(dot_x, dot_y, 4)  # Darker soil spot
                            pyxel.pset(dot_x+1, dot_y, 4)  # Wider spot
                    
                    # Occasional pebbles
                    if (x * 3 + y * 5) % 11 == 0:
                        pyxel.pset(screen_x + (x*7) % TILE_SIZE, screen_y + (y*5) % TILE_SIZE, 5)
                
                elif terrain_type == MOUNTAIN:
                    # Draw mining nodes/rocks in Stardew style
                    # Base rock
                    rock_size = 1 + (x * y) % 3  # Varies rock size
                    rock_x = screen_x + TILE_SIZE//2
                    rock_y = screen_y + TILE_SIZE//2
                    
                    # Main rock shape
                    pyxel.circ(rock_x, rock_y, rock_size + 2, 13)  # Gray rock
                    
                    # Add ore specks/sparkles
                    ore_type = (x + y) % 3
                    ore_color = 10 if ore_type == 0 else (14 if ore_type == 1 else 6)  # Gold, pink or light gray
                    
                    # Ore specks in the rock
                    pyxel.pset(rock_x + 1, rock_y - 1, ore_color)
                    if rock_size > 1:
                        pyxel.pset(rock_x - 1, rock_y, ore_color)
                
                elif terrain_type == WATER:
                    # Create pond water effect with ripples
                    wave_time = (pyxel.frame_count + x*3 + y*2) % 60
                    
                    # Water sparkles - more vibrant blue water like in Stardew
                    sparkle_color = 7  # White sparkle
                    
                    if wave_time < 15:
                        # First wave pattern
                        pyxel.line(screen_x + 2, screen_y + 4, 
                                  screen_x + 6, screen_y + 4, 7)  # White highlight
                    elif wave_time < 30:
                        # Second wave pattern
                        pyxel.line(screen_x + 7, screen_y + 7, 
                                  screen_x + 11, screen_y + 7, 7)  # White highlight
                    elif wave_time < 45:
                        # Occasional sparkle
                        pyxel.pset(screen_x + (x+wave_time) % TILE_SIZE, 
                                 screen_y + (y+wave_time) % TILE_SIZE, sparkle_color)
        
        # Second pass: Draw foreground elements (trees, buildings, etc)
        for y in range(start_y, end_y):
            for x in range(start_x, end_x):
                screen_x = (x - self.camera_x) * TILE_SIZE
                screen_y = (y - self.camera_y) * TILE_SIZE
                
                # Skip if outside screen or world boundaries
                if not (0 <= screen_x < WIDTH and 0 <= screen_y < HEIGHT):
                    continue
                
                # Extra boundary check to avoid index errors
                if not (0 <= y < self.world_height and 0 <= x < self.world_width):
                    continue
                    
                terrain_type = self.world[y][x]
                
                # Draw detailed terrain features
                if terrain_type == FOREST:
                    # Draw Stardew Valley style trees and plants
                    # Use position hash for subtle tree variety
                    tree_type = (x * 5 + y * 7) % 4
                    
                    if tree_type == 0:  # Pine/evergreen tree
                        # Tree trunk
                        trunk_color = 4  # Brown
                        pyxel.rect(screen_x + 6, screen_y + 7, 2, 5, trunk_color)
                        
                        # Tree top - multiple layers of leaves like Stardew pines
                        leaf_color = 3  # Dark green
                        # Bottom layer (widest)
                        pyxel.rect(screen_x + 3, screen_y + 6, 8, 2, leaf_color)
                        # Middle layer
                        pyxel.rect(screen_x + 4, screen_y + 4, 6, 2, leaf_color)
                        # Top layer
                        pyxel.rect(screen_x + 5, screen_y + 2, 4, 2, leaf_color)
                        # Tree top
                        pyxel.rect(screen_x + 6, screen_y, 2, 2, leaf_color)
                        
                        # Add occasional berries/fruit
                        if (x * y) % 7 == 0:
                            pyxel.pset(screen_x + 4, screen_y + 5, 8)  # Red berry
                    
                    elif tree_type == 1:  # Oak/maple tree (round top)
                        # Tree trunk
                        trunk_color = 4  # Brown
                        pyxel.rect(screen_x + 6, screen_y + 8, 2, 4, trunk_color)
                        
                        # Tree top - round Stardew style
                        leaf_color = 3  # Dark green
                        # Create a fuller round top
                        pyxel.circ(screen_x + 7, screen_y + 4, 4, leaf_color)
                        
                        # Add lighter accents on leaves
                        accent_color = 11  # Light green
                        for i in range(2):
                            leaf_x = screen_x + 5 + i*4
                            leaf_y = screen_y + 3 + i*2
                            pyxel.pset(leaf_x, leaf_y, accent_color)
                    
                    elif tree_type == 2:  # Berry bush/crops
                        # Bush base
                        bush_color = 3  # Dark green
                        pyxel.rect(screen_x + 3, screen_y + 7, 8, 5, bush_color)
                        
                        # Bush top - slightly rounded
                        pyxel.rect(screen_x + 4, screen_y + 5, 6, 2, bush_color)
                        pyxel.rect(screen_x + 5, screen_y + 3, 4, 2, bush_color)
                        
                        # Add berries/crops
                        berry_count = (x + y) % 4 + 1  # 1-4 berries
                        berry_color = 8  # Red berries
                        for i in range(berry_count):
                            berry_x = screen_x + 4 + (i * 7) % 6
                            berry_y = screen_y + 5 + (i * 3) % 5
                            pyxel.pset(berry_x, berry_y, berry_color)
                    
                    else:  # Fruit tree
                        # Tree trunk
                        trunk_color = 4  # Brown
                        pyxel.rect(screen_x + 6, screen_y + 8, 2, 4, trunk_color)
                        
                        # Fuller round top - like fruit trees in Stardew
                        leaf_color = 11  # Lighter green for fruit trees
                        pyxel.circ(screen_x + 7, screen_y + 4, 5, leaf_color)
                        
                        # Add fruits - orange/peach color
                        fruit_color = 9  # Orange/peach
                        fruit_count = (x * y) % 3 + 1  # 1-3 fruits
                        for i in range(fruit_count):
                            fruit_x = screen_x + 5 + (i * 5) % 5
                            fruit_y = screen_y + 3 + (i * 7) % 5
                            pyxel.pset(fruit_x, fruit_y, fruit_color)
                            pyxel.pset(fruit_x + 1, fruit_y, fruit_color)  # Slightly larger fruits
                
                elif terrain_type == CASTLE:
                    # Draw Stardew Valley style buildings and structures
                    # Check neighbors to determine if this is part of a larger structure
                    is_edge = False
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            nx, ny = x + dx, y + dy
                            if not (0 <= nx < self.world_width and 0 <= ny < self.world_height):
                                continue
                            if self.world[ny][nx] != CASTLE:
                                is_edge = True
                                break
                    
                    # Determine building type
                    building_type = (x * 3 + y * 5) % 4
                    
                    if is_edge:  # Edge pieces are facades/fronts of buildings
                        if building_type == 0:  # Stardew house/cabin
                            # Main house walls
                            wall_color = 6  # Light gray walls
                            pyxel.rect(screen_x + 1, screen_y + 5, TILE_SIZE - 2, TILE_SIZE - 6, wall_color)
                            
                            # Roof (triangle/pitched roof)
                            roof_color = 8  # Red roof
                            pyxel.tri(screen_x, screen_y + 5,
                                    screen_x + TILE_SIZE//2, screen_y,
                                    screen_x + TILE_SIZE, screen_y + 5, roof_color)
                            
                            # Door
                            door_color = 4  # Brown door
                            pyxel.rect(screen_x + 5, screen_y + 8, 4, 4, door_color)
                            
                            # Window
                            window_color = 12  # Blue window
                            pyxel.rect(screen_x + 2, screen_y + 6, 3, 2, window_color)
                        
                        elif building_type == 1:  # Shop/general store
                            # Main building
                            wall_color = 9  # Tan/brown walls
                            pyxel.rect(screen_x + 1, screen_y + 3, TILE_SIZE - 2, TILE_SIZE - 4, wall_color)
                            
                            # Flat roof with overhang
                            roof_color = 4  # Brown roof
                            pyxel.rect(screen_x, screen_y + 2, TILE_SIZE, 2, roof_color)
                            
                            # Storefront window
                            window_color = 7  # White window frame
                            window_glass = 12  # Blue glass
                            pyxel.rect(screen_x + 2, screen_y + 5, 8, 4, window_color)
                            pyxel.rect(screen_x + 3, screen_y + 6, 6, 2, window_glass)
                            
                            # Store sign
                            if (x + y) % 3 == 0:
                                sign_color = 10  # Yellow sign
                                pyxel.rect(screen_x + 4, screen_y + 1, 4, 1, sign_color)
                        
                        elif building_type == 2:  # Farm building/barn
                            # Main barn structure
                            wall_color = 4  # Brown walls
                            pyxel.rect(screen_x + 1, screen_y + 4, TILE_SIZE - 2, TILE_SIZE - 5, wall_color)
                            
                            # Barn roof
                            roof_color = 8  # Red roof
                            pyxel.rect(screen_x, screen_y + 2, TILE_SIZE, 2, roof_color)
                            
                            # Barn door (large)
                            door_color = 4  # Darker brown
                            door_accent = 9  # Light brown accents
                            pyxel.rect(screen_x + 4, screen_y + 6, 6, 5, door_color)
                            pyxel.line(screen_x + 7, screen_y + 6, screen_x + 7, screen_y + 11, door_accent)
                        
                        else:  # Community center/special building
                            # Fancy building
                            wall_color = 6  # Light stone
                            pyxel.rect(screen_x + 1, screen_y + 4, TILE_SIZE - 2, TILE_SIZE - 5, wall_color)
                            
                            # Ornate roof
                            roof_color = 9  # Copper/gold colored roof
                            pyxel.rect(screen_x, screen_y + 1, TILE_SIZE, 3, roof_color)
                            
                            # Decorative elements
                            if pyxel.frame_count % 60 < 30:
                                # Flags or decorations
                                pyxel.tri(screen_x + 2, screen_y + 1,
                                        screen_x + 2, screen_y + 3,
                                        screen_x + 4, screen_y + 2, 10)  # Yellow flag
                                
                                pyxel.tri(screen_x + TILE_SIZE - 2, screen_y + 1,
                                        screen_x + TILE_SIZE - 2, screen_y + 3,
                                        screen_x + TILE_SIZE - 4, screen_y + 2, 8)  # Red flag
                    
                    else:  # Interior pieces are simpler
                        # Simple interior floor/courtyard
                        floor_color = 4 if building_type < 2 else 9  # Brown or tan floor
                        pyxel.rect(screen_x + 1, screen_y + 1, TILE_SIZE - 2, TILE_SIZE - 2, floor_color)
                        
                        # Add floor patterns
                        if (x + y) % 3 == 0:
                            pattern_color = 5  # Darker accent
                            pyxel.pset(screen_x + 4, screen_y + 4, pattern_color)
                            pyxel.pset(screen_x + 10, screen_y + 10, pattern_color)
                            
                        # Add furniture or decorations for visual interest
                        if (x * y) % 5 == 0:
                            item_type = (x + y) % 3
                            if item_type == 0:  # Table
                                pyxel.rect(screen_x + 5, screen_y + 5, 4, 3, 4)
                            elif item_type == 1:  # Carpet
                                pyxel.rect(screen_x + 3, screen_y + 3, 8, 8, 8)
                            else:  # Plant/decoration
                                pyxel.rect(screen_x + 7, screen_y + 7, 2, 3, 3)
                
                elif terrain_type == PATH:
                    # Create Stardew Valley style paths/dirt
                    path_variant = (x * 3 + y * 5) % 4
                    
                    if path_variant == 0:  # Stone path tiles
                        # Main stone tile
                        tile_color = 5  # Dark gray stone
                        accent_color = 13  # Light gray highlights
                        
                        # Draw four cobblestones
                        stone_positions = [
                            (2, 2, 5, 5),  # top-left: x, y, width, height
                            (8, 2, 4, 5),  # top-right
                            (2, 8, 5, 4),  # bottom-left
                            (8, 8, 4, 4)   # bottom-right
                        ]
                        
                        for sx, sy, sw, sh in stone_positions:
                            pyxel.rect(screen_x + sx, screen_y + sy, sw, sh, tile_color)
                            
                        # Add stone borders/cracks
                        pyxel.line(screen_x + 7, screen_y + 2, screen_x + 7, screen_y + TILE_SIZE - 2, accent_color)
                        pyxel.line(screen_x + 2, screen_y + 7, screen_x + TILE_SIZE - 2, screen_y + 7, accent_color)
                        
                    elif path_variant == 1:  # Dirt path with footprints
                        # Base dirt
                        dirt_color = 4  # Brown
                        highlight_color = 9  # Lighter brown highlights
                        
                        # Stardew-style dirt path with subtle texture
                        pyxel.rect(screen_x + 1, screen_y + 1, TILE_SIZE - 2, TILE_SIZE - 2, dirt_color)
                        
                        # Add footprint or path texture
                        if (x + y) % 2 == 0:  # Alternate texture pattern
                            # Left footprint
                            pyxel.oval(screen_x + 3, screen_y + 4, 2, 4, highlight_color)
                            # Right footprint
                            pyxel.oval(screen_x + 9, screen_y + 8, 2, 4, highlight_color)
                        else:
                            # Scattered dirt texture
                            dirt_spots = [(4, 3), (7, 9), (10, 5), (3, 10)]
                            for dx, dy in dirt_spots:
                                pyxel.pset(screen_x + dx, screen_y + dy, highlight_color)
                        
                    elif path_variant == 2:  # Wood plank path
                        # Wooden plank path like in Stardew farms
                        wood_color = 4  # Brown
                        highlight_color = 9  # Light brown grain
                        
                        # Base planks - horizontal
                        for i in range(3):
                            pyxel.rect(screen_x + 2, screen_y + 1 + i*4, TILE_SIZE - 4, 3, wood_color)
                            
                        # Add wood grain
                        for i in range(3):
                            grain_y = screen_y + 2 + i*4
                            # Shorter grain marks
                            pyxel.line(screen_x + 3, grain_y, screen_x + 6, grain_y, highlight_color)
                            pyxel.line(screen_x + 8, grain_y, screen_x + 11, grain_y, highlight_color)
                            
                    else:  # Decorated path or special tile
                        # Fancy stone/decorated path
                        base_color = 5  # Gray stone
                        accent_color = 10  # Gold/yellow accents
                        
                        # Base stone
                        pyxel.rect(screen_x + 1, screen_y + 1, TILE_SIZE - 2, TILE_SIZE - 2, base_color)
                        
                        # Decorative pattern (rune or symbol)
                        # Simple rune pattern
                        pyxel.rect(screen_x + 5, screen_y + 3, 4, 1, accent_color)  # Top horizontal
                        pyxel.rect(screen_x + 5, screen_y + 10, 4, 1, accent_color)  # Bottom horizontal
                        pyxel.rect(screen_x + 7, screen_y + 3, 1, 8, accent_color)  # Vertical
                        # Optional accent mark
                        if (x + y) % 3 == 0:
                            pyxel.pset(screen_x + 4, screen_y + 5, accent_color)
                
                elif terrain_type == POTION:
                    # Draw Stardew Valley style forageable/item
                    item_type = (x * 3 + y * 5) % 4
                    
                    if item_type == 0:  # Health potion/red mushroom
                        # Stardew red mushroom
                        cap_color = 8  # Red cap
                        stem_color = 7  # White stem
                        
                        # Draw mushroom stem
                        pyxel.rect(screen_x + 6, screen_y + 7, 2, 4, stem_color)
                        
                        # Draw mushroom cap
                        pyxel.circ(screen_x + 7, screen_y + 5, 3, cap_color)
                        
                        # Add white dots on cap like a toadstool
                        pyxel.pset(screen_x + 5, screen_y + 4, stem_color)
                        pyxel.pset(screen_x + 8, screen_y + 5, stem_color)
                        
                    elif item_type == 1:  # Blue berry/blue potion
                        # Stardew blueberry
                        berry_color = 12  # Blue
                        leaf_color = 11  # Light green
                        
                        # Draw berry cluster
                        pyxel.circ(screen_x + 7, screen_y + 7, 3, berry_color)
                        
                        # Add berry highlights
                        pyxel.pset(screen_x + 6, screen_y + 6, 1)  # Darker blue shade
                        pyxel.pset(screen_x + 8, screen_y + 8, 7)  # White highlight
                        
                        # Add leaves
                        pyxel.rect(screen_x + 7, screen_y + 3, 1, 2, leaf_color)  # Stem
                        pyxel.line(screen_x + 5, screen_y + 4, screen_x + 7, screen_y + 3, leaf_color)  # Leaf
                        
                    elif item_type == 2:  # Star drop (rare special item in Stardew)
                        # Stardew star item
                        star_color = 10  # Yellow/gold
                        glow_color = 9  # Orange glow
                        
                        # Draw star center
                        pyxel.circ(screen_x + 7, screen_y + 7, 2, star_color)
                        
                        # Draw star points with animation
                        star_angle = pyxel.frame_count * 0.1
                        for i in range(5):
                            angle = star_angle + i * (2 * math.pi / 5)
                            length = 3 + math.sin(pyxel.frame_count * 0.2) * 0.5
                            end_x = screen_x + 7 + int(math.cos(angle) * length)
                            end_y = screen_y + 7 + int(math.sin(angle) * length)
                            pyxel.line(screen_x + 7, screen_y + 7, end_x, end_y, star_color)
                        
                        # Subtle glow effect
                        if pyxel.frame_count % 30 < 15:
                            pyxel.circb(screen_x + 7, screen_y + 7, 4, glow_color)
                    
                    else:  # Golden/special potion
                        # Stardew rare item bottle
                        bottle_color = 7  # Clear glass
                        liquid_color = 10  # Gold liquid
                        cork_color = 4  # Brown cork
                        
                        # Draw bottle
                        pyxel.rect(screen_x + 4, screen_y + 6, 6, 5, bottle_color)  # Bottle body
                        pyxel.rect(screen_x + 6, screen_y + 3, 2, 3, bottle_color)  # Bottle neck
                        
                        # Draw liquid
                        pyxel.rect(screen_x + 5, screen_y + 7, 4, 3, liquid_color)  # Liquid
                        
                        # Draw cork
                        pyxel.rect(screen_x + 6, screen_y + 2, 2, 1, cork_color)
                        
                        # Magical sparkles/bubbles
                        for i in range(2):
                            bubble_time = (pyxel.frame_count + i*15) % 30
                            if bubble_time < 15:
                                bubble_y = screen_y + 9 - (bubble_time / 15) * 2
                                pyxel.pset(screen_x + 5 + i*3, bubble_y, 7)  # White bubble
                        
                elif terrain_type == WALL:
                    # Draw Stardew Valley style fences or walls
                    # Determine wall/fence type based on position
                    wall_type = (x * 3 + y * 5) % 4
                    
                    if wall_type == 0:  # Wooden fence
                        # Wood fence colors
                        wood_color = 4  # Brown
                        accent_color = 9  # Light brown for details
                        
                        # Horizontal fence boards
                        pyxel.rect(screen_x + 1, screen_y + 3, TILE_SIZE - 2, 2, wood_color)  # Top rail
                        pyxel.rect(screen_x + 1, screen_y + 9, TILE_SIZE - 2, 2, wood_color)  # Bottom rail
                        
                        # Vertical posts
                        post_positions = [3, 10]  # x positions for posts
                        for post_x in post_positions:
                            pyxel.rect(screen_x + post_x, screen_y + 1, 2, TILE_SIZE - 2, wood_color)
                            
                        # Add wood grain detail
                        for post_x in post_positions:
                            pyxel.line(screen_x + post_x + 1, screen_y + 3, 
                                      screen_x + post_x + 1, screen_y + 10, accent_color)
                    
                    elif wall_type == 1:  # Stone fence/wall
                        # Stone wall colors
                        stone_color = 5  # Dark gray
                        highlight_color = 13  # Light gray for details
                        
                        # Stardew-style stone pile fence
                        # Base stones
                        stone_positions = [
                            (2, 5, 3, 7),   # left stone: x, y, width, height
                            (6, 4, 4, 8),    # middle stone
                            (10, 6, 2, 6)    # right stone
                        ]
                        
                        for sx, sy, sw, sh in stone_positions:
                            pyxel.rect(screen_x + sx, screen_y + sy, sw, sh, stone_color)
                            
                        # Add highlights to stones
                        for sx, sy, sw, sh in stone_positions:
                            # Just a dot highlight on each stone
                            pyxel.pset(screen_x + sx + 1, screen_y + sy + 1, highlight_color)
                    
                    elif wall_type == 2:  # Hardwood/decorative fence
                        # Hardwood fence colors
                        dark_wood = 4  # Dark brown
                        light_wood = 9  # Lighter brown
                        
                        # Vertical posts
                        pyxel.rect(screen_x + 2, screen_y + 1, 2, TILE_SIZE - 2, dark_wood)  # Left post
                        pyxel.rect(screen_x + TILE_SIZE - 4, screen_y + 1, 2, TILE_SIZE - 2, dark_wood)  # Right post
                        
                        # Decorative lattice pattern
                        # Cross pattern
                        pyxel.line(screen_x + 4, screen_y + 3, 
                                  screen_x + TILE_SIZE - 5, screen_y + TILE_SIZE - 3, light_wood)
                        pyxel.line(screen_x + TILE_SIZE - 5, screen_y + 3, 
                                  screen_x + 4, screen_y + TILE_SIZE - 3, light_wood)
                        
                        # Horizontal bars
                        pyxel.rect(screen_x + 2, screen_y + 4, TILE_SIZE - 4, 1, light_wood)
                        pyxel.rect(screen_x + 2, screen_y + TILE_SIZE - 5, TILE_SIZE - 4, 1, light_wood)
                    
                    else:  # Iron/metal fence
                        # Metal fence colors
                        metal_color = 5  # Dark gray
                        highlight_color = 6  # Light gray for details
                        
                        # Vertical bars
                        bar_positions = [3, 6, 9]
                        for bar_x in bar_positions:
                            pyxel.rect(screen_x + bar_x, screen_y + 2, 1, TILE_SIZE - 4, metal_color)
                            
                        # Horizontal bars
                        pyxel.rect(screen_x + 2, screen_y + 3, TILE_SIZE - 4, 1, metal_color)
                        pyxel.rect(screen_x + 2, screen_y + 10, TILE_SIZE - 4, 1, metal_color)
                        
                        # Decorative top caps
                        for bar_x in bar_positions:
                            # Spear point tops
                            pyxel.tri(screen_x + bar_x - 1, screen_y + 2,
                                     screen_x + bar_x + 2, screen_y + 2,
                                     screen_x + bar_x + 0.5, screen_y, highlight_color)
        
        # Draw the human player character (knight)
        self.draw_player()
        
        # Draw the AI companion character (wizard)
        self.draw_companion()
        
        # Draw goal marker with improved visual appeal
        goal_screen_x = (self.goal_x - self.camera_x) * TILE_SIZE
        goal_screen_y = (self.goal_y - self.camera_y) * TILE_SIZE
        if 0 <= goal_screen_x < WIDTH and 0 <= goal_screen_y < HEIGHT:
            # Draw the goal chest with same code as before
            self.draw_goal(goal_screen_x, goal_screen_y)
            
    def draw_player(self):
        """Draw the human player character in Stardew Valley style"""
        player_screen_x = (self.player_x - self.camera_x) * TILE_SIZE
        player_screen_y = (self.player_y - self.camera_y) * TILE_SIZE
        
        # Only draw if on screen
        if not (0 <= player_screen_x < WIDTH and 0 <= player_screen_y < HEIGHT):
            return
            
        # Draw outline to show it's controlled by both AI and human
        outline_color = 10  # Yellow outline
        pyxel.rectb(player_screen_x, player_screen_y, TILE_SIZE, TILE_SIZE, outline_color)
        
        # Draw farmer/adventurer body
        body_color = 9  # Brown for clothing
        face_color = 7  # Light skin tone
        hair_color = 4  # Dark brown hair
        
        # Base body - shirt/overalls
        pyxel.rect(player_screen_x + 2, player_screen_y + 4, TILE_SIZE - 4, TILE_SIZE - 5, body_color)
        
        # Animate player based on frame
        if self.player_frame == 0:
            # Head
            pyxel.rect(player_screen_x + 3, player_screen_y + 1, 6, 4, face_color)
            
            # Hair - slightly covering forehead
            pyxel.rect(player_screen_x + 3, player_screen_y, 6, 2, hair_color)
            pyxel.rect(player_screen_x + 2, player_screen_y + 1, 2, 2, hair_color)
            
            # Face details
            pyxel.pset(player_screen_x + 4, player_screen_y + 2, 0)  # Left eye
            pyxel.pset(player_screen_x + 7, player_screen_y + 2, 0)  # Right eye
            pyxel.pset(player_screen_x + 5, player_screen_y + 3, 0)  # Mouth dot
            
            # Arms - holding tool
            pyxel.rect(player_screen_x + 1, player_screen_y + 5, 2, 4, body_color)  # Left arm
            pyxel.rect(player_screen_x + TILE_SIZE - 3, player_screen_y + 5, 2, 3, body_color)  # Right arm
            
            # Tool - sword/farming tool
            pyxel.rect(player_screen_x + TILE_SIZE - 2, player_screen_y + 4, 1, 6, 5)  # Tool handle
            pyxel.rect(player_screen_x + TILE_SIZE - 4, player_screen_y + 3, 3, 2, 7)  # Tool head
        else:
            # Head - slightly different position for animation
            pyxel.rect(player_screen_x + 3, player_screen_y + 2, 6, 4, face_color)
            
            # Hair - slightly covering forehead
            pyxel.rect(player_screen_x + 3, player_screen_y + 1, 6, 2, hair_color)
            pyxel.rect(player_screen_x + 2, player_screen_y + 2, 2, 2, hair_color)
            
            # Face details - different expression
            pyxel.pset(player_screen_x + 4, player_screen_y + 3, 0)  # Left eye
            pyxel.pset(player_screen_x + 7, player_screen_y + 3, 0)  # Right eye
            pyxel.line(player_screen_x + 5, player_screen_y + 4, player_screen_x + 6, player_screen_y + 4, 0)  # Smile
            
            # Arms - different position
            pyxel.rect(player_screen_x + 1, player_screen_y + 6, 2, 3, body_color)  # Left arm
            pyxel.rect(player_screen_x + TILE_SIZE - 3, player_screen_y + 6, 2, 4, body_color)  # Right arm
            
            # Tool - in swinging position
            pyxel.line(player_screen_x + TILE_SIZE - 3, player_screen_y + 5, 
                      player_screen_x + TILE_SIZE - 1, player_screen_y + 2, 5)  # Tool handle
            pyxel.rect(player_screen_x + TILE_SIZE - 2, player_screen_y + 1, 3, 2, 7)  # Tool head
        
        # Legs
        pyxel.rect(player_screen_x + 3, player_screen_y + 9, 3, 4, 4)  # Left leg - darker pants
        pyxel.rect(player_screen_x + 7, player_screen_y + 9, 3, 4, 4)  # Right leg
        
        # Show both AI and human are controlling
        pyxel.text(player_screen_x - 3, player_screen_y - 6, "HUMAN+AI", 10)
    
    def draw_companion(self):
        """
        In single character mode, we don't draw the companion.
        This function is kept for compatibility but does nothing.
        """
        # With a single character, we don't draw the wizard
        pass
    
    def draw_goal(self, goal_screen_x, goal_screen_y):
        """Draw the goal in Stardew Valley style"""
        # Create a magical Junimo hut or Stardrop fruit as goal
        
        # Add subtle glow effect around the goal
        if pyxel.frame_count % 30 < 15:
            # Pulse a faint glow
            pyxel.circb(goal_screen_x + TILE_SIZE//2, goal_screen_y + TILE_SIZE//2, 7, 12)  # Blue glow
        
        # Draw a Junimo hut (small cabin/shrine from Stardew Valley)
        wood_color = 4  # Brown for wood
        roof_color = 8  # Red for roof
        accent_color = 10  # Yellow for highlights
        
        # Base structure - small wooden cabin
        pyxel.rect(goal_screen_x + 2, goal_screen_y + 6, 10, 6, wood_color)  # Main cabin body
        
        # Roof (triangular)
        pyxel.tri(goal_screen_x + 1, goal_screen_y + 6,
                 goal_screen_x + 7, goal_screen_y + 1,
                 goal_screen_x + 13, goal_screen_y + 6, roof_color)
        
        # Door
        door_color = 9  # Lighter brown
        pyxel.rect(goal_screen_x + 6, goal_screen_y + 8, 2, 4, door_color)
        
        # Window
        window_frame_color = wood_color
        window_color = 12  # Blue window
        pyxel.rect(goal_screen_x + 3, goal_screen_y + 7, 2, 2, window_frame_color)
        pyxel.pset(goal_screen_x + 3, goal_screen_y + 7, window_color)
        pyxel.pset(goal_screen_x + 4, goal_screen_y + 7, window_color)
        pyxel.pset(goal_screen_x + 3, goal_screen_y + 8, window_color)
        pyxel.pset(goal_screen_x + 4, goal_screen_y + 8, window_color)
        
        # Magical elements - animate based on frame count
        if pyxel.frame_count % 60 < 30:
            # Closed door with magical stars
            pyxel.pset(goal_screen_x + 4, goal_screen_y + 3, accent_color)  # Star near roof
            pyxel.pset(goal_screen_x + 10, goal_screen_y + 4, accent_color)  # Star near roof
            
            # Small sparkles around hut
            star_angle = pyxel.frame_count * 0.1
            pyxel.pset(
                goal_screen_x + 7 + int(math.cos(star_angle) * 6),
                goal_screen_y + 7 + int(math.sin(star_angle) * 6),
                7  # White sparkle
            )
        else:
            # Door slightly open with magical glow
            pyxel.line(goal_screen_x + 6, goal_screen_y + 10, goal_screen_x + 7, goal_screen_y + 10, 7)  # Door crack
            
            # More visible magical elements
            for i in range(3):
                star_angle = pyxel.frame_count * 0.1 + i * (math.pi * 2 / 3)
                star_distance = 4 + math.sin(pyxel.frame_count * 0.05) * 2
                pyxel.pset(
                    goal_screen_x + 7 + int(math.cos(star_angle) * star_distance),
                    goal_screen_y + 7 + int(math.sin(star_angle) * star_distance),
                    accent_color  # Yellow/gold magical particles
                )
            
            # Small Junimo peeking out (the colorful spirits from the game)
            junimo_color = 11  # Green Junimo
            pyxel.circ(goal_screen_x + 7, goal_screen_y + 9, 1, junimo_color)
        
        # Draw player stats in medieval style frame - LARGER for the higher resolution
        pyxel.rectb(2, 2, 220, 120, 7)  # Border - much larger for vertical height
        pyxel.rect(3, 3, 218, 118, 0)   # Background - much larger for vertical height
        
        # Draw stats with small decorative elements
        # Player stats label - BIGGER TEXT for visibility
        pyxel.text(10, 10, "CHAMPION:", 10)  # Yellow label
        # Draw multiple times for "bold" effect to make more visible
        pyxel.text(10, 20, f"HP: {self.player_hp}/{self.player_max_hp}", 10)  # Base color
        pyxel.text(11, 20, f"HP: {self.player_hp}/{self.player_max_hp}", 10)  # Bold effect
        
        # Removed hot reload test banner
        
        # Inventory - larger
        pyxel.text(10, 60, f"POTIONS: {self.potions}", 9)  # Orange potions
        
        # Draw decorative corners
        pyxel.pset(2, 2, 9)  # Top-left
        pyxel.pset(91, 2, 9)  # Top-right
        pyxel.pset(2, 27, 9)  # Bottom-left
        pyxel.pset(91, 27, 9)  # Bottom-right
        
        # Draw mini-map in top right corner
        mini_map_size = 60
        mini_map_x = WIDTH - mini_map_size - 4
        mini_map_y = 4
        mini_map_scale = max(self.world_width, self.world_height) / mini_map_size
        
        # Draw mini-map background and border
        pyxel.rectb(mini_map_x - 1, mini_map_y - 1, mini_map_size + 2, mini_map_size + 2, 7)  # Border
        pyxel.rect(mini_map_x, mini_map_y, mini_map_size, mini_map_size, 0)  # Background
        
        # Draw decorative corners for mini-map
        pyxel.pset(mini_map_x - 1, mini_map_y - 1, 9)  # Top-left
        pyxel.pset(mini_map_x + mini_map_size, mini_map_y - 1, 9)  # Top-right
        pyxel.pset(mini_map_x - 1, mini_map_y + mini_map_size, 9)  # Bottom-left
        pyxel.pset(mini_map_x + mini_map_size, mini_map_y + mini_map_size, 9)  # Bottom-right
        
        # Draw simplified world on mini-map
        for y in range(0, self.world_height, 2):
            for x in range(0, self.world_width, 2):
                mini_x = mini_map_x + int(x / mini_map_scale)
                mini_y = mini_map_y + int(y / mini_map_scale)
                
                if 0 <= mini_x < mini_map_x + mini_map_size and 0 <= mini_y < mini_map_y + mini_map_size:
                    # Use simplified colors for mini-map
                    color = self.get_terrain_color(self.world[y][x])
                    pyxel.pset(mini_x, mini_y, color)
        
        # Draw player on mini-map
        player_mini_x = mini_map_x + int(self.player_x / mini_map_scale)
        player_mini_y = mini_map_y + int(self.player_y / mini_map_scale)
        pyxel.pset(player_mini_x, player_mini_y, 8)  # Red for player
        
        # Draw goal on mini-map
        goal_mini_x = mini_map_x + int(self.goal_x / mini_map_scale)
        goal_mini_y = mini_map_y + int(self.goal_y / mini_map_scale)
        pyxel.pset(goal_mini_x, goal_mini_y, 10)  # Yellow for goal
    
    def draw_battle(self):
        # Handle battle transition animation if needed
        if hasattr(self, 'battle_transition_frame') and self.battle_transition_frame < self.battle_transition_max_frames:
            # During transition, create visual effects
            transition_progress = self.battle_transition_frame / self.battle_transition_max_frames
            
            # Flash effect
            if self.battle_transition_frame % 6 < 3:  # Flash every few frames
                # Flash the screen with white
                pyxel.cls(7)
                
                # Draw a swirling vortex effect
                center_x, center_y = WIDTH // 2, HEIGHT // 2
                max_radius = max(WIDTH, HEIGHT) * (1 - transition_progress)
                
                for r in range(0, int(max_radius), 10):
                    angle_offset = pyxel.frame_count * 0.1 + r * 0.02
                    segments = 16
                    for i in range(segments):
                        angle1 = i * (2 * math.pi / segments) + angle_offset
                        angle2 = (i + 1) * (2 * math.pi / segments) + angle_offset
                        x1 = center_x + math.cos(angle1) * r
                        y1 = center_y + math.sin(angle1) * r
                        x2 = center_x + math.cos(angle2) * r
                        y2 = center_y + math.sin(angle2) * r
                        color = (i % 8) + 8
                        pyxel.line(x1, y1, x2, y2, color)
                
                # Add camera shake
                shake_x = random.randint(-self.battle_shake_intensity, self.battle_shake_intensity)
                shake_y = random.randint(-self.battle_shake_intensity, self.battle_shake_intensity)
                pyxel.camera(shake_x, shake_y)
            else:
                # Reset camera position between shakes
                pyxel.camera(0, 0)
            
            # Increment transition frame
            self.battle_transition_frame += 1
            
            # When transition completes, reset camera and take a screenshot
            if self.battle_transition_frame >= self.battle_transition_max_frames:
                pyxel.camera(0, 0)
                self.screenshot_mgr.take_screenshot("battle_transition_complete")
            
            # Return early during transition frames
            return
        
        # Reset camera position to ensure it's centered for battle
        pyxel.camera(0, 0)
        
        # Removed hot reload test banner
            
        # Draw battle background with an enhanced gradient and animated sky
        for y in range(HEIGHT):
            intensity = max(0, min(1, 1 - y / HEIGHT * 1.5))
            if y < HEIGHT * 0.6:  # Sky part
                # Create a more dynamic sky with time-based animation
                if y < HEIGHT * 0.3:  # Upper sky
                    color = 12  # Lighter blue for upper sky
                else:  # Lower sky with gradient
                    color = 1  # Darker blue
                
                # Add animated clouds using perlin noise
                if y > HEIGHT * 0.1 and y < HEIGHT * 0.5:
                    # Use frame count to animate clouds
                    cloud_value = noise.pnoise2(y * 0.05, (pyxel.frame_count * 0.01) % 100, octaves=1)
                    if cloud_value > 0.2:
                        color = 7  # White clouds
            else:  # Ground part
                color = 3  # Green ground
                
            pyxel.line(0, y, WIDTH - 1, y, color)
        
        # Draw terrain-based background details
        terrain = self.world[self.player_y][self.player_x]
        
        # Draw medieval-style scene based on terrain
        if terrain == FOREST:
            # Draw trees - more trees for larger screen
            for i in range(8):
                x = 30 + i * 40
                height = 25 + (i % 3) * 10
                pyxel.rect(x - 5, HEIGHT * 0.65 - height, 10, height, 4)  # Trunk
                pyxel.circ(x, HEIGHT * 0.65 - height - 15, 15, 3)  # Leaves
                # Add detail to trees
                pyxel.circ(x + 5, HEIGHT * 0.65 - height - 10, 7, 3)  # More leaves
        elif terrain == MOUNTAIN:
            # Draw mountains - adjusted for larger screen
            for i in range(5):
                x = 40 + i * 60
                pyxel.tri(x, HEIGHT * 0.3, x - 50, HEIGHT * 0.65, x + 50, HEIGHT * 0.65, 13)  # Mountain
                pyxel.tri(x, HEIGHT * 0.35, x - 20, HEIGHT * 0.45, x + 20, HEIGHT * 0.45, 7)  # Snow cap
                # Add details to mountains
                pyxel.line(x - 20, HEIGHT * 0.45, x - 30, HEIGHT * 0.5, 13)  # Ridge line
        elif terrain == CASTLE:
            # Draw castle background - centered and larger
            castle_center = WIDTH // 2
            castle_width = 100
            castle_height = 60
            castle_y = HEIGHT * 0.4
            
            # Main castle structure
            pyxel.rect(castle_center - castle_width/2, castle_y, castle_width, castle_height, 5)  # Castle wall
            pyxel.rect(castle_center - castle_width/2 - 15, castle_y - 20, 25, castle_height - 10, 5)  # Left tower
            pyxel.rect(castle_center + castle_width/2 - 10, castle_y - 20, 25, castle_height - 10, 5)  # Right tower
            
            # Windows
            for i in range(3):
                for j in range(2):
                    window_x = castle_center - 30 + i * 30
                    window_y = castle_y + 15 + j * 30
                    pyxel.rect(window_x, window_y, 10, 15, 7)  # Window frame
                    pyxel.rect(window_x + 1, window_y + 1, 8, 13, 1)  # Window glass
            
            # Crenellations on top
            for i in range(10):
                pyxel.rect(castle_center - castle_width/2 + 10 * i, castle_y - 5, 5, 5, 5)
        else:
            # Default grassy plain with more details
            pyxel.rect(0, HEIGHT * 0.65, WIDTH, HEIGHT * 0.35, 3)  # Ground
            
            # Add some small hills
            for i in range(6):
                x = WIDTH * (0.1 + i * 0.15)
                hill_size = 30 + (i % 3) * 15
                pyxel.circ(x, HEIGHT * 0.65, hill_size, 3)
                
                # Add some flowers or rocks
                if i % 2 == 0:
                    for j in range(3):
                        flower_x = x + random.randint(-20, 20)
                        flower_y = HEIGHT * 0.65 - random.randint(5, 10)
                        pyxel.pset(flower_x, flower_y, 10)  # Yellow flowers
                else:
                    rock_x = x - 5
                    rock_y = HEIGHT * 0.65 - 5
                    pyxel.rect(rock_x, rock_y, 10, 5, 13)  # Gray rocks
        
        # Draw enemy - scaled up and more detailed
        enemy_x = WIDTH // 2
        enemy_y = HEIGHT // 3
        scale = 1.5  # Scale factor for larger enemies
        
        # Different enemy appearances
        if self.battle_enemy.name == "Goblin":
            # Draw goblin with more details
            head_size = int(10 * scale)
            body_size = int(12 * scale)
            
            # Body parts
            pyxel.rect(enemy_x - head_size//2, enemy_y - head_size//2, head_size, head_size, 2)  # Green head
            pyxel.rect(enemy_x - body_size//2, enemy_y + head_size//2, body_size, body_size, 2)  # Body
            
            # Arms
            arm_length = int(8 * scale)
            pyxel.rect(enemy_x - body_size//2 - arm_length, enemy_y + head_size//2 + body_size//4, 
                      arm_length, int(4 * scale), 2)  # Left arm
            pyxel.rect(enemy_x + body_size//2, enemy_y + head_size//2 + body_size//4, 
                      arm_length, int(4 * scale), 2)  # Right arm
            
            # Face details
            pyxel.pset(enemy_x - head_size//4, enemy_y - head_size//4, 8)  # Left eye
            pyxel.pset(enemy_x + head_size//4, enemy_y - head_size//4, 8)  # Right eye
            pyxel.line(enemy_x - head_size//4, enemy_y + head_size//4, 
                      enemy_x + head_size//4, enemy_y + head_size//4, 0)  # Mouth
            
            # Weapon - crude club
            pyxel.rect(enemy_x + body_size//2 + arm_length, enemy_y, int(5 * scale), int(12 * scale), 4)
            
        elif self.battle_enemy.name == "Skeleton":
            # Draw skeleton with better proportions and WATCHMAN hot reload animations
            skull_size = int(10 * scale)
            body_size = int(15 * scale)
            
            # Animation effects
            bob_offset = math.sin(pyxel.frame_count * 0.1) * 3
            glow_color = 7
            if (pyxel.frame_count // 10) % 6 == 0:  # Flash effect to test hot reload
                glow_color = 8 + (pyxel.frame_count // 5) % 8  # Cycle through colors
            
            # Skull outline glow effect
            pyxel.rectb(enemy_x - skull_size//2 - 1, enemy_y - skull_size - 1, 
                      skull_size + 2, skull_size + 2, glow_color)
            
            # Skull and ribcage
            pyxel.rect(enemy_x - skull_size//2, enemy_y - skull_size + bob_offset, 
                      skull_size, skull_size, 7)  # Skull
            pyxel.rect(enemy_x - body_size//2, enemy_y + bob_offset, 
                      body_size, body_size, 7)  # Ribcage
            
            # Draw ribs with animation
            for i in range(4):
                rib_y = enemy_y + bob_offset + 2 + i * 4
                # Animate ribs - they expand and contract
                rib_width = body_size * (0.9 + math.sin(pyxel.frame_count * 0.05) * 0.1)
                pyxel.line(enemy_x - rib_width//2, rib_y, enemy_x + rib_width//2, rib_y, 7)
            
            # Eyes with blinking
            if (pyxel.frame_count // 30) % 6 != 0:  # Eyes open most of the time
                pyxel.pset(enemy_x - skull_size//4, enemy_y - skull_size//2 + bob_offset, 0)  # Left eye
                pyxel.pset(enemy_x + skull_size//4, enemy_y - skull_size//2 + bob_offset, 0)  # Right eye
                # Glowing eye effect
                if (pyxel.frame_count // 15) % 8 == 0:
                    pyxel.pset(enemy_x - skull_size//4, enemy_y - skull_size//2 + bob_offset, 8)  # Glowing left eye
                    pyxel.pset(enemy_x + skull_size//4, enemy_y - skull_size//2 + bob_offset, 8)  # Glowing right eye
            else:
                # Eyes closed/blinking
                pyxel.line(enemy_x - skull_size//4 - 1, enemy_y - skull_size//2 + bob_offset, 
                          enemy_x - skull_size//4 + 1, enemy_y - skull_size//2 + bob_offset, 0)
                pyxel.line(enemy_x + skull_size//4 - 1, enemy_y - skull_size//2 + bob_offset, 
                          enemy_x + skull_size//4 + 1, enemy_y - skull_size//2 + bob_offset, 0)
            
            # Nose - now a triangle for better skull look
            pyxel.tri(enemy_x, enemy_y - skull_size//4 + bob_offset - 1,
                     enemy_x - 1, enemy_y - skull_size//4 + bob_offset + 2,
                     enemy_x + 1, enemy_y - skull_size//4 + bob_offset + 2, 0)
            
            # Arms and legs
            arm_length = int(15 * scale)
            pyxel.line(enemy_x - body_size//2, enemy_y + body_size//4, 
                     enemy_x - body_size//2 - arm_length, enemy_y + body_size//2, 7)  # Left arm
            pyxel.line(enemy_x + body_size//2, enemy_y + body_size//4, 
                     enemy_x + body_size//2 + arm_length, enemy_y + body_size//2, 7)  # Right arm
            
            # Legs
            leg_length = int(20 * scale)
            pyxel.line(enemy_x - body_size//4, enemy_y + body_size, 
                     enemy_x - body_size//2, enemy_y + body_size + leg_length, 7)  # Left leg
            pyxel.line(enemy_x + body_size//4, enemy_y + body_size, 
                     enemy_x + body_size//2, enemy_y + body_size + leg_length, 7)  # Right leg
            
            # Weapon - sword
            pyxel.rect(enemy_x + body_size//2 + arm_length - 2, enemy_y + body_size//4 - 10, 
                      2, int(20 * scale), 5)  # Blade
            pyxel.rect(enemy_x + body_size//2 + arm_length - 5, enemy_y + body_size//4 - 10, 
                      10, 3, 10)  # Hilt
            
        elif self.battle_enemy.name == "Orc":
            # Draw orc with better proportions and more detail
            head_size = int(14 * scale)
            body_size = int(20 * scale)
            
            # Body and head
            pyxel.rect(enemy_x - head_size//2, enemy_y - head_size, head_size, head_size, 2)  # Head
            pyxel.rect(enemy_x - body_size//2, enemy_y, body_size, body_size, 2)  # Body
            
            # Armor
            pyxel.rect(enemy_x - body_size//2 + 2, enemy_y + 5, body_size - 4, 5, 5)  # Chest plate
            pyxel.rect(enemy_x - body_size//2 + 2, enemy_y + body_size - 7, body_size - 4, 7, 5)  # Belt
            
            # Face details
            pyxel.pset(enemy_x - head_size//3, enemy_y - head_size//2, 0)  # Left eye
            pyxel.pset(enemy_x + head_size//3, enemy_y - head_size//2, 0)  # Right eye
            pyxel.circ(enemy_x, enemy_y - head_size//3, 2, 8)  # Nose
            
            # Tusks
            pyxel.rect(enemy_x - head_size//3, enemy_y - head_size//4, 2, 5, 7)
            pyxel.rect(enemy_x + head_size//3 - 2, enemy_y - head_size//4, 2, 5, 7)
            
            # Arms
            arm_width = int(6 * scale)
            arm_length = int(15 * scale)
            pyxel.rect(enemy_x - body_size//2 - arm_width, enemy_y + 5, 
                      arm_width, arm_length, 2)  # Left arm
            pyxel.rect(enemy_x + body_size//2, enemy_y + 5, 
                      arm_width, arm_length, 2)  # Right arm
            
            # Weapon - axe
            pyxel.rect(enemy_x - body_size//2 - arm_width - 3, enemy_y, 3, int(25 * scale), 5)  # Handle
            pyxel.tri(enemy_x - body_size//2 - arm_width - 15, enemy_y, 
                     enemy_x - body_size//2 - arm_width - 3, enemy_y - 10,
                     enemy_x - body_size//2 - arm_width - 3, enemy_y + 10, 7)  # Blade
            
        elif self.battle_enemy.name == "Troll":
            # Draw troll with better proportions
            head_size = int(18 * scale)
            body_size = int(25 * scale)
            
            # Body and head
            pyxel.rect(enemy_x - head_size//2, enemy_y - head_size, head_size, head_size, 2)  # Head
            pyxel.rect(enemy_x - body_size//2, enemy_y, body_size, body_size * 1.2, 2)  # Body
            
            # Face details
            pyxel.circ(enemy_x - head_size//3, enemy_y - head_size//2, 3, 0)  # Left eye
            pyxel.circ(enemy_x + head_size//3, enemy_y - head_size//2, 3, 0)  # Right eye
            pyxel.circ(enemy_x, enemy_y - head_size//3, 3, 13)  # Nose
            
            # Add teeth
            for i in range(3):
                tooth_x = enemy_x - 6 + i * 6
                pyxel.rect(tooth_x, enemy_y - head_size//4, 2, 3, 7)
            
            # Arms
            arm_width = int(8 * scale)
            arm_length = int(30 * scale)
            pyxel.rect(enemy_x - body_size//2 - arm_width, enemy_y + 10, 
                      arm_width, arm_length, 2)  # Left arm
            pyxel.rect(enemy_x + body_size//2, enemy_y + 10, 
                      arm_width, arm_length, 2)  # Right arm
            
            # Weapon - massive club
            club_width = int(12 * scale)
            club_length = int(25 * scale)
            pyxel.rect(enemy_x + body_size//2 + arm_width, enemy_y, 
                      club_width, club_length, 5)  # Club
            # Add spikes to club
            for i in range(3):
                spike_y = enemy_y + 5 + i * 7
                pyxel.tri(enemy_x + body_size//2 + arm_width + club_width, spike_y,
                         enemy_x + body_size//2 + arm_width + club_width + 5, spike_y - 3,
                         enemy_x + body_size//2 + arm_width + club_width + 5, spike_y + 3, 7)
            
        else:  # Dragon
            # Draw dragon with more elaborate details
            head_size = int(16 * scale)
            body_size = int(30 * scale)
            
            # Body and head
            pyxel.rect(enemy_x - head_size//2, enemy_y - head_size, head_size, head_size, 8)  # Head
            pyxel.rect(enemy_x - body_size//2, enemy_y, body_size, body_size, 8)  # Body
            
            # Tail
            pyxel.tri(enemy_x, enemy_y + body_size,
                     enemy_x - 10, enemy_y + body_size + 30,
                     enemy_x + 10, enemy_y + body_size + 30, 8)
            
            # Face details
            pyxel.circ(enemy_x - head_size//3, enemy_y - head_size//2, 2, 0)  # Left eye
            pyxel.circ(enemy_x + head_size//3, enemy_y - head_size//2, 2, 0)  # Right eye
            pyxel.pset(enemy_x - head_size//3, enemy_y - head_size//2, 2)  # Eye glow
            pyxel.pset(enemy_x + head_size//3, enemy_y - head_size//2, 2)  # Eye glow
            
            # Horns
            pyxel.tri(enemy_x - head_size//2, enemy_y - head_size,
                     enemy_x - head_size, enemy_y - head_size - 10,
                     enemy_x - head_size//2 - 5, enemy_y - head_size + 5, 7)  # Left horn
            pyxel.tri(enemy_x + head_size//2, enemy_y - head_size,
                     enemy_x + head_size, enemy_y - head_size - 10,
                     enemy_x + head_size//2 + 5, enemy_y - head_size + 5, 7)  # Right horn
            
            # Wings
            wing_width = int(40 * scale)
            wing_height = int(30 * scale)
            pyxel.tri(enemy_x - body_size//2, enemy_y + 10,
                     enemy_x - body_size//2 - wing_width, enemy_y - wing_height,
                     enemy_x - body_size//2 - wing_width//2, enemy_y + 20, 8)  # Left wing
            pyxel.tri(enemy_x + body_size//2, enemy_y + 10,
                     enemy_x + body_size//2 + wing_width, enemy_y - wing_height,
                     enemy_x + body_size//2 + wing_width//2, enemy_y + 20, 8)  # Right wing
            
            # Wing details - membranes
            pyxel.line(enemy_x - body_size//2 - wing_width//2, enemy_y - wing_height//2,
                      enemy_x - body_size//2 - wing_width, enemy_y - wing_height//4, 8)
            pyxel.line(enemy_x + body_size//2 + wing_width//2, enemy_y - wing_height//2,
                      enemy_x + body_size//2 + wing_width, enemy_y - wing_height//4, 8)
            
            # Fire breath animation
            if pyxel.frame_count % 20 < 10:
                for i in range(7):
                    flame_size = 5 - i * 0.5
                    flame_color = 8 + (i % 3)  # Red, orange, yellow
                    pyxel.circ(enemy_x - head_size//2 - 10 - i*8, enemy_y - 5, flame_size, flame_color)
        
        # Draw enemy HP bar - larger and more detailed
        enemy_hp_percent = self.battle_enemy.hp / self.battle_enemy.max_hp
        pyxel.text(enemy_x - 25, enemy_y - 30, self.battle_enemy.name, 7)
        
        # Fancy HP bar
        bar_width = 50
        pyxel.rectb(enemy_x - bar_width//2 - 1, enemy_y - 25 - 1, bar_width + 2, 7, 5)  # Outer border
        pyxel.rect(enemy_x - bar_width//2, enemy_y - 25, bar_width, 5, 0)  # HP bar background
        pyxel.rect(enemy_x - bar_width//2, enemy_y - 25, int(bar_width * enemy_hp_percent), 5, 8)  # HP bar
        
        # Add tick marks to HP bar
        for i in range(5):
            tick_x = enemy_x - bar_width//2 + (bar_width * i) // 4
            pyxel.line(tick_x, enemy_y - 25, tick_x, enemy_y - 23, 5)
        
        # Draw battle text with improved visuals - enlarged for bigger screen
        text_box_y = HEIGHT * 0.85
        text_box_height = 60
        
        # Battle text box
        pyxel.rectb(10, text_box_y, WIDTH - 20, text_box_height, 7)  # Text box border - now white
        pyxel.rect(11, text_box_y + 1, WIDTH - 22, text_box_height - 2, 0)  # Text box background
        
        # Removed hot reload test text
        
        # Add decorative corners to text box
        pyxel.pset(10, text_box_y, 9)   # Top-left
        pyxel.pset(WIDTH - 10, text_box_y, 9)  # Top-right
        pyxel.pset(10, text_box_y + text_box_height, 9)  # Bottom-left
        pyxel.pset(WIDTH - 10, text_box_y + text_box_height, 9)  # Bottom-right
        
        # Add scroll-like decorations to the text box
        pyxel.rect(10, text_box_y, 5, 2, 4)  # Top scroll piece
        pyxel.rect(WIDTH - 15, text_box_y, 5, 2, 4)  # Top scroll piece
        pyxel.rect(10, text_box_y + text_box_height - 2, 5, 2, 4)  # Bottom scroll piece
        pyxel.rect(WIDTH - 15, text_box_y + text_box_height - 2, 5, 2, 4)  # Bottom scroll piece
        
        # Improve text rendering with shadow for better visibility
        lines = self.battle_text.split('\n')
        for i, line in enumerate(lines):
            # Add text shadow for better visibility
            pyxel.text(21, text_box_y + 10 + i * 10, line, 5)  # Dark shadow
            pyxel.text(20, text_box_y + 9 + i * 10, line, 7)  # Main text
        
        # Draw battle options in medieval style menu - right side
        options_x = WIDTH - 80
        options_y = text_box_y
        options_width = 70
        options_height = text_box_height
        
        pyxel.rectb(options_x, options_y, options_width, options_height, 7)  # Options box
        pyxel.rect(options_x + 1, options_y + 1, options_width - 2, options_height - 2, 0)  # Menu background
        
        # Add decorative corners to menu
        pyxel.pset(options_x, options_y, 9)  # Top-left
        pyxel.pset(options_x + options_width, options_y, 9)  # Top-right
        pyxel.pset(options_x, options_y + options_height, 9)  # Bottom-left
        pyxel.pset(options_x + options_width, options_y + options_height, 9)  # Bottom-right
        
        # Add title to menu
        pyxel.text(options_x + 8, options_y + 5, "ACTIONS", 9)
        pyxel.line(options_x + 5, options_y + 14, options_x + options_width - 5, options_y + 14, 5)
        
        for i, option in enumerate(self.battle_options):
            color = 10 if i == self.selected_option else 7  # Highlight selected option
            # Add a small sword icon before selected option
            if i == self.selected_option:
                pyxel.rect(options_x + 10, options_y + 20 + i * 12, 5, 2, 6)  # Sword hilt
                pyxel.line(options_x + 15, options_y + 21 + i * 12, options_x + 25, options_y + 21 + i * 12, 7)  # Sword blade
            pyxel.text(options_x + 30, options_y + 20 + i * 12, option, color)
        
        # Draw player stats and characters - on the left side
        stats_width = 100
        stats_height = text_box_height
        stats_x = 10
        stats_y = text_box_y - stats_height - 5
        
        pyxel.rectb(stats_x, stats_y, stats_width, stats_height, 7)  # Stats box
        pyxel.rect(stats_x + 1, stats_y + 1, stats_width - 2, stats_height - 2, 0)  # Stats background
        
        # Add decorative corners to stats box
        pyxel.pset(stats_x, stats_y, 9)  # Top-left
        pyxel.pset(stats_x + stats_width, stats_y, 9)  # Top-right
        pyxel.pset(stats_x, stats_y + stats_height, 9)  # Bottom-left
        pyxel.pset(stats_x + stats_width, stats_y + stats_height, 9)  # Bottom-right
        
        # Draw both character stats with health bars and fancy decorations
        
        # Draw Knight stats with decorative elements
        pyxel.text(stats_x + 5, stats_y + 5, "CHAMPION", 15)
        
        # HP bar for Knight
        hp_percent = self.player_hp / self.player_max_hp
        hp_bar_width = 60
        hp_bar_height = 5
        
        # Fancy HP bar with border and fill
        pyxel.rectb(stats_x + 25, stats_y + 15, hp_bar_width + 2, hp_bar_height + 2, 5)  # Border
        pyxel.rect(stats_x + 26, stats_y + 16, hp_bar_width, hp_bar_height, 0)  # Background
        pyxel.rect(stats_x + 26, stats_y + 16, int(hp_bar_width * hp_percent), hp_bar_height, 8)  # Fill
        
        # Text showing exact HP values
        pyxel.text(stats_x + 5, stats_y + 15, f"HP:", 7)
        pyxel.text(stats_x + 90, stats_y + 15, f"{self.player_hp}/{self.player_max_hp}", 8)
        
        # Draw Wizard stats (even though it's a single character game, showing both for visual appeal)
        pyxel.text(stats_x + 5, stats_y + 30, "WIZARD", 12)
        
        # HP bar for Wizard (showing full since it's not actively used)
        pyxel.rectb(stats_x + 25, stats_y + 40, hp_bar_width + 2, hp_bar_height + 2, 5)  # Border
        pyxel.rect(stats_x + 26, stats_y + 41, hp_bar_width, hp_bar_height, 0)  # Background
        pyxel.rect(stats_x + 26, stats_y + 41, hp_bar_width, hp_bar_height, 12)  # Fill at 100%
        
        # Text showing exact HP values
        pyxel.text(stats_x + 5, stats_y + 40, f"HP:", 7)
        pyxel.text(stats_x + 90, stats_y + 40, f"{self.companion_hp}/{self.companion_max_hp}", 12)
        
        # Draw potion count with animated icon
        potion_color = 9 if pyxel.frame_count % 30 < 15 else 12  # Pulsating potion icon
        pyxel.circ(stats_x + 40, stats_y + 25, 3, potion_color)  # Potion flask
        pyxel.rect(stats_x + 38, stats_y + 22, 5, 2, potion_color)  # Potion top
        pyxel.text(stats_x + 50, stats_y + 25, f"Potions: {self.potions}", 7)
        
        # Add battle clock/timer with animated hourglass icon
        battle_time_secs = (pyxel.frame_count // 30) % 60  # 0-59 seconds
        battle_time_mins = (pyxel.frame_count // 1800) % 60  # 0-59 minutes
        timer_color = 10  # Yellow
        
        # Draw hourglass icon
        sand_ratio = battle_time_secs / 60  # Shows sand flowing in hourglass
        hourglass_x = stats_x + 70
        hourglass_y = stats_y + 10
        
        # Hourglass outline
        pyxel.rect(hourglass_x - 3, hourglass_y - 3, 6, 8, 7)  # Outline
        
        # Sand in top/bottom based on timer
        top_sand_height = int(4 * (1 - sand_ratio))
        if top_sand_height > 0:
            pyxel.rect(hourglass_x - 2, hourglass_y - 2, 4, top_sand_height, timer_color)
        
        # Bottom sand
        bottom_sand_height = int(4 * sand_ratio)
        if bottom_sand_height > 0:
            pyxel.rect(hourglass_x - 2, hourglass_y + 2 - bottom_sand_height, 4, bottom_sand_height, timer_color)
            
        # Display time
        pyxel.text(hourglass_x + 5, hourglass_y, f"{battle_time_mins:02d}:{battle_time_secs:02d}", timer_color)
        
        # Draw player characters - improved and larger
        # Knight - with animation and effects
        knight_x = 60
        knight_y = text_box_y - 50
        knight_size = 14  # Slightly larger
        
        # Animate knight position slightly for life-like effect
        bob_offset = math.sin(pyxel.frame_count * 0.2) * 1.5
        
        # Draw battle aura (subtle glow effect)
        if self.player_hp > self.player_max_hp * 0.7:  # Full health glow
            for i in range(3):
                glow_size = 20 + i*2 + math.sin(pyxel.frame_count * 0.1) * 2
                glow_color = 9 if i % 2 == 0 else 10
                pyxel.circb(knight_x, knight_y, int(glow_size), glow_color)
        elif self.player_hp < self.player_max_hp * 0.3:  # Low health warning glow
            if pyxel.frame_count % 30 < 15:
                pyxel.circb(knight_x, knight_y, 22, 8)
        
        # Draw knight with more detail and animation
        pyxel.rect(knight_x - knight_size//2, knight_y - knight_size + bob_offset, knight_size, knight_size, 7)  # Helmet
        pyxel.rect(knight_x - knight_size//2 - 2, knight_y + bob_offset, knight_size + 4, knight_size * 1.5, PLAYER_COLOR)  # Body/Armor
        
        # Helmet details
        pyxel.line(knight_x - knight_size//2, knight_y - knight_size//2 + bob_offset, 
                  knight_x + knight_size//2, knight_y - knight_size//2 + bob_offset, 6)  # Helmet band
        
        # Knight face
        pyxel.rect(knight_x - knight_size//4, knight_y - knight_size//2 + bob_offset, knight_size//2, knight_size//4, 0)  # Visor
        
        # Shield with animated battle damage and details
        shield_x = knight_x - knight_size//2 - 8
        shield_y = knight_y + 2 + bob_offset
        pyxel.rect(shield_x, shield_y, 8, 12, 9)  # Shield base
        
        # Shield design
        pyxel.rect(shield_x + 1, shield_y + 1, 6, 10, 6)  # Shield inner
        pyxel.line(shield_x + 4, shield_y + 1, shield_x + 4, shield_y + 11, 10)  # Vertical line
        pyxel.line(shield_x + 1, shield_y + 6, shield_x + 7, shield_y + 6, 10)  # Horizontal line
        
        # Shield dents/damage when health is low
        if self.player_hp < self.player_max_hp * 0.5:
            pyxel.pset(shield_x + 2, shield_y + 3, 0)  # Dent
            pyxel.pset(shield_x + 6, shield_y + 9, 0)  # Dent
        
        # Animate sword based on frame with more dramatic movement
        sword_angle = math.sin(pyxel.frame_count * 0.1) * 0.5  # Sword slight wobble
        
        if pyxel.frame_count % 40 < 20:  # Longer animation cycle
            # Sword raised position with gleam effect
            sword_tip_x = knight_x + knight_size//2 + 10
            sword_tip_y = knight_y - 12 + bob_offset
            pyxel.line(knight_x + knight_size//2 + 2, knight_y + 2 + bob_offset, 
                     sword_tip_x, sword_tip_y, 7)  # Sword up
                     
            # Add details to sword
            pyxel.rect(knight_x + knight_size//2 + 1, knight_y + 1 + bob_offset, 3, 3, 10)  # Sword hilt
            
            # Add gleam to sword when raised
            if pyxel.frame_count % 10 < 5:
                pyxel.pset(sword_tip_x - 1, sword_tip_y + 1, 7)  # Gleam effect
        else:
            # Sword forward position with motion blur
            sword_tip_x = knight_x + knight_size//2 + 18
            sword_tip_y = knight_y + 2 + bob_offset
            
            # Create motion blur effect
            for i in range(3):
                blur_offset = i * 2
                blur_color = 5 if i > 0 else 7
                pyxel.line(knight_x + knight_size//2 + 2, knight_y + 2 + bob_offset, 
                         sword_tip_x - blur_offset, sword_tip_y, blur_color)  # Sword with motion trail
            
            # Add details to sword
            pyxel.rect(knight_x + knight_size//2 + 1, knight_y + 1 + bob_offset, 3, 3, 10)  # Sword hilt
            
        # Wizard (companion) - with enhanced magical effects and animation
        wizard_x = 120
        wizard_y = text_box_y - 50
        wizard_size = 14  # Slightly larger
        
        # Animate wizard floating effect
        float_offset = math.sin(pyxel.frame_count * 0.15) * 2
        
        # Draw magical aura around wizard
        for i in range(2):
            aura_size = 18 + i*3 + math.sin(pyxel.frame_count * 0.15) * 2
            aura_color = 1 if i % 2 == 0 else 12
            pyxel.circb(wizard_x, wizard_y + float_offset, int(aura_size), aura_color)
        
        # Draw wizard with more detail and magical effects
        WIZARD_COLOR = 12  # Light blue
        
        # Robe with flowing animation
        robe_width = wizard_size + 4 + math.sin(pyxel.frame_count * 0.2) * 1
        pyxel.rect(wizard_x - robe_width//2, wizard_y + float_offset, 
                  robe_width, wizard_size * 1.5, WIZARD_COLOR)  # Flowing robe
        
        # Hat with star
        pyxel.rect(wizard_x - wizard_size//2, wizard_y - wizard_size + float_offset, 
                  wizard_size, wizard_size, 1)  # Hat base
        
        # Hat details with animation
        hat_tip_height = 8 + math.sin(pyxel.frame_count * 0.1) * 2  # Animating hat tip
        pyxel.tri(wizard_x, wizard_y - wizard_size - hat_tip_height + float_offset, 
                 wizard_x - wizard_size//2, wizard_y - wizard_size + float_offset,
                 wizard_x + wizard_size//2, wizard_y - wizard_size + float_offset, 1)  # Animated pointed hat
        
        # Animated star on hat
        star_brightness = (pyxel.frame_count % 30) // 10
        star_color = 10 if star_brightness != 1 else 7  # Blink between yellow and white
        pyxel.pset(wizard_x, wizard_y - wizard_size - hat_tip_height//2 + float_offset, star_color)  # Star on hat
        
        # Add hat band
        pyxel.line(wizard_x - wizard_size//2, wizard_y - wizard_size + 2 + float_offset,
                  wizard_x + wizard_size//2, wizard_y - wizard_size + 2 + float_offset, 9)  # Hat band
        
        # Wizard face with expressive eyes
        eye_state = (pyxel.frame_count % 90) // 30  # 0=normal, 1=closed, 2=wide
        
        if eye_state == 1:  # Closed eyes (blinking)
            pyxel.line(wizard_x - wizard_size//4, wizard_y - wizard_size//2 + float_offset,
                      wizard_x - wizard_size//6, wizard_y - wizard_size//2 + float_offset, 7)  # Left eye closed
            pyxel.line(wizard_x + wizard_size//6, wizard_y - wizard_size//2 + float_offset,
                      wizard_x + wizard_size//4, wizard_y - wizard_size//2 + float_offset, 7)  # Right eye closed
        else:
            eye_size = 1 if eye_state == 0 else 2  # Wider eyes during casting
            pyxel.pset(wizard_x - wizard_size//4, wizard_y - wizard_size//2 + float_offset, 7)  # Left eye
            pyxel.pset(wizard_x + wizard_size//4, wizard_y - wizard_size//2 + float_offset, 7)  # Right eye
            
            # Add pupils that follow action
            pupil_offset = math.sin(pyxel.frame_count * 0.05) * 1
            pyxel.pset(wizard_x - wizard_size//4 + pupil_offset, wizard_y - wizard_size//2 + float_offset, 0)  # Left pupil
            pyxel.pset(wizard_x + wizard_size//4 + pupil_offset, wizard_y - wizard_size//2 + float_offset, 0)  # Right pupil
        
        # Mouth changes with animation
        if pyxel.frame_count % 60 < 30:
            # Normal mouth
            pyxel.rect(wizard_x - wizard_size//4, wizard_y - wizard_size//4 + float_offset, 
                      wizard_size//2, 1, 7)  # Neutral mouth
        else:
            # Casting mouth (slight smile)
            pyxel.rect(wizard_x - wizard_size//4, wizard_y - wizard_size//4 + float_offset, 
                      wizard_size//2, 1, 7)  # Mouth base
            pyxel.pset(wizard_x - wizard_size//4, wizard_y - wizard_size//4 + 1 + float_offset, 7)  # Left corner down
            pyxel.pset(wizard_x + wizard_size//4 - 1, wizard_y - wizard_size//4 + 1 + float_offset, 7)  # Right corner down
        
        # Elaborate staff with magical animation
        staff_angle = math.sin(pyxel.frame_count * 0.05) * 0.3  # Staff movement
        staff_length = 20 + math.sin(pyxel.frame_count * 0.1) * 2  # Staff extending/retracting slightly
        
        # Staff position changes based on animation cycle
        if pyxel.frame_count % 60 < 30:
            # Raised staff - charging magic
            staff_tip_x = wizard_x + wizard_size//2 + math.sin(staff_angle) * staff_length
            staff_tip_y = wizard_y - 18 + float_offset + math.cos(staff_angle) * staff_length
            
            # Draw staff with slight curve for magic feel
            pyxel.line(wizard_x + wizard_size//2, wizard_y + float_offset, 
                      staff_tip_x - 2, staff_tip_y + 8, 4)  # Lower staff
            pyxel.line(staff_tip_x - 2, staff_tip_y + 8, 
                      staff_tip_x, staff_tip_y, 4)  # Upper staff
            
            # Glowing orb with pulsating effect
            orb_size = 3 + math.sin(pyxel.frame_count * 0.2) * 1
            pyxel.circ(staff_tip_x, staff_tip_y, orb_size, 12)  # Outer glow
            pyxel.circ(staff_tip_x, staff_tip_y, orb_size - 1, 9)  # Inner glow
            
            # Add elaborate magic sparkles around orb
            for i in range(5):
                angle = pyxel.frame_count * 0.1 + i * (2 * math.pi / 5)
                distance = 5 + math.sin(pyxel.frame_count * 0.2 + i) * 2
                spark_x = staff_tip_x + int(distance * math.cos(angle))
                spark_y = staff_tip_y + int(distance * math.sin(angle))
                spark_color = (i % 3) + 8  # Cycle through colors 8, 9, 10
                pyxel.pset(spark_x, spark_y, spark_color)
        else:
            # Casting position - releasing magic
            staff_tip_x = wizard_x + wizard_size//2 + 15
            staff_tip_y = wizard_y - 5 + float_offset
            
            # Draw staff pointed forward
            pyxel.line(wizard_x + wizard_size//2, wizard_y + float_offset, 
                      staff_tip_x, staff_tip_y, 4)  # Staff
            
            # Magic blast effect
            blast_width = ((pyxel.frame_count % 30) / 30) * 15  # Growing blast
            
            # Multiple colored layers for blast
            for i in range(3):
                blast_color = 8 + i  # Red, orange, yellow
                blast_size = blast_width - i * 2
                if blast_size > 0:
                    pyxel.circb(staff_tip_x + 5 + blast_width//2, staff_tip_y, blast_size, blast_color)
            
            # Magic particles streaming from staff
            for i in range(5):
                particle_dist = ((pyxel.frame_count + i*5) % 20) / 20
                particle_x = staff_tip_x + int(particle_dist * 20)
                particle_y = staff_tip_y + int(math.sin(particle_dist * 6) * 3)
                particle_color = 8 + (i % 3)
                particle_size = 2 if particle_dist < 0.5 else 1
                pyxel.circb(particle_x, particle_y, particle_size, particle_color)
    
    def draw_game_over(self):
        pyxel.cls(0)
        
        # Draw medieval game over screen
        
        # Draw castle ruins background
        for i in range(5):
            x = 20 + i * 30
            height = 30 + i % 3 * 10
            pyxel.rect(x, HEIGHT - height, 10, height, 5)  # Broken tower
            
            # Crenellations (broken)
            if i % 2 == 0:
                pyxel.rect(x, HEIGHT - height - 5, 5, 5, 5)
        
        # Draw fancy ornate border
        for i in range(WIDTH // 8):
            pyxel.rect(i * 8, 0, 4, 4, 5)  # Top border
            pyxel.rect(i * 8, HEIGHT - 4, 4, 4, 5)  # Bottom border
        
        for i in range(HEIGHT // 8):
            pyxel.rect(0, i * 8, 4, 4, 5)  # Left border
            pyxel.rect(WIDTH - 4, i * 8, 4, 4, 5)  # Right border
            
        # Draw text with shadows
        pyxel.text(WIDTH // 2 - 28, HEIGHT // 2 - 1, "GAME OVER", 5)  # Shadow
        pyxel.text(WIDTH // 2 - 29, HEIGHT // 2 - 2, "GAME OVER", 8)
        
        # Draw stats
        play_time = int(time.time() - self.start_time)
        pyxel.text(WIDTH // 2 - 40, HEIGHT // 2 + 10, f"Your quest lasted {play_time} seconds", 7)
        pyxel.text(WIDTH // 2 - 45, HEIGHT // 2 + 20, "Press Q to quit", 7)
        
        # Draw skeleton decorations
        skull_x = WIDTH // 4
        skull_y = HEIGHT // 4
        pyxel.circ(skull_x, skull_y, 8, 7)  # Skull
        pyxel.circ(skull_x - 3, skull_y - 2, 2, 0)  # Left eye
        pyxel.circ(skull_x + 3, skull_y - 2, 2, 0)  # Right eye
        
        skull_x = WIDTH - WIDTH // 4
        pyxel.circ(skull_x, skull_y, 8, 7)  # Skull
        pyxel.circ(skull_x - 3, skull_y - 2, 2, 0)  # Left eye
        pyxel.circ(skull_x + 3, skull_y - 2, 2, 0)  # Right eye
        
        # Add a tip about screenshots
        pyxel.text(WIDTH // 2 - 75, HEIGHT - 10, "Screenshots saved to ~/.cache/fun-with-pyxel", 6)
    
    def draw_win(self):
        pyxel.cls(0)
        
        # Draw medieval victory screen
        
        # Draw castle celebration background
        pyxel.rect(WIDTH//2 - 40, HEIGHT//2 - 30, 80, 60, 5)  # Castle
        pyxel.rect(WIDTH//2 - 45, HEIGHT//2 - 50, 10, 50, 5)  # Left tower
        pyxel.rect(WIDTH//2 + 35, HEIGHT//2 - 50, 10, 50, 5)  # Right tower
        
        # Crenellations
        for i in range(8):
            pyxel.rect(WIDTH//2 - 40 + i*10, HEIGHT//2 - 35, 5, 5, 5)
        
        # Flags
        pyxel.tri(WIDTH//2 - 45, HEIGHT//2 - 60, 
                 WIDTH//2 - 45, HEIGHT//2 - 50, 
                 WIDTH//2 - 35, HEIGHT//2 - 55, 8)
                 
        pyxel.tri(WIDTH//2 + 45, HEIGHT//2 - 60, 
                 WIDTH//2 + 45, HEIGHT//2 - 50, 
                 WIDTH//2 + 35, HEIGHT//2 - 55, 8)
        
        # Draw fancy animated stars
        for i in range(20):
            x = (pyxel.frame_count * 0.5 + i * 20) % WIDTH
            y = (pyxel.frame_count * 0.3 + i * 30) % HEIGHT
            size = 1 + (pyxel.frame_count + i * 10) % 3
            pyxel.rect(x, y, size, size, 10)  # Yellow stars
        
        # Draw ornate text
        pyxel.text(WIDTH // 2 - 19, HEIGHT // 2 - 1, "VICTORY!", 5)  # Shadow
        pyxel.text(WIDTH // 2 - 20, HEIGHT // 2 - 2, "VICTORY!", 10)
        
        # Draw treasure at the bottom
        chest_x = WIDTH // 2
        chest_y = HEIGHT - 30
        pyxel.rect(chest_x - 10, chest_y, 20, 10, 4)  # Chest
        pyxel.rect(chest_x - 10, chest_y - 5, 20, 5, 9)  # Lid
        
        # Draw gold spilling out
        for i in range(8):
            gold_x = chest_x - 8 + i * 2
            gold_y = chest_y + 2 + (i % 3)
            pyxel.pset(gold_x, gold_y, 10)
        
        # Draw stats
        play_time = int(time.time() - self.start_time)
        pyxel.text(WIDTH // 2 - 65, HEIGHT // 2 + 15, f"You completed your quest in {play_time} seconds!", 7)
        pyxel.text(WIDTH // 2 - 45, HEIGHT // 2 + 25, "Press Q to quit", 7)
        
        # Add a tip about screenshots
        pyxel.text(WIDTH // 2 - 75, HEIGHT - 10, "Screenshots saved to ~/.cache/fun-with-pyxel", 6)

def show_screenshots():
    """Function to show the most recent screenshots"""
    try:
        screenshots = sorted(Path(SCREENSHOT_DIR).glob("*.png"), key=os.path.getmtime, reverse=True)
        if screenshots:
            print("\nRecent screenshots:")
            for i, screenshot in enumerate(screenshots[:5]):  # Show the 5 most recent
                print(f"{i+1}. {screenshot.name}")
        else:
            print("\nNo screenshots found.")
    except Exception as e:
        print(f"Error listing screenshots: {e}")

class SocketServer:
    """Socket server for controlling the game remotely"""
    def __init__(self, game, socket_path):
        self.game = game
        self.socket_path = socket_path
        self.server_socket = None
        self.running = False
        self.thread = None
        
    def start(self):
        """Start the socket server in a separate thread"""
        if self.running:
            return
            
        # Remove existing socket if it exists
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
            
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.socket_path), exist_ok=True)
        
        # Create socket
        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.bind(self.socket_path)
        self.server_socket.listen(5)
        
        # Set socket permissions to be readable/writable by all users
        os.chmod(self.socket_path, 0o777)
        
        # Start thread
        self.running = True
        self.thread = threading.Thread(target=self.run_server)
        self.thread.daemon = True
        self.thread.start()
        
        logging.info(f"Socket server started at {self.socket_path}")
        
    def stop(self):
        """Stop the socket server"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
                
        logging.info("Socket server stopped")
        
    def run_server(self):
        """Run the socket server loop"""
        self.server_socket.settimeout(1.0)  # 1 second timeout
        
        while self.running:
            try:
                # Accept connection
                client, _ = self.server_socket.accept()
                
                # Receive data
                data = client.recv(4096).decode('utf-8')
                
                # Parse JSON command
                command = json.loads(data)
                
                # Process command
                response = self.process_command(command)
                
                # Send response
                client.sendall(json.dumps(response).encode('utf-8'))
                
                # Close connection
                client.close()
                
            except socket.timeout:
                # Timeout is expected for non-blocking operation
                continue
            except Exception as e:
                logging.error(f"Socket server error: {str(e)}")
                if not self.running:
                    break
    
    def process_command(self, command):
        """Process a command from the socket"""
        cmd_type = command.get("command", "")
        args = command.get("args", {})
        
        logging.info(f"Received command: {cmd_type} with args: {args}")
        
        # Command handlers
        if cmd_type == "ping":
            return {"status": "success", "message": "Game is running"}
            
        elif cmd_type == "quit":
            # Schedule game to quit - can't quit immediately in this thread
            self.game.scheduled_quit = True
            return {"status": "success", "message": "Game will quit"}
            
        elif cmd_type == "move":
            direction = args.get("direction", "")
            return self.handle_move_command(direction)
            
        elif cmd_type == "action":
            action_type = args.get("type", "")
            return self.handle_action_command(action_type)
            
        elif cmd_type == "switch_character":
            character = args.get("character", "")
            return self.handle_switch_character(character)
            
        elif cmd_type == "toggle_party_control":
            return self.handle_toggle_ai()
            
        elif cmd_type == "status":
            return self.handle_status()
            
        elif cmd_type == "save_state":
            return self.handle_save_state()
            
        else:
            return {"status": "error", "message": f"Unknown command: {cmd_type}"}
    
    def handle_move_command(self, direction):
        """Handle move commands"""
        # Store movement command to be processed in the game loop
        self.game.command_queue.append(("move", direction))
        return {"status": "success", "message": f"Moving {direction}"}
    
    def handle_action_command(self, action_type):
        """Handle action commands"""
        if action_type == "screenshot":
            # Take screenshot immediately
            self.game.draw()
            path = self.game.screenshot_mgr.take_screenshot("cli_command")
            return {"status": "success", "message": "Screenshot taken", "path": path}
        else:
            # Queue other actions for the game loop
            self.game.command_queue.append(("action", action_type))
            return {"status": "success", "message": f"Action {action_type} queued"}
    
    def handle_switch_character(self, character):
        """Handle character switch commands"""
        if character == "knight":
            # Only knight is available
            return {"status": "success", "message": "Only Knight is available"}
        else:
            return {"status": "error", "message": "Only Knight character is available"}
    
    def handle_toggle_ai(self):
        """Handle AI toggle command - now just returns info about simultaneous control"""
        return {"status": "success", "message": "Both AI and human control the character simultaneously"}
        
    def handle_save_state(self):
        """Handle save state command for game reload"""
        try:
            logging.info("Handling save_state command")
            # Create GameState object
            from hot_reload import GameState
            state = GameState()
            
            # Extract game state
            game = self.game
            logging.info(f"Extracting game state: player_x={game.player_x}, player_y={game.player_y}")
            # Get attributes safely with defaults
            state.player_x = getattr(game, "player_x", 0)
            state.player_y = getattr(game, "player_y", 0)
            state.health = getattr(game, "player_hp", 25)
            state.potions = getattr(game, "potions", 0)
            state.in_battle = game.game_state == game.BATTLE  # Use game_state instead of in_battle
            state.seed = getattr(game, "seed", 0)
            state.map_data = getattr(game, "map_data", [])
            state.enemy_positions = getattr(game, "enemy_positions", [])
            state.potion_positions = getattr(game, "potion_positions", [])
        
            # Extract battle state if in battle
            game = self.game
            if game.game_state == game.BATTLE:
                logging.info("Game is in battle, extracting battle state")
                state.current_enemy = {
                    "hp": game.enemy_hp if hasattr(game, "enemy_hp") else 10,
                    "x": game.enemy_battle_x if hasattr(game, "enemy_battle_x") else 0,
                    "y": game.enemy_battle_y if hasattr(game, "enemy_battle_y") else 0,
                    "type": getattr(game, "enemy_type", 0)
                }
            
            state.game_started = True
            
            # Save to file
            state_path = os.path.expanduser("~/.cache/fun-with-pyxel/game_reload_state.pickle")
            os.makedirs(os.path.dirname(state_path), exist_ok=True)
            state.save_to_file(state_path)
            logging.info(f"Game state saved to {state_path}")
            
            return {"status": "success", "message": "Game state saved", "path": state_path}
        except Exception as e:
            logging.error(f"Error saving game state: {e}", exc_info=True)
            return {"status": "error", "message": f"Error saving game state: {e}"}
    
    def handle_status(self):
        """Get game status"""
        # Get current game state
        game = self.game
        
        status = {
            "knight_hp": game.player_hp,
            "knight_max_hp": game.player_max_hp,
            "potions": game.potions,
            "control": "simultaneous",
            "game_state": game.game_state
        }
        
        return {"status": "success", "game_status": status}

if __name__ == "__main__":
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Medieval Fantasy RPG")
        parser.add_argument("--socket", help="Path to socket file for remote control")
        parser.add_argument("--headless", action="store_true", help="Run in headless mode (no window)")
        parser.add_argument("--tail-logs", action="store_true", help="Tail the log file")
        parser.add_argument("--show-screenshots", action="store_true", help="Show recent screenshots")
        parser.add_argument("--load-state", help="Path to saved state file to load")
        args = parser.parse_args()
        
        # Check special modes
        if args.tail_logs:
            # Just tail the log file
            import subprocess
            subprocess.call(["tail", "-f", log_file])
            sys.exit(0)
        elif args.show_screenshots:
            # Display recent screenshots
            show_screenshots()
            sys.exit(0)
            
        # Start the game
        logging.info("===== STARTING NEW GAME =====")
        
        # Print controls if not headless
        if not args.headless:
            print("\nControls:")
            print("- Move: Arrow keys or WASD")
            print("- Action/Select: Space or Enter")
            print("- Screenshot: S key")
            print("- Quit: Q key")
            print("\nScreenshots are saved to: ~/.cache/fun-with-pyxel")
        
        # Initialize the game
        game = MazeGame(headless=args.headless, socket_path=args.socket)
        
        # Load state if specified
        if args.load_state and os.path.exists(args.load_state):
            try:
                from hot_reload import GameState
                state = GameState()
                if state.load_from_file(args.load_state):
                    logging.info(f"Loading game state from {args.load_state}")
                    # Add restore_state method if needed
                    if not hasattr(game, "restore_state"):
                        def restore_state(self, state):
                            """Restore game state from saved file"""
                            if not state.game_started:
                                return False
                            
                            # Restore basic variables
                            self.player_x = state.player_x
                            self.player_y = state.player_y
                            self.player_hp = state.health
                            self.potions = state.potions
                            self.in_battle = state.in_battle
                            
                            # Restore world data
                            if state.seed is not None:
                                self.seed = state.seed
                            if state.map_data is not None:
                                self.map_data = state.map_data
                            if state.enemy_positions:
                                self.enemy_positions = state.enemy_positions
                            if state.potion_positions:
                                self.potion_positions = state.potion_positions
                            
                            # Restore battle state if needed
                            if self.in_battle and state.current_enemy:
                                self.enemy_hp = state.current_enemy.get("hp", 10)
                                self.enemy_battle_x = state.current_enemy.get("x", 0)
                                self.enemy_battle_y = state.current_enemy.get("y", 0)
                                if "type" in state.current_enemy:
                                    self.enemy_type = state.current_enemy["type"]
                            
                            return True
                        
                        # Add method to the game instance
                        import types
                        game.restore_state = types.MethodType(restore_state, game)
                    
                    # Restore the game state
                    if game.restore_state(state):
                        logging.info("Game state restored successfully")
                        print("Game state restored successfully")
                    else:
                        logging.warning("Failed to restore game state")
                        print("Failed to restore game state")
            except Exception as e:
                logging.error(f"Error loading state: {e}")
                print(f"Error loading state: {e}")
        
        # Run until completion
        game.run()
        show_screenshots()  # Show screenshots after exiting
        
    except Exception as e:
        logging.error(f"Game crashed: {str(e)}", exc_info=True)
        print(f"Game crashed: {str(e)}")
        show_screenshots()  # Show screenshots even if crashed