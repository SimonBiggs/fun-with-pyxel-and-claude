#!/usr/bin/env python3

import os
import sys
import time
import json
import importlib
import pickle
import threading
import signal
from pathlib import Path
from typing import Any, Dict, Optional, List
import pywatchman

# Game modules to watch for changes
WATCHED_MODULES = [
    "maze_game",
    # Add other modules that might be edited during gameplay
]

class GameState:
    """Container for game state that persists across reloads"""
    
    def __init__(self):
        self.player_x: int = 0
        self.player_y: int = 0
        self.health: int = 25
        self.potions: int = 0
        self.in_battle: bool = False
        self.seed: Optional[int] = None
        self.map_data: Optional[List[List[int]]] = None
        self.enemy_positions: List[tuple] = []
        self.potion_positions: List[tuple] = []
        self.current_enemy: Dict[str, Any] = {}
        self.game_started: bool = False
        # Add any other state variables that need to persist
    
    def save_to_file(self, filepath: str) -> None:
        """Save game state to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.__dict__, f)
    
    def load_from_file(self, filepath: str) -> bool:
        """Load game state from file"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    self.__dict__.update(pickle.load(f))
                return True
            return False
        except Exception as e:
            print(f"Error loading game state: {e}")
            return False


class HotReloader:
    """Watches for file changes and reloads game modules using Watchman"""
    
    def __init__(self, game_state_path: str = "/tmp/maze_game_state.pickle"):
        self.game_state = GameState()
        self.game_state_path = game_state_path
        self.module_files: Dict[str, str] = {}  # Maps module names to file paths
        self.game_instance = None
        self.original_modules = set(sys.modules.keys())
        self.stopping = False
        self.watcher_thread = None
        self.watchman_client = None
    
    def start_watcher(self):
        """Start the file watcher thread using Watchman"""
        self.watcher_thread = threading.Thread(target=self._watch_for_changes, daemon=True)
        self.watcher_thread.start()
    
    def _setup_watchman(self):
        """Setup Watchman client and subscription"""
        try:
            self.watchman_client = pywatchman.client()
            
            # Get the root directory to watch
            root_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Check if watchman is available
            self.watchman_client.query('version')
            
            # Check if watch already exists
            try:
                self.watchman_client.query('watch-list')
            except pywatchman.WatchmanError as e:
                print(f"[HOT RELOAD] Watchman error: {e}")
                return False
            
            print(f"[HOT RELOAD] Setting up watchman for directory: {root_dir}")
            self.watchman_client.query('watch', root_dir)
            
            # Collect paths of all modules we're watching
            for module_name in WATCHED_MODULES:
                if module_name in sys.modules:
                    module = sys.modules[module_name]
                    if hasattr(module, "__file__") and module.__file__:
                        self.module_files[module_name] = module.__file__
                        print(f"[HOT RELOAD] Watching file: {module.__file__}")
            
            return True
            
        except Exception as e:
            print(f"[HOT RELOAD] Error setting up watchman: {e}")
            return False
    
    def _watch_for_changes(self):
        """Watch files for changes using Watchman and trigger reload when needed"""
        if not self._setup_watchman():
            print("[HOT RELOAD] Falling back to polling for file changes")
            self._watch_for_changes_polling()
            return
            
        # Build file expression to watch
        file_patterns = [os.path.basename(file_path) for file_path in self.module_files.values()]
        
        # Create a watchman subscription for each file
        root_dir = os.path.dirname(os.path.abspath(__file__))
        sub_name = 'pyxel-hot-reload'
        
        try:
            self.watchman_client.query('subscribe', root_dir, sub_name, {
                'expression': ['name', file_patterns, 'wholename'],
                'fields': ['name', 'exists', 'type']
            })
            
            print(f"[HOT RELOAD] Watchman subscription created for {', '.join(file_patterns)}")
            
            # Process subscription updates
            while not self.stopping:
                try:
                    result = self.watchman_client.receive()
                    if 'subscription' in result and result['subscription'] == sub_name:
                        if 'files' in result and result['files']:
                            print(f"[HOT RELOAD] Changes detected: {result['files']}")
                            self._perform_reload()
                except Exception as e:
                    print(f"[HOT RELOAD] Error receiving watchman updates: {e}")
                    time.sleep(1)
        except Exception as e:
            print(f"[HOT RELOAD] Watchman subscription error: {e}")
            print("[HOT RELOAD] Falling back to polling for file changes")
            self._watch_for_changes_polling()
    
    def _watch_for_changes_polling(self):
        """Fallback method that polls for file changes"""
        # Get initial timestamps
        module_times = {}
        for module_name in WATCHED_MODULES:
            if module_name in sys.modules:
                module = sys.modules[module_name]
                if hasattr(module, "__file__") and module.__file__:
                    module_times[module.__file__] = os.path.getmtime(module.__file__)
        
        # Watch for changes
        while not self.stopping:
            reload_needed = False
            
            for module_name in WATCHED_MODULES:
                if module_name in sys.modules:
                    module = sys.modules[module_name]
                    if hasattr(module, "__file__") and module.__file__:
                        # Check module file
                        module_file = module.__file__
                        current_mtime = os.path.getmtime(module_file)
                        prev_mtime = module_times.get(module_file, 0)
                        
                        if current_mtime > prev_mtime:
                            print(f"[HOT RELOAD] Change detected in {module_file}")
                            module_times[module_file] = current_mtime
                            reload_needed = True
            
            if reload_needed:
                self._perform_reload()
            
            time.sleep(1)
    
    def _perform_reload(self):
        """Reload changed modules and restore game state"""
        print("[HOT RELOAD] Reloading game modules...")
        
        # Capture current game state if game is running
        if self.game_instance and hasattr(self.game_instance, "extract_state"):
            print("[HOT RELOAD] Extracting current game state...")
            self.game_instance.extract_state(self.game_state)
            self.game_state.save_to_file(self.game_state_path)
            print(f"[HOT RELOAD] Game state saved to {self.game_state_path}")
        
        # Reload modules
        for module_name in WATCHED_MODULES:
            if module_name in sys.modules:
                print(f"[HOT RELOAD] Reloading module: {module_name}")
                try:
                    importlib.reload(sys.modules[module_name])
                    print(f"[HOT RELOAD] Successfully reloaded {module_name}")
                except Exception as e:
                    print(f"[HOT RELOAD] Error reloading {module_name}: {str(e)}")
        
        # Restart game with saved state
        self._restart_game()
    
    def _restart_game(self):
        """Restart the game and restore state"""
        try:
            # Import the main game module (will get the freshly reloaded version)
            import maze_game
            
            # Stop existing game if it's running
            if self.game_instance and hasattr(self.game_instance, "stop"):
                print("[HOT RELOAD] Stopping current game instance...")
                self.game_instance.stop()
            
            # Create new game instance with same parameters
            kwargs = {}
            if hasattr(self, 'socket_path') and self.socket_path:
                kwargs["socket_path"] = self.socket_path
                print(f"[HOT RELOAD] Using socket: {self.socket_path}")
            
            # Also pass any headless flag
            if hasattr(self, 'headless') and self.headless:
                kwargs["headless"] = True
                print("[HOT RELOAD] Using headless mode")
            
            print("[HOT RELOAD] Creating new game instance...")
            self.game_instance = maze_game.MazeGame(**kwargs)
            
            # Restore state if we have it
            if self.game_state.game_started:
                print("[HOT RELOAD] Restoring previous game state...")
                if hasattr(self.game_instance, "restore_state"):
                    success = self.game_instance.restore_state(self.game_state)
                    if success:
                        print("[HOT RELOAD] Game state successfully restored")
                    else:
                        print("[HOT RELOAD] Failed to restore game state")
            
            # Start the game again
            if hasattr(self.game_instance, "start"):
                print("[HOT RELOAD] Starting game with new code...")
                self.game_instance.start()
            elif hasattr(self.game_instance, "run"):
                print("[HOT RELOAD] Running game with new code...")
                # Don't call run() directly as it's blocking - game will handle this
            else:
                print("[HOT RELOAD] Warning: No start() or run() method found")
        except Exception as e:
            print(f"[HOT RELOAD] Error restarting game: {e}")
    
    def start_game(self, socket_path=None, headless=False):
        """Start the game with hot reloading enabled"""
        try:
            # Store parameters for reuse during restarts
            self.socket_path = socket_path
            self.headless = headless
            
            print(f"[HOT RELOAD] Starting game with hot reload...")
            if socket_path:
                print(f"[HOT RELOAD] Using socket: {socket_path}")
            if headless:
                print("[HOT RELOAD] Running in headless mode")
            
            # Check if there's a saved state
            state_exists = self.game_state.load_from_file(self.game_state_path)
            if state_exists:
                print(f"[HOT RELOAD] Found existing state file: {self.game_state_path}")
            
            # Import and start the game
            import maze_game
            
            # Patch the maze_game with state extraction and restoration methods
            self._patch_game_module(maze_game)
            
            # Create and store game instance
            kwargs = {}
            if socket_path:
                kwargs["socket_path"] = socket_path
            
            self.game_instance = maze_game.MazeGame(**kwargs)
            
            # Restore state if available
            if state_exists and hasattr(self.game_instance, "restore_state"):
                print("Restoring previous game state...")
                self.game_instance.restore_state(self.game_state)
            
            # Start watcher thread
            self.start_watcher()
            
            # Run the game
            if hasattr(self.game_instance, "run"):
                self.game_instance.run()
        except KeyboardInterrupt:
            pass
        finally:
            self.stopping = True
            if self.watcher_thread:
                self.watcher_thread.join(timeout=0.5)
    
    def _patch_game_module(self, module):
        """Patch the game module with required methods if they don't exist"""
        
        # Only patch if the MazeGame class exists in the module
        if not hasattr(module, "MazeGame"):
            print("Warning: Could not patch game module, MazeGame class not found")
            return
        
        MazeGame = module.MazeGame
        
        # Add extract_state method if it doesn't exist
        if not hasattr(MazeGame, "extract_state"):
            def extract_state(self, state):
                """Extract current game state for hot reloading"""
                # Extract player position and stats
                state.player_x = self.knight_x
                state.player_y = self.knight_y
                state.health = self.knight_hp
                state.potions = self.potions
                state.in_battle = self.in_battle
                
                # Extract world data
                state.seed = self.seed
                state.map_data = self.map_data
                state.enemy_positions = self.enemy_positions
                state.potion_positions = self.potion_positions
                
                # Extract battle state if in battle
                if self.in_battle:
                    state.current_enemy = {
                        "hp": self.enemy_hp,
                        "x": self.enemy_battle_x,
                        "y": self.enemy_battle_y,
                        "type": getattr(self, "enemy_type", 0)
                    }
                
                state.game_started = True
                return state
            
            MazeGame.extract_state = extract_state
        
        # Add restore_state method if it doesn't exist
        if not hasattr(MazeGame, "restore_state"):
            def restore_state(self, state):
                """Restore game state after hot reloading"""
                # Check if we have game state
                if not state.game_started:
                    return False
                
                # Restore basic variables
                self.knight_x = state.player_x
                self.knight_y = state.player_y
                self.knight_hp = state.health
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
            
            MazeGame.restore_state = restore_state


def main():
    """Start the game with hot reloading enabled"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run maze game with hot reloading")
    parser.add_argument("--socket", help="Socket path for game controller")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no window)")
    parser.add_argument("--use-watchman", action="store_true", help="Force use of Watchman (otherwise falls back to polling)")
    args = parser.parse_args()
    
    socket_path = args.socket
    headless = args.headless
    use_watchman = args.use_watchman
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n[HOT RELOAD] Exiting hot reload system...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create cache directory if it doesn't exist
    socket_dir = os.path.dirname(socket_path) if socket_path else None
    if socket_dir and not os.path.exists(socket_dir):
        os.makedirs(socket_dir)
    
    # Create state directory if needed
    state_dir = os.path.dirname("/tmp/maze_game_state.pickle")
    if state_dir and not os.path.exists(state_dir):
        os.makedirs(state_dir)
    
    print("[HOT RELOAD] Starting hot reload system...")
    
    # Check if watchman is available
    if use_watchman:
        try:
            client = pywatchman.client()
            version = client.query('version')
            print(f"[HOT RELOAD] Using Watchman {version.get('version', 'unknown version')}")
        except Exception as e:
            print(f"[HOT RELOAD] Watchman error: {e}")
            print("[HOT RELOAD] Will fall back to polling mechanism")
    else:
        print("[HOT RELOAD] Using polling mechanism (use --use-watchman to try Watchman instead)")
    
    print("[HOT RELOAD] Any changes to maze_game.py will be automatically detected and applied")
    
    # Start game with hot reloading
    reloader = HotReloader()
    reloader.start_game(socket_path=socket_path, headless=headless)


if __name__ == "__main__":
    main()