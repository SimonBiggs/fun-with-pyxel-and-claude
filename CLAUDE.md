# CLAUDE.md - Guidelines for Pyxel Game Development

## Project Overview
Fun with Pyxel and Claude - A project using Pyxel retro game engine for Python with hot reload capabilities for seamless development

## Commands and Special Instructions
- When the user writes "memory:" followed by text, add that information to this CLAUDE.md file for future reference
- When the user says "store in your memory", it means to add the information to this CLAUDE.md file
- When running the game, run it in the background
- Design the game with a CLI interface that allows taking screenshots or performing game actions via command line
- Always use the game_controller.py script to start, stop, and interact with the game rather than running maze_game.py directly
- Don't use sleep commands in bash scripts or commands

## Key Technical Implementations
- **Collaborative Control System**: Single character controlled by both AI and human simultaneously
- **Intelligent AI Decision-Making**: Prioritizes health, resources, and goal progress
- **Socket-Based CLI**: Controls the game via Unix socket communication
- **Direct Action Processing**: Optimized battle system with direct command execution
- **Data URI Integration**: Automatic conversion of screenshots to viewable images
- **Headless Mode Support**: Run the game without visible window for automation
- **CLI Command Interface**: Python controller script to send commands to running game
- **Debug Commands**: Special commands like `force_battle` for testing
- **Hot Reload System**: Modify game code while it's running without losing game state

## Build and Run Commands
```bash
# Install requirements (first time setup)
pip install -r requirements.txt  # If requirements file exists
pip install pyxel               # Otherwise install directly

# Run game
python maze_game.py             # Run the maze RPG game (interactive mode)
python maze_game.py --socket ~/.cache/fun-with-pyxel/game_socket  # Run with socket control

# Run game with hot reload (for development)
./game_controller.py start --hot-reload  # Start with hot reload support (recommended)
python hot_reload.py --socket ~/.cache/fun-with-pyxel/game_socket  # Run hot reload directly (not preferred)

# CLI controller for the game
./game_controller.py start            # Start the game in the background
./game_controller.py start --headless # Start the game in headless mode (no window)
./game_controller.py start --hot-reload # Start with hot reload enabled
./game_controller.py start --hot-reload --use-watchman # Start with hot reload using watchman
./game_controller.py stop             # Stop the running game
./game_controller.py reload           # Reload the game while preserving state
./game_controller.py reload --headless # Reload in headless mode

# Game control commands
./game_controller.py move up|down|left|right  # Move the active character
./game_controller.py action screenshot        # Take a screenshot
./game_controller.py action potion            # Use a potion
./game_controller.py action attack            # Attack (in battle)
./game_controller.py action run               # Run (in battle)
./game_controller.py action force_battle      # Force start a battle (debug)
./game_controller.py status                   # Get game status

# Image utilities
python image_to_datauri.py <image_path> [quality]  # Convert images to data URIs

# Run tests
pytest tests/                   # Run all tests
pytest tests/test_file.py       # Run specific test file
pytest tests/test_file.py::test_function  # Run specific test
```

## Game Instructions
- **Movement**: Arrow keys or WASD
- **Action/Select**: Space or Enter
- **Switch Characters**: Tab (toggle between knight and wizard)
- **Toggle AI Control**: C key (enable/disable Claude AI for the wizard)
- **Use Potion**: H key
- **Take Screenshot**: S key
- **Quit**: Q key

## Collaborative Control Mode
The game now uses a collaborative control system:
- Single Knight character controlled by both AI and human simultaneously
- Both AI and human provide inputs for the same character
- "HUMAN+AI" label displayed above character to indicate shared control
- Both AI and human contribute to game progress

### Collaborative Control Features
- Human provides direct movement via keyboard or CLI
- AI provides supplementary movement based on game state
- AI makes intelligent decisions to help the human player
- Both sets of inputs affect the same character
- The inputs are processed independently, allowing true collaboration
- Collaborative system works both in exploration and battle modes

### AI Decision Making
The AI continues to make intelligent decisions:
- Searches for potions when health is low
- Heads toward the goal when health is good or potions are available
- Uses terrain advantages to navigate efficiently
- Explores systematically
- Follows paths and avoids dangerous areas

### CLI Collaborative Control
Control the game through the command-line interface:
- Start the game in the background with `./game_controller.py start`
- Move the character with `./game_controller.py move [direction]`
- Force a battle for testing with `./game_controller.py action force_battle`
- Attack in battle with `./game_controller.py action attack`
- Use a potion with `./game_controller.py action potion`
- Run from battle with `./game_controller.py action run`
- Get game status with `./game_controller.py status`

## Important File Locations
- **Screenshots**: `~/.cache/fun-with-pyxel/`
- **Log File**: `game_logs.txt` in the project directory

## View Screenshots
To view the most recent screenshots, you can use the `image_to_datauri.py` script:
```bash
python image_to_datauri.py /home/david/.cache/fun-with-pyxel/YourScreenshot.png [quality]
```

## Code Style Guidelines
- **Formatting**: Use Black with default settings (line length 88)
- **Linting**: Use Flake8 and pylint
- **Type Hints**: Use Python type hints for all function parameters and return values
- **Imports**: Sort using isort, group standard library, third-party, and local imports
- **Naming**:
  - Classes: PascalCase
  - Functions/Variables: snake_case
  - Constants: UPPER_SNAKE_CASE
- **Error Handling**: Use explicit exception handling with appropriate exception types
- **Documentation**: Docstrings for modules, classes, and functions using Google style

## Best Practices for Pyxel
- Keep game loop clean and organized with update and draw functions
- Organize game assets in dedicated directories
- Use Pyxel's built-in resource system for images and sounds
- For battles and UI elements, scale according to screen size with relative positioning
- Ensure battle scenes use screen height/width proportionally (e.g., `HEIGHT * 0.7`)
- For larger/higher resolution displays, limit size to 220x160 to avoid crashes

## Collaborative Game Design Patterns
- **Simultaneous Control**: Both AI and human can control the same character
- **Complementary Inputs**: Human and AI inputs complement each other
- **Shared Resources**: Both controllers share the same inventory and health
- **Unified Experience**: Both controllers work toward the same goal
- **Independent Processing**: Input processing for human and AI is independent
- **AI Assistance**: AI provides intelligent assistance based on game state
- **Terrain-Based Strategy**: AI makes decisions based on terrain advantages
- **CLI Control Interface**: Command-line tools allow programmatic control
- **Data URI Visualization**: Game state can be visualized via screenshots converted to data URIs
- **Direct Battle Commands**: Battle commands execute immediately without waiting for next frame

## Performance and Implementation Patterns

### Game Reload and Hot Reload
- **When to use hot reload**: Use during active development when making frequent small changes
- **When to use game reload**: Use when hot reload isn't working or for more substantial code changes
- **Socket timeout handling**: Always use timeouts (5-10 seconds) when communicating via sockets
- **Background processes**: Use subprocess.Popen with start_new_session=True for non-blocking operation
- **Error handling**: Always check for attribute existence with getattr() or hasattr() in state handlers
- **State persistence**: Store state in ~/.cache/fun-with-pyxel/ directory using pickle serialization
- **Debugging socket issues**: Use verbose logging and check game_logs.txt for communication problems
- **File watching options**: Watchman is more responsive than polling but requires additional setup

## Common Issues and Solutions
- **Socket Communication**: Use Unix domain sockets for fast local communication
- **Headless Mode**: Initialize Pyxel normally but without showing the window
- **Command Queue**: Process commands asynchronously to avoid blocking the game loop
- **Screenshot Management**: Save screenshots with timestamps and event descriptions
- **Path Resolution**: Always use absolute paths for file operations
- **Error Handling**: Gracefully handle invalid movement attempts
- **CLI Authentication**: No authentication needed for local Unix socket connections
- **Toggle AI Command Issue**: The `character --toggle-ai` command has an implementation issue with the error "Socket server error: 'SocketServer' object has no attribute 'handle_toggle_party_control'"
- **CLI Response Time**: All CLI commands should complete in under 0.2 seconds to avoid blocking the main thread
- **Observed Game Behavior**: The AI companion (Wizard) has pathfinding to navigate "toward potion" and uses "tactical spacing" behavior
- **Screenshot Locations**: Screenshots are saved with timestamps and descriptive names like `20250302-004906_002_cli_command.png`
- **Game Status Format**: The status command shows both characters' HP, potion count, active character, and AI control status
- **Independent Character Positioning**: When viewed in screenshots, knight and wizard can be in different positions (knight at (15,25) and wizard at (3,6))
- **Character Roles**: The knight has 25 HP max while the wizard has 20 HP max
- **Game World Generation**: The game generates a new random world with each run using seeds (e.g., "Generating world with seed: 832705")
- **Game Socket Location**: Socket for CLI control is at `/home/david/.cache/fun-with-pyxel/game_socket`
- **Resource Placement**: The game places potions and enemies during world generation
- **Screenshot Output**: Always keep data URI output clean with no additional text when taking screenshots
- **Window Resolution**: The ideal window size is 675 x 700 pixels (half of 1350 width and 700 height)
- **Performance Limits**: Pyxel may have performance issues at very high resolutions
- **Larger Minimap**: Use a 60x60 minimap size for better overview in the game
- **Terrain Generation**: Using multiple noise layers and smoothing creates more natural-looking terrain
- **Strategic Potion Placement**: Placing potions in clusters in forests and along paths improves gameplay
- **Castle Structure**: Creating larger castles with walls, gates, and connecting villages makes the map more interesting
- **Regional Connections**: Creating distinct connected regions with major trade routes provides better exploration
- **Terrain Variety**: Using different visual styles for trees, paths and other terrain features improves visual appeal
- **Map Smoothing**: Using a smoothing algorithm prevents isolated terrain tiles and creates more natural biomes
- **Direct Battle Action Processing**: Added `execute_battle_action()` and helper methods (`_process_attack()`, `_process_potion_use()`, `_process_run_attempt()`) to directly process battle actions from CLI commands
- **Debugging Commands**: Added `force_battle` command to force start a battle for testing
- **Single Character Control Model**: Modified the game to use a single character controlled simultaneously by both AI and human rather than a party system
- **Collaborative Control UI**: Added "HUMAN+AI" indicator above the character to show shared control
- **Data URI Integration**: Modified the game controller to automatically convert screenshots to Data URIs when moving for easy viewing
- **Hot Reload Implementation**: Added hot reload capability with state preservation via:
  - File watching system that detects code changes
  - State extraction before reload and restoration after reload
  - Monkey patching of the game class with state handlers
  - Pickle-based state serialization
  - Clean handling of game world and battle state
  - Integration with game_controller.py for background operation
  - Separate log file for hot reload debugging
  - Visual indicator in game showing hot reload is active
  - Watchman integration for more responsive file change detection
  - Fallback to polling when watchman isn't available

- **Game Reload System**: Added complete game reload functionality that preserves state:
  - Socket-based state saving mechanism (`save_state` command)
  - Clean game shutdown and restart with saved state
  - Background processing with immediate CLI return
  - Attribute-safe state extraction with error handling
  - Separate Python script for reliable socket communication
  - Detailed logging of reload process
  - Compatible with existing hot reload state format