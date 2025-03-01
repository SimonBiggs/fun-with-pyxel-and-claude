# Fun with Pyxel and Claude

A collaborative maze RPG game using Pyxel (a retro game engine for Python) with AI assistance from Claude.

## Features

- Retro-style maze RPG game with exploration and battles
- Single character controlled collaboratively by both AI and human
- Socket-based CLI for controlling the game via command line
- Hot-reloading capability for game development

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/fun-with-pyxel-and-claude.git
cd fun-with-pyxel-and-claude

# Install requirements
pip install pyxel
```

## Running the Game

### Standard Mode

```bash
python maze_game.py
```

### Socket Control Mode

```bash
python maze_game.py --socket ~/.cache/fun-with-pyxel/game_socket
```

### With Hot Reload Support

```bash
python hot_reload.py --socket ~/.cache/fun-with-pyxel/game_socket
```

## Game Controls

- **Movement**: Arrow keys or WASD
- **Action/Select**: Space or Enter
- **Use Potion**: H key
- **Take Screenshot**: S key
- **Quit**: Q key

## CLI Control

```bash
# Start the game in the background
./game_controller.py start

# Start in headless mode (no window)
./game_controller.py start --headless

# Stop the running game
./game_controller.py stop

# Movement commands
./game_controller.py move up
./game_controller.py move down
./game_controller.py move left
./game_controller.py move right

# Action commands
./game_controller.py action screenshot  # Take a screenshot
./game_controller.py action potion      # Use a potion
./game_controller.py action attack      # Attack (in battle)
./game_controller.py action run         # Run (in battle)
./game_controller.py action force_battle  # Force a battle (debug)

# Get game status
./game_controller.py status
```

## Hot Reload Development

The hot reload feature allows you to modify the game code while it's running without losing the current game state. This is extremely useful for development and debugging.

### How to use it:

1. Start the game with hot reload support:
   ```bash
   python hot_reload.py --socket ~/.cache/fun-with-pyxel/game_socket
   ```

2. Edit the maze_game.py file while the game is running

3. Save the file - the game will automatically:
   - Save the current game state
   - Reload the modified code
   - Restore the game state
   - Continue running with the new code changes

### Adding Hot Reload Support to Other Projects

You can add hot reload support to your own Pyxel projects:

1. Copy hot_reload.py and maze_game_patch.py to your project
2. Modify the GameState class to include any game-specific state variables
3. Update the extract_state and restore_state methods to handle your game's specific state
4. Run your game with the hot_reload.py launcher

## Viewing Screenshots

Screenshots are saved in ~/.cache/fun-with-pyxel/. You can view them using:

```bash
python image_to_datauri.py /path/to/screenshot.png
```

## License

[MIT License](LICENSE)