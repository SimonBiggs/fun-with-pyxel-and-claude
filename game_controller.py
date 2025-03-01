#!/usr/bin/env python3
import argparse
import socket
import os
import sys
import time
import json
import subprocess
from pathlib import Path

# Default socket file location
SOCKET_PATH = os.path.expanduser("~/.cache/fun-with-pyxel/game_socket")

def ensure_socket_dir():
    """Ensure the socket directory exists"""
    socket_dir = os.path.dirname(SOCKET_PATH)
    if not os.path.exists(socket_dir):
        os.makedirs(socket_dir, exist_ok=True)

def send_command(command, args=None, verbose=False):
    """Send a command to the running game"""
    if args is None:
        args = {}
        
    # Prepare message
    message = {
        "command": command,
        "args": args
    }
    
    try:
        # Create socket
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(5.0)  # 5 second timeout
        
        # Connect to the game socket
        client.connect(SOCKET_PATH)
        
        # Send the message
        if verbose:
            print(f"Sending: {json.dumps(message)}")
        client.sendall(json.dumps(message).encode('utf-8'))
        
        # Wait for response
        response = client.recv(4096).decode('utf-8')
        if verbose:
            print(f"Raw response: {response}")
        
        # Close connection
        client.close()
        
        # Parse and return response
        return json.loads(response)
    except socket.timeout:
        print("Socket timeout: No response received after 5 seconds")
        return {"status": "error", "message": "Socket timeout"}
    except Exception as e:
        print(f"Error communicating with the game: {str(e)}")
        return {"status": "error", "message": str(e)}

def handle_move(args):
    """Handle move command"""
    direction = args.direction.lower()
    if direction not in ["up", "down", "left", "right"]:
        print(f"Invalid direction: {direction}")
        return
        
    # First move in the requested direction  
    response = send_command("move", {"direction": direction})
    
    # Then take a screenshot to show the current state
    screenshot_response = send_command("action", {"type": "screenshot"})
    
    if screenshot_response.get("status") == "success" and "path" in screenshot_response:
        # Get the screenshot path
        screenshot_path = screenshot_response.get("path")
        
        # Convert to data URI using the image_to_datauri.py script
        try:
            import subprocess
            result = subprocess.run(
                ["python", "image_to_datauri.py", screenshot_path],
                capture_output=True, 
                text=True
            )
            # Output only the data URI - absolutely nothing else
            print(result.stdout.strip())
        except Exception as e:
            # Print nothing on error
            pass
    # No else clause - don't print anything if screenshot fails

def handle_action(args):
    """Handle action command"""
    action = args.action.lower()
    if action not in ["attack", "potion", "run", "screenshot", "force_battle"]:
        print(f"Invalid action: {action}")
        return
        
    response = send_command("action", {"type": action})
    
    # If screenshot, automatically convert to data URI and display it
    if action == "screenshot" and response.get("status") == "success":
        screenshot_path = response.get('path')
        
        # Convert to data URI using the image_to_datauri.py script
        try:
            import subprocess
            result = subprocess.run(
                ["python", "image_to_datauri.py", screenshot_path],
                capture_output=True, 
                text=True
            )
            # Output only the data URI - absolutely nothing else
            print(result.stdout.strip())
        except Exception as e:
            # Print nothing on error
            pass
    else:
        # Only print messages for non-screenshot actions
        if action != "screenshot":
            print(response.get("message", "Command sent"))

def handle_character(args):
    """Handle character switching and party control"""
    if args.toggle_ai:
        response = send_command("toggle_party_control")
        print(response.get("message", "Command sent"))
    else:
        character = args.character.lower()
        if character not in ["knight", "wizard"]:
            print(f"Invalid character: {character}")
            return
            
        # Just changes the visual focus (party leader)
        player_idx = 0 if character == "knight" else 1
        response = send_command("switch", player_idx)
        print(response.get("message", "Command sent"))

def handle_status(args):
    """Handle status request"""
    import subprocess
    response = send_command("status")
    
    if response.get("status") == "success":
        status = response.get("game_status", {})
        print("=== Game Status ===")
        print(f"Knight: HP {status.get('knight_hp', '?')}/{status.get('knight_max_hp', '?')}")
        print(f"Wizard: HP {status.get('wizard_hp', '?')}/{status.get('wizard_max_hp', '?')}")
        print(f"Potions: {status.get('potions', '?')}")
        print(f"Active Character: {status.get('active_character', '?')}")
        print(f"AI Control: {'Enabled' if status.get('ai_control', False) else 'Disabled'}")
        print(f"Game State: {status.get('game_state', '?')}")
        
        # Check if hot reload is enabled (safely)
        try:
            hot_reload_process = subprocess.check_output(
                "ps aux | grep -v grep | grep hot_reload.py", 
                shell=True, 
                stderr=subprocess.STDOUT
            ).decode('utf-8')
            
            if hot_reload_process:
                print("Hot Reload: ACTIVE - Code changes will be applied automatically")
        except subprocess.CalledProcessError:
            # Process not found, which is fine
            pass
    else:
        print(response.get("message", "Failed to get status"))

def handle_start(args):
    """Start the game in the background"""
    import subprocess
    
    # Check if game is already running
    try:
        response = send_command("ping")
        if response.get("status") == "success":
            print("Game is already running!")
            return
    except:
        pass  # Game not running, which is what we want
    
    # Start the game with the socket argument
    try:
        # Determine which script to run based on hot reload flag
        script = "hot_reload.py" if args.hot_reload else "maze_game.py"
        
        cmd = [sys.executable, script, "--socket", SOCKET_PATH]
        if args.headless:
            cmd.append("--headless")
        
        # Add watchman flag if requested and using hot reload
        if args.hot_reload and args.use_watchman:
            cmd.append("--use-watchman")
        
        # Add log redirection
        log_file = os.path.expanduser("~/.cache/fun-with-pyxel/game.log")
        with open(log_file, 'a') as log:
            log.write(f"\n\n=== Starting game at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            log.write(f"Command: {' '.join(cmd)}\n\n")
            
            proc = subprocess.Popen(
                cmd, 
                stdout=log, 
                stderr=log,
                start_new_session=True  # Ensures the process continues even if the terminal closes
            )
        
        # Wait a bit for the game to initialize
        time.sleep(2)
        
        # Check if process started successfully
        if proc.poll() is None:
            if args.hot_reload:
                print("Game started successfully with hot reload enabled")
                print(f"Logs redirected to {log_file}")
                if args.use_watchman:
                    print("Using Watchman for file watching - should be more responsive")
                else:
                    print("Using polling for file watching (1s interval)")
                print("Hot reload will automatically detect and apply code changes")
            else:
                print("Game started successfully in the background")
        else:
            print("Failed to start game. Check the log file for details:")
            print(f"Log file: {log_file}")
    except Exception as e:
        print(f"Error starting game: {str(e)}")

def handle_stop(args):
    """Stop the running game"""
    response = send_command("quit")
    print(response.get("message", "Quit command sent"))

def handle_reload(args):
    """Reload the game from scratch while preserving state (guaranteed immediate return)"""
    # Create a simple reload script that will handle everything
    reload_script = """#!/bin/bash
# Reload script created by game_controller.py
# This script is self-deleting after execution

# Log to a specific file
LOGFILE="$HOME/.cache/fun-with-pyxel/reload.log"
echo -e "\\n\\n=== Reload started at $(date) ===\\n" >> "$LOGFILE"

# 1. Try to get game state - use a separate Python script for reliability
echo "Sending save_state command..." >> "$LOGFILE"

# Create a temporary Python script that just does the save_state command
TMP_SCRIPT="$HOME/.cache/fun-with-pyxel/temp_save_state.py"
cat > $TMP_SCRIPT << EOF
#!/usr/bin/env python3
import socket
import json
import sys
import os

# Connect to socket
s = socket.socket(socket.AF_UNIX)
s.settimeout(10)
s.connect('""" + SOCKET_PATH + """')

# Send save_state command
s.sendall(json.dumps({"command": "save_state"}).encode())

# Get response
response = s.recv(4096).decode()
print(f"RAW RESPONSE: {response}")

# Parse JSON response
try:
    data = json.loads(response)
    if data.get("status") == "success" and "path" in data:
        print(f"PATH: {data['path']}")
    else:
        print(f"ERROR: {data.get('message', 'Unknown error')}")
except Exception as e:
    print(f"PARSE ERROR: {str(e)}")
EOF

chmod +x $TMP_SCRIPT
RESULT=$($TMP_SCRIPT)
echo "Script output: $RESULT" >> "$LOGFILE"

# Extract state path from the output
STATE_PATH=$(echo "$RESULT" | grep "PATH:" | cut -d' ' -f2)
if [ -z "$STATE_PATH" ]; then
    echo "Error: No state path in response" >> "$LOGFILE"
    echo "Full script output: $RESULT" >> "$LOGFILE"
    exit 1
fi
echo "Extracted state path: $STATE_PATH" >> "$LOGFILE"

echo "State saved to: $STATE_PATH" >> "$LOGFILE"

# 2. Stop the game
echo "Sending quit command..." >> "$LOGFILE"
QUIT_JSON=$(python -c "import socket, json, sys, os; s = socket.socket(socket.AF_UNIX); s.connect('""" + SOCKET_PATH + """'); s.sendall(json.dumps({'command': 'quit'}).encode()); print(s.recv(4096).decode())")
echo "Quit response: $QUIT_JSON" >> "$LOGFILE"

# Wait for the game to exit
sleep 1
echo "Waiting for socket to be available..." >> "$LOGFILE"

# 3. Start new game with saved state
PYTHON_PATH="$(which python)"
CMD="$PYTHON_PATH maze_game.py --socket """ + SOCKET_PATH + """ --load-state $STATE_PATH"
""" + ('CMD="$CMD --headless"' if args.headless else '') + """

echo "Executing command: $CMD" >> "$LOGFILE"
$CMD >> "$LOGFILE" 2>&1 &

# Self-delete this script
rm "$0"
"""

    # Write the reload script to a temporary file
    script_path = os.path.expanduser("~/.cache/fun-with-pyxel/reload_script.sh")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(script_path), exist_ok=True)
    
    # Write the script
    with open(script_path, 'w') as f:
        f.write(reload_script)
    
    # Make it executable
    os.chmod(script_path, 0o755)
    
    # Run the script in background with nohup
    subprocess.Popen(['nohup', script_path], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE,
                      start_new_session=True)
    
    # Return immediately
    print("Game reload initiated")
    print("Log file: " + os.path.expanduser("~/.cache/fun-with-pyxel/reload.log"))
    print("Use 'status' command after a few seconds to verify game is running")

def main():
    """Main CLI function"""
    ensure_socket_dir()
    
    parser = argparse.ArgumentParser(description="Control the maze game")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start the game in the background")
    start_parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    start_parser.add_argument("--hot-reload", action="store_true", help="Enable hot reload for development")
    start_parser.add_argument("--use-watchman", action="store_true", help="Use Watchman for file watching (with hot-reload)")
    
    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop the running game")
    
    # Reload command
    reload_parser = subparsers.add_parser("reload", help="Reload the game from scratch (preserving state)")
    reload_parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    
    # Move command
    move_parser = subparsers.add_parser("move", help="Move the active character")
    move_parser.add_argument("direction", choices=["up", "down", "left", "right"], 
                            help="Direction to move")
    
    # Action command
    action_parser = subparsers.add_parser("action", help="Perform an action")
    action_parser.add_argument("action", choices=["attack", "potion", "run", "screenshot", "force_battle"], 
                             help="Action to perform")
    
    # Character command
    char_parser = subparsers.add_parser("character", help="Character control")
    char_group = char_parser.add_mutually_exclusive_group(required=True)
    char_group.add_argument("--switch", dest="character", choices=["knight", "wizard"],
                          help="Switch to controlling this character")
    char_group.add_argument("--toggle-ai", action="store_true", 
                          help="Toggle AI control for the wizard")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Get game status")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle commands
    if args.command == "start":
        handle_start(args)
    elif args.command == "stop":
        handle_stop(args)
    elif args.command == "reload":
        handle_reload(args)
    elif args.command == "move":
        handle_move(args)
    elif args.command == "action":
        handle_action(args)
    elif args.command == "character":
        handle_character(args)
    elif args.command == "status":
        handle_status(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()