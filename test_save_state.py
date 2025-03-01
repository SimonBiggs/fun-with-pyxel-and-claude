#\!/usr/bin/env python3
import socket
import json
import sys
import os

# Socket path
SOCKET_PATH = os.path.expanduser("~/.cache/fun-with-pyxel/game_socket")

def send_command(command, args=None):
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
        
        # Connect to the game socket
        client.connect(SOCKET_PATH)
        
        # Send the message
        client.sendall(json.dumps(message).encode('utf-8'))
        
        # Wait for response
        response = client.recv(4096).decode('utf-8')
        
        # Close connection
        client.close()
        
        # Parse and return response
        print(f"Raw response: {response}")
        return json.loads(response)
    except Exception as e:
        print(f"Error communicating with the game: {str(e)}")
        return {"status": "error", "message": str(e)}

# Test save_state command
response = send_command("save_state")
print(f"Response: {response}")
