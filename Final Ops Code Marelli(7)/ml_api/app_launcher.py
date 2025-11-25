# app_launcher.py
import os
import sys
import threading
import time
import webbrowser
import subprocess
import signal
from pathlib import Path
import socket

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Set Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ml_backend.settings')

import django
from django.core.management import execute_from_command_line
from django.core.wsgi import get_wsgi_application

class AppLauncher:
    def __init__(self):
        self.django_process = None
        self.node_red_process = None
        self.base_dir = current_dir
        
    def is_port_in_use(self, port):
        """Check if a port is already in use"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
    
    def wait_for_server(self, port, timeout=30):
        """Wait for server to start on given port"""
        for _ in range(timeout):
            if self.is_port_in_use(port):
                return True
            time.sleep(1)
        return False
    
    def start_node_red(self):
        """Start Node-RED server"""
        try:
            print("Starting Node-RED...")
            
            # Try to find Node-RED installation
            node_red_cmd = None
            possible_commands = ['node-red', 'node-red.cmd', 'npx node-red']
            
            for cmd in possible_commands:
                try:
                    # Test if command exists
                    result = subprocess.run(cmd.split()[0], capture_output=True, timeout=5)
                    node_red_cmd = cmd
                    break
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    continue
            
            if not node_red_cmd:
                print("Warning: Node-RED not found. Please install Node-RED separately.")
                return False
            
            # Start Node-RED
            self.node_red_process = subprocess.Popen(
                node_red_cmd.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.base_dir
            )
            
            # Wait for Node-RED to start
            if self.wait_for_server(1880, timeout=30):
                print("Node-RED started successfully on http://localhost:1880")
                return True
            else:
                print("Node-RED failed to start within timeout")
                return False
                
        except Exception as e:
            print(f"Error starting Node-RED: {e}")
            return False
    
    def start_django(self):
        """Start Django development server"""
        try:
            print("Setting up Django...")
            
            # Setup Django
            django.setup()
            
            # Create database tables if needed
            try:
                execute_from_command_line(['manage.py', 'migrate'])
            except:
                pass  # Ignore migration errors
            
            print("Starting Django server...")
            
            # Start Django server in a separate thread
            def run_django():
                execute_from_command_line(['manage.py', 'runserver', '127.0.0.1:8000'])
            
            django_thread = threading.Thread(target=run_django, daemon=True)
            django_thread.start()
            
            # Wait for Django to start
            if self.wait_for_server(8000, timeout=30):
                print("Django started successfully on http://localhost:8000")
                return True
            else:
                print("Django failed to start within timeout")
                return False
                
        except Exception as e:
            print(f"Error starting Django: {e}")
            return False
    
    def open_browser(self):
        """Open web browser to the application"""
        try:
            print("Opening web browser...")
            webbrowser.open('http://localhost:8000')
            return True
        except Exception as e:
            print(f"Error opening browser: {e}")
            return False
    
    def cleanup(self):
        """Clean up processes"""
        if self.node_red_process:
            try:
                self.node_red_process.terminate()
                self.node_red_process.wait(timeout=5)
            except:
                try:
                    self.node_red_process.kill()
                except:
                    pass
        
        # Django runs in thread, will be cleaned up automatically
    
    def run(self):
        """Main application entry point"""
        print("="*50)
        print("Starting Marelli Industrial Nut Detection System")
        print("="*50)
        
        try:
            # Start Node-RED first
            node_red_started = self.start_node_red()
            
            # Start Django
            django_started = self.start_django()
            
            if django_started:
                # Wait a moment for server to be fully ready
                time.sleep(2)
                
                # Open browser
                self.open_browser()
                
                print("\n" + "="*50)
                print("Application started successfully!")
                print("Django: http://localhost:8000")
                if node_red_started:
                    print("Node-RED: http://localhost:1880")
                print("="*50)
                print("\nPress Ctrl+C to stop the application")
                
                # Keep the application running
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\nShutting down...")
                    self.cleanup()
                    sys.exit(0)
            else:
                print("Failed to start Django server")
                self.cleanup()
                sys.exit(1)
                
        except Exception as e:
            print(f"Error starting application: {e}")
            self.cleanup()
            sys.exit(1)

def main():
    launcher = AppLauncher()
    launcher.run()

if __name__ == '__main__':
    main()