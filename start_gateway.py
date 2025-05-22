#!/usr/bin/env python3
"""
Gateway starter script for EDA4GNET Framework Enhancement
"""

import os
import sys
import subprocess
import signal
import time
import argparse
from pathlib import Path


class GatewayManager:
    def __init__(self, port: int = 25333):
        """
        Initialize Gateway Manager
        
        Args:
            port: Port number for Java Gateway
        """
        self.port = port
        self.java_process = None
        self.project_root = Path(__file__).parent
        self.java_dir = self.project_root / "java"
        self.bin_dir = self.java_dir / "bin"
        self.lib_dir = self.java_dir / "lib"
        
    def find_java_home(self) -> str:
        """Find Java home directory"""
        # Try JAVA_HOME environment variable
        java_home = os.environ.get('JAVA_HOME')
        if java_home and Path(java_home).exists():
            return java_home
        
        # Try to find java executable
        try:
            result = subprocess.run(['which', 'java'], capture_output=True, text=True)
            if result.returncode == 0:
                java_path = Path(result.stdout.strip())
                # Navigate up to find JAVA_HOME
                java_home = java_path.parent.parent
                if java_home.exists():
                    return str(java_home)
        except Exception:
            pass
        
        # Default locations to check
        default_locations = [
            "/usr/lib/jvm/default-java",
            "/usr/lib/jvm/java-17-openjdk",
            "/usr/lib/jvm/java-11-openjdk",
            "/System/Library/Frameworks/JavaVM.framework/Home",
        ]
        
        for location in default_locations:
            if Path(location).exists():
                return location
        
        return None
    
    def build_classpath(self) -> str:
        """Build Java classpath"""
        classpath_parts = []
        
        # Add bin directory
        if self.bin_dir.exists():
            classpath_parts.append(str(self.bin_dir))
        
        # Add all JAR files in lib directory
        if self.lib_dir.exists():
            for jar_file in self.lib_dir.glob("*.jar"):
                classpath_parts.append(str(jar_file))
        
        return os.pathsep.join(classpath_parts)
    
    def compile_java_code(self) -> bool:
        """Compile Java code if needed"""
        print("Checking Java compilation...")
        
        # Check if compile script exists
        compile_script = self.java_dir / "compile.sh"
        if not compile_script.exists():
            print("Compile script not found. Please ensure Java code is compiled.")
            return False
        
        # Check if bin directory exists and has class files
        if self.bin_dir.exists() and list(self.bin_dir.rglob("*.class")):
            print("Java classes found, skipping compilation.")
            return True
        
        # Run compilation
        print("Compiling Java code...")
        try:
            result = subprocess.run(
                ["bash", str(compile_script)],
                cwd=str(self.java_dir),
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("Java compilation successful")
                return True
            else:
                print(f"Java compilation failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"Error running compilation: {e}")
            return False
    
    def start_gateway(self) -> bool:
        """Start Java Gateway server"""
        print(f"Starting EDA4GNET Gateway on port {self.port}...")
        
        # Build classpath
        classpath = self.build_classpath()
        if not classpath:
            print("Error: Unable to build classpath")
            return False
        
        # Build Java command
        java_cmd = [
            "java",
            "-cp", classpath,
            "main.bridge.EDA4GNETGateway",
            str(self.port)
        ]
        
        try:
            # Start Java process
            self.java_process = subprocess.Popen(
                java_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Wait a moment for startup
            time.sleep(2)
            
            # Check if process is still running
            if self.java_process.poll() is None:
                print(f"Gateway started successfully on port {self.port}")
                print(f"Process ID: {self.java_process.pid}")
                return True
            else:
                # Process died, get error output
                stdout, stderr = self.java_process.communicate()
                print(f"Gateway failed to start:")
                if stderr:
                    print(f"Error: {stderr}")
                if stdout:
                    print(f"Output: {stdout}")
                return False
                
        except Exception as e:
            print(f"Error starting gateway: {e}")
            return False
    
    def stop_gateway(self):
        """Stop Java Gateway server"""
        if self.java_process:
            print("Stopping Gateway...")
            try:
                # Send SIGTERM first
                self.java_process.terminate()
                
                # Wait for graceful shutdown
                try:
                    self.java_process.wait(timeout=5)
                    print("Gateway stopped gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if needed
                    print("Force killing gateway...")
                    self.java_process.kill()
                    self.java_process.wait()
                    print("Gateway force stopped")
                    
            except Exception as e:
                print(f"Error stopping gateway: {e}")
            
            self.java_process = None
    
    def run_interactive(self):
        """Run gateway in interactive mode"""
        def signal_handler(signum, frame):
            print("\nReceived interrupt signal")
            self.stop_gateway()
            sys.exit(0)
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Compile if needed
        if not self.compile_java_code():
            print("Cannot start gateway without compiled Java code")
            return False
        
        # Start gateway
        if not self.start_gateway():
            return False
        
        print("Gateway is running. Press Ctrl+C to stop.")
        
        try:
            # Keep the script running and monitor the Java process
            while True:
                if self.java_process.poll() is not None:
                    print("Gateway process has terminated")
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.stop_gateway()
        
        return True
    
    def test_connection(self) -> bool:
        """Test connection to the gateway"""
        try:
            from src.eda4gnet.java_bridge import JavaBridge
            
            print(f"Testing connection to gateway on port {self.port}...")
            bridge = JavaBridge(self.port)
            
            if bridge.connect():
                print("Connection test successful!")
                bridge.disconnect()
                return True
            else:
                print("Connection test failed!")
                return False
                
        except ImportError:
            print("Cannot test connection: JavaBridge module not available")
            return False
        except Exception as e:
            print(f"Connection test error: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description='Start EDA4GNET Java Gateway')
    parser.add_argument('--port', type=int, default=25333, help='Gateway port number')
    parser.add_argument('--test', action='store_true', help='Test connection to running gateway')
    parser.add_argument('--compile-only', action='store_true', help='Only compile Java code')
    
    args = parser.parse_args()
    
    gateway_manager = GatewayManager(args.port)
    
    if args.test:
        gateway_manager.test_connection()
    elif args.compile_only:
        gateway_manager.compile_java_code()
    else:
        gateway_manager.run_interactive()


if __name__ == '__main__':
    main()