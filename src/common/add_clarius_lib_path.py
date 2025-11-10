"""
Module for adding Clarius library path to system PATH based on OS type.
This should be called before importing the clarius streamer to ensure
proper library loading.
"""

import os
import sys
import platform
from pathlib import Path


def add_clarius_lib_path():
    """
    Add the Clarius library path to the system PATH based on the operating system.
    
    This function should be called before importing ClariusStreamer to ensure
    that the required shared libraries can be found by the system.
    
    Returns:
        bool: True if the path was successfully added, False otherwise.
    """
    try:
        # Get the current script's directory and navigate to the clarius lib path
        current_dir = Path(__file__).parent.parent
        clarius_lib_path = current_dir / "data_access" / "clarius"
        
        # Convert to absolute path
        clarius_lib_path = clarius_lib_path.resolve()
        
        # Check if the clarius directory exists
        if not clarius_lib_path.exists():
            print(f"Warning: Clarius library path does not exist: {clarius_lib_path}")
            return False
        
        # Get the current operating system
        os_type = platform.system().lower()
        
        # Add path based on OS type
        if os_type == "linux":
            # On Linux, add to LD_LIBRARY_PATH
            current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
            if str(clarius_lib_path) not in current_ld_path:
                if current_ld_path:
                    os.environ["LD_LIBRARY_PATH"] = f"{clarius_lib_path}:{current_ld_path}"
                else:
                    os.environ["LD_LIBRARY_PATH"] = str(clarius_lib_path)
                print(f"Added Clarius library path to LD_LIBRARY_PATH: {clarius_lib_path}")
        
        elif os_type == "darwin":  # macOS
            # On macOS, add to DYLD_LIBRARY_PATH
            current_dyld_path = os.environ.get("DYLD_LIBRARY_PATH", "")
            if str(clarius_lib_path) not in current_dyld_path:
                if current_dyld_path:
                    os.environ["DYLD_LIBRARY_PATH"] = f"{clarius_lib_path}:{current_dyld_path}"
                else:
                    os.environ["DYLD_LIBRARY_PATH"] = str(clarius_lib_path)
                print(f"Added Clarius library path to DYLD_LIBRARY_PATH: {clarius_lib_path}")
        
        elif os_type == "windows":
            # On Windows, add to PATH
            current_path = os.environ.get("PATH", "")
            if str(clarius_lib_path) not in current_path:
                os.environ["PATH"] = f"{clarius_lib_path};{current_path}"
                print(f"Added Clarius library path to PATH: {clarius_lib_path}")
        
        else:
            print(f"Warning: Unsupported operating system: {os_type}")
            return False
        
        # Also add to Python sys.path for module imports
        if str(clarius_lib_path) not in sys.path:
            sys.path.insert(0, str(clarius_lib_path))
        
        return True
        
    except Exception as e:
        print(f"Error adding Clarius library path: {e}")
        return False


def check_clarius_availability():
    """
    Check if Clarius libraries are available and can be imported.
    
    Returns:
        bool: True if Clarius is available, False otherwise.
    """
    try:
        # First add the library path
        add_clarius_lib_path()
        
        # Try to import the clarius streamer
        from ..data_access.clarius.clarius_streamer import ClariusStreamer
        return True
    except ImportError as e:
        print(f"Clarius SDK not available: {e}")
        return False
    except Exception as e:
        print(f"Error checking Clarius availability: {e}")
        return False


def setup_clarius_environment():
    """
    Complete setup function that adds library paths and checks availability.
    
    Returns:
        tuple: (is_available: bool, error_message: str or None)
    """
    try:
        # Add library paths
        path_added = add_clarius_lib_path()
        if not path_added:
            return False, "Failed to add Clarius library path"
        
        # Check availability
        is_available = check_clarius_availability()
        if not is_available:
            return False, "Clarius SDK libraries not found or cannot be imported"
        
        return True, None
        
    except Exception as e:
        return False, f"Error setting up Clarius environment: {e}"


if __name__ == "__main__":
    # Test the function when run directly
    print("Testing Clarius library path setup...")
    
    success = add_clarius_lib_path()
    if success:
        print("✓ Library path setup completed successfully")
        
        # Test availability
        is_available, error = setup_clarius_environment()
        if is_available:
            print("✓ Clarius SDK is available and ready to use")
        else:
            print(f"✗ Clarius SDK check failed: {error}")
    else:
        print("✗ Failed to setup library path")
