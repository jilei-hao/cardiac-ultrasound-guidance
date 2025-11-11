"""
Example usage of the Clarius library path setup functions.

This shows how to properly initialize Clarius before importing the streamer.
"""

# Example 1: Simple setup before importing
from src.common.add_clarius_lib_path import add_clarius_lib_path

# Add library paths before any Clarius imports
success = add_clarius_lib_path()
if success:
    try:
        # Now safe to import ClariusStreamer
        from src.data_access.clarius.clarius_streamer import ClariusStreamer
        print("✅ ClariusStreamer imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import ClariusStreamer: {e}")
else:
    print("❌ Failed to setup Clarius library path")


# Example 2: Complete setup with availability check
from src.common.add_clarius_lib_path import setup_clarius_environment

def initialize_clarius():
    """
    Initialize Clarius with proper error handling.
    
    Returns:
        ClariusStreamer module or None if not available
    """
    is_available, error_message = setup_clarius_environment()
    
    if not is_available:
        print(f"Clarius not available: {error_message}")
        return None
    
    try:
        from src.data_access.clarius.clarius_streamer import ClariusStreamer
        return ClariusStreamer
    except ImportError as e:
        print(f"Failed to import ClariusStreamer: {e}")
        return None

# Usage
ClariusStreamer = initialize_clarius()
if ClariusStreamer is not None:
    print("✅ Ready to use ClariusStreamer")
    # ClariusStreamer.configure(ip, port, "research")
    # ClariusStreamer.start()
else:
    print("❌ ClariusStreamer not available")


# Example 3: Using in conditional imports (recommended pattern)
from src.common.add_clarius_lib_path import setup_clarius_environment

# Setup environment first
clarius_available, clarius_error = setup_clarius_environment()

# Conditional import
if clarius_available:
    try:
        from src.data_access.clarius.clarius_streamer import ClariusStreamer
    except ImportError:
        ClariusStreamer = None
        clarius_available = False
else:
    ClariusStreamer = None
    print(f"Clarius setup failed: {clarius_error}")

# Later in code
def get_frame_from_source(source_type):
    if source_type == 'clarius':
        if not clarius_available or ClariusStreamer is None:
            raise RuntimeError("Clarius is not available")
        return ClariusStreamer.get_frame()
    else:
        # Handle other sources
        pass