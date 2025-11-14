#!/usr/bin/env python3
"""
Quick verification that test_guidance.py will use QtImageWindow.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ui.image_window import create_dual_window_setup
from ui.image_window.window_factory import get_image_window_class

# Check which window class is selected
window_class = get_image_window_class()
print(f"✅ Default window class: {window_class.__name__}")
print(f"   Module: {window_class.__module__}")

# Create dual window setup (don't initialize, just verify creation)
main_window, guidance_window = create_dual_window_setup(
    main_title="Main Test",
    guidance_title="Guidance Test",
    width=512,
    height=512
)

print(f"\n✅ Main window type: {type(main_window).__name__}")
print(f"✅ Guidance window type: {type(guidance_window).__name__}")

if window_class.__name__ == "QtImageWindow":
    print("\n🎉 SUCCESS: test_guidance.py will use QtImageWindow!")
    print("   This provides cross-platform support (macOS, Linux, Windows)")
else:
    print(f"\n⚠️  WARNING: Using {window_class.__name__} instead of QtImageWindow")

print("\nTo run test_guidance.py:")
print("  conda run -n vigil python -m tests.test_guidance frames /path/to/frames")
