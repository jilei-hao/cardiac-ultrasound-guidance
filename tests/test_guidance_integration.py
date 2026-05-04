#!/usr/bin/env python3
"""
Integration test for the 3D guidance widget renderer.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.ui.guidance_widget import GuidanceWidgetRenderer3D


def test_renderer_creation():
    """Test creating the 3D renderer."""
    print("=" * 60)
    print("Testing Guidance Widget Renderer")
    print("=" * 60)

    probe_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'transducer.stl')
    if not os.path.exists(probe_path):
        probe_path = '/tmp/dummy_probe.stl'
        print(f"⚠️  Probe STL not found, using dummy path: {probe_path}")
    else:
        print(f"📁 Using probe STL: {probe_path}")

    print("\n--- Testing 3D Renderer ---")
    try:
        renderer = GuidanceWidgetRenderer3D(
            probe_path=probe_path,
            render_mode='fullscreen',
            window_size=(400, 400),
            wireframe=False,
        )
        print(f"✅ 3D Renderer created: {type(renderer).__name__}")

        frame = renderer.render_widget(qw=1.0, qx=0.1, qy=0.2, qz=0.1)
        print(f"✅ 3D Render successful: {frame.shape}")

        print(f"🔄 Current mode: {renderer.get_render_mode()}")
        renderer.set_render_mode('overlay')
        print(f"🔄 Switched to: {renderer.get_render_mode()}")

        renderer.destroy()

    except Exception as e:
        print(f"❌ 3D Renderer failed: {e}")


if __name__ == "__main__":
    test_renderer_creation()

    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)
