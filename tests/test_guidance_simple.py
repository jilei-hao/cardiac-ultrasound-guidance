#!/usr/bin/env python3
"""
Simple test to verify the 3D guidance widget renderer works in vigil environment.
Tests renderer creation and basic functionality without heavy inference pipeline.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ui.guidance_widget import GuidanceWidgetRenderer3D


def test_renderer_creation():
    """Test creating the 3D renderer and basic rendering."""
    print("=== 3D Guidance Renderer Test ===")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")

    probe_path = os.path.join('assets', 'transducer.stl')
    if not os.path.exists(probe_path):
        print(f"❌ Probe mesh not found at: {probe_path}")
        return False

    print(f"✅ Probe mesh found: {probe_path}")

    try:
        renderer = GuidanceWidgetRenderer3D(
            probe_path=probe_path,
            window_size=(300, 300),
            wireframe=False,
            render_mode='fullscreen',
        )
        print(f"✅ 3D Renderer created: {type(renderer).__name__}")

        qw, qx, qy, qz = 1.0, 0.1, 0.2, 0.1
        norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm

        frame = renderer.render_widget(qw=qw, qx=qx, qy=qy, qz=qz)
        print(f"✅ 3D Render successful: {frame.shape}")

        renderer.set_wireframe(True)
        frame_wireframe = renderer.render_widget(qw=qw, qx=qx, qy=qy, qz=qz)
        print(f"✅ 3D Wireframe render: {frame_wireframe.shape}")
        renderer.set_wireframe(False)

        print(f"Current mode: {renderer.get_render_mode()}")
        renderer.set_render_mode('overlay')
        print(f"Switched to: {renderer.get_render_mode()}")

        renderer.destroy()
        print("✅ 3D Renderer destroyed successfully")

    except Exception as e:
        print(f"❌ 3D Renderer failed: {e}")
        return False

    return True


def main():
    print("Vigil Environment Guidance Widget Test")
    print("=" * 50)

    success = test_renderer_creation()

    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
