#!/usr/bin/env python3
"""
Simple test to verify the unified guidance widget renderer works in vigil environment.
Tests renderer creation and basic functionality without heavy inference pipeline.
"""

import sys
import os
import time
import numpy as np

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ui.guidance_widget import create_guidance_widget_renderer


def test_renderer_creation():
    """Test creating both 2D and 3D renderers."""
    print("=== Unified Guidance Renderer Test ===")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    probe_path = os.path.join('assets', 'transducer.stl')
    if not os.path.exists(probe_path):
        print(f"❌ Probe mesh not found at: {probe_path}")
        return False
    
    print(f"✅ Probe mesh found: {probe_path}")
    
    # Test 3D renderer
    print("\n--- Testing 3D Renderer ---")
    try:
        renderer_3d = create_guidance_widget_renderer(
            probe_path=probe_path,
            renderer_type='3d',
            window_size=(300, 300),
            scale_factor=1.5,
            wireframe=False,
            render_mode='fullscreen'
        )
        print(f"✅ 3D Renderer created: {type(renderer_3d).__name__}")
        
        # Test basic rendering
        qw, qx, qy, qz = 1.0, 0.1, 0.2, 0.1
        # Normalize quaternion
        norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
        
        frame_3d = renderer_3d.render_widget(qw=qw, qx=qx, qy=qy, qz=qz)
        print(f"✅ 3D Render successful: {frame_3d.shape}")
        
        # Test wireframe mode
        renderer_3d.set_wireframe(True)
        frame_wireframe = renderer_3d.render_widget(qw=qw, qx=qx, qy=qy, qz=qz)
        print(f"✅ 3D Wireframe render: {frame_wireframe.shape}")
        renderer_3d.set_wireframe(False)
        
        # Test mode switching
        print(f"Current mode: {renderer_3d.get_render_mode()}")
        renderer_3d.set_render_mode('overlay')
        print(f"Switched to: {renderer_3d.get_render_mode()}")
        
        renderer_3d.destroy()
        print("✅ 3D Renderer destroyed successfully")
        
    except Exception as e:
        print(f"❌ 3D Renderer failed: {e}")
        return False
    
    # Test 2D renderer
    print("\n--- Testing 2D Renderer ---")
    try:
        renderer_2d = create_guidance_widget_renderer(
            probe_path=probe_path,
            renderer_type='2d',
            window_size=(300, 300),
            background_color=(20, 20, 40),
            axis_thickness=3,
            probe_scale=0.7,
            render_mode='fullscreen'
        )
        print(f"✅ 2D Renderer created: {type(renderer_2d).__name__}")
        
        # Test basic rendering (2D has different signature)
        bg_image = np.zeros((300, 300), dtype=np.uint8)
        rotation_matrix = np.eye(3)  # Identity matrix
        
        frame_2d = renderer_2d.render_widget(
            bg_image, qw, qx, qy, qz, 
            predicted_rotation_matrix=rotation_matrix
        )
        print(f"✅ 2D Render successful: {frame_2d.shape}")
        
        renderer_2d.destroy()
        print("✅ 2D Renderer destroyed successfully")
        
    except Exception as e:
        print(f"❌ 2D Renderer failed: {e}")
        return False
    
    # Test auto renderer
    print("\n--- Testing Auto Renderer ---")
    try:
        renderer_auto = create_guidance_widget_renderer(
            probe_path=probe_path,
            renderer_type='auto',
            window_size=(300, 300),
            fallback_to_2d=True,
            render_mode='fullscreen'
        )
        print(f"✅ Auto Renderer created: {type(renderer_auto).__name__}")
        
        # Test with the appropriate signature based on renderer type
        if 'GuidanceWidgetRenderer3D' in str(type(renderer_auto)):
            frame_auto = renderer_auto.render_widget(qw=qw, qx=qx, qy=qy, qz=qz)
        else:
            frame_auto = renderer_auto.render_widget(
                bg_image, qw, qx, qy, qz, 
                predicted_rotation_matrix=rotation_matrix
            )
        print(f"✅ Auto render successful: {frame_auto.shape}")
        
        renderer_auto.destroy()
        print("✅ Auto Renderer destroyed successfully")
        
    except Exception as e:
        print(f"❌ Auto Renderer failed: {e}")
        return False
    
    return True


def test_renderer_switching():
    """Test dynamic renderer switching."""
    print("\n--- Testing Dynamic Renderer Switching ---")
    
    probe_path = os.path.join('assets', 'transducer.stl')
    
    try:
        # Start with 3D
        renderer = create_guidance_widget_renderer(
            probe_path=probe_path,
            renderer_type='3d',
            window_size=(200, 200)
        )
        print(f"Initial: {type(renderer).__name__}")
        
        # Render a frame
        frame1 = renderer.render_widget(qw=1.0, qx=0.1, qy=0.2, qz=0.1)
        print(f"3D frame: {frame1.shape}")
        
        renderer.destroy()
        
        # Switch to 2D
        renderer = create_guidance_widget_renderer(
            probe_path=probe_path,
            renderer_type='2d',
            window_size=(200, 200)
        )
        print(f"Switched to: {type(renderer).__name__}")
        
        # Render with 2D signature
        bg_image = np.zeros((200, 200), dtype=np.uint8)
        frame2 = renderer.render_widget(
            bg_image, 1.0, 0.1, 0.2, 0.1,
            predicted_rotation_matrix=np.eye(3)
        )
        print(f"2D frame: {frame2.shape}")
        
        renderer.destroy()
        print("✅ Dynamic switching successful")
        
    except Exception as e:
        print(f"❌ Dynamic switching failed: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("Vigil Environment Guidance Widget Test")
    print("=" * 50)
    
    success = True
    
    # Test 1: Basic renderer creation
    if not test_renderer_creation():
        success = False
    
    # Test 2: Dynamic switching
    if not test_renderer_switching():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed!")
        print("✅ Vigil environment is working correctly with unified renderers")
    else:
        print("❌ Some tests failed!")
        print("⚠️  Check the error messages above")
    
    print("=" * 50)
    

if __name__ == "__main__":
    main()