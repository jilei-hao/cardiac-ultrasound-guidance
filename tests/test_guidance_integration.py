#!/usr/bin/env python3
"""
Test script for the unified guidance widget renderer with fallback.
Demonstrates both 2D and 3D rendering capabilities.
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.ui.guidance_widget import create_guidance_widget_renderer, GuidanceWidgetRendererFactory


def test_renderer_creation():
    """Test creating different types of renderers."""
    print("=" * 60)
    print("Testing Guidance Widget Renderer Factory")
    print("=" * 60)
    
    # Find probe path
    probe_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'transducer.stl')
    if not os.path.exists(probe_path):
        probe_path = '/tmp/dummy_probe.stl'
        print(f"⚠️  Probe STL not found, using dummy path: {probe_path}")
    else:
        print(f"📁 Using probe STL: {probe_path}")
    
    # Test available renderers
    available = GuidanceWidgetRendererFactory.get_available_renderers()
    print(f"🔍 Available renderers: {available}")
    
    # Test 3D renderer (if available)
    if '3d' in available:
        print("\n--- Testing 3D Renderer ---")
        try:
            renderer_3d = create_guidance_widget_renderer(
                probe_path=probe_path,
                renderer_type='3d',
                render_mode='fullscreen',
                window_size=(400, 400),
                scale_factor=2.0,
                wireframe=False
            )
            print(f"✅ 3D Renderer created: {type(renderer_3d).__name__}")
            
            # Test render with sample quaternion
            frame = renderer_3d.render_widget(qw=1.0, qx=0.1, qy=0.2, qz=0.1)
            print(f"✅ 3D Render successful: {frame.shape}")
            
            renderer_3d.destroy()
            
        except Exception as e:
            print(f"❌ 3D Renderer failed: {e}")
    
    # Test 2D renderer
    if '2d' in available:
        print("\n--- Testing 2D Renderer ---")
        try:
            renderer_2d = create_guidance_widget_renderer(
                probe_path=probe_path,
                renderer_type='2d',
                render_mode='fullscreen',
                window_size=(400, 400),
                background_color=(20, 20, 20),
                axis_thickness=3
            )
            print(f"✅ 2D Renderer created: {type(renderer_2d).__name__}")
            
            # Test render with sample quaternion
            frame = renderer_2d.render_widget(qw=1.0, qx=0.1, qy=0.2, qz=0.1)
            print(f"✅ 2D Render successful: {frame.shape}")
            
            renderer_2d.destroy()
            
        except Exception as e:
            print(f"❌ 2D Renderer failed: {e}")
    
    # Test automatic selection with fallback
    print("\n--- Testing Auto Selection with Fallback ---")
    try:
        renderer_auto = create_guidance_widget_renderer(
            probe_path=probe_path,
            renderer_type='auto',
            fallback_to_2d=True,
            render_mode='fullscreen',
            window_size=(400, 400)
        )
        print(f"✅ Auto Renderer created: {type(renderer_auto).__name__}")
        
        # Test render
        frame = renderer_auto.render_widget(qw=0.9, qx=0.1, qy=0.3, qz=0.2)
        print(f"✅ Auto Render successful: {frame.shape}")
        
        # Test mode switching
        print(f"🔄 Current mode: {renderer_auto.get_render_mode()}")
        renderer_auto.set_render_mode('overlay')
        print(f"🔄 Switched to: {renderer_auto.get_render_mode()}")
        
        renderer_auto.destroy()
        
    except Exception as e:
        print(f"❌ Auto Renderer failed: {e}")


def test_compatibility():
    """Test backward compatibility."""
    print("\n--- Testing Backward Compatibility ---")
    
    probe_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'transducer.stl')
    
    try:
        # Test the old-style import should still work through factory
        renderer = create_guidance_widget_renderer(
            probe_path=probe_path,
            render_mode='fullscreen',
            window_size=(256, 256)
        )
        print(f"✅ Backward compatibility: {type(renderer).__name__}")
        renderer.destroy()
        
    except Exception as e:
        print(f"❌ Backward compatibility failed: {e}")


if __name__ == "__main__":
    test_renderer_creation()
    test_compatibility()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)