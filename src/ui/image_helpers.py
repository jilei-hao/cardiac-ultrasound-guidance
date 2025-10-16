import numpy as np
import cv2

def draw_circle_on_image(center, radius, image, bgr):
    # Draw a circle on the image
    cv2.circle(image, center, radius, bgr, -1)
    return image

def draw_border_around_image(image, width, bgr):
    # Draw a rectangle around the image
    cv2.rectangle(image, (0, 0), (image.shape[1]-1, image.shape[0]-1), bgr, width)
    return image

def rgb_to_bgr(rgb):
    return (rgb[2], rgb[1], rgb[0])

def add_rgb_circle_to_result(input_frame, rgb):
    center = (90, 22)
    radius = 5
    return draw_circle_on_image(center, radius, input_frame, rgb)

def add_rgb_border_to_result(input_frame, rgb):
    width = 40
    return draw_border_around_image(input_frame, width, rgb)


def bgr_to_hsv_tuple(bgr):
    """Convert BGR tuple to HSV tuple."""
    color = np.uint8([[list(bgr)]])
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)[0, 0]
    return int(hsv[0]), int(hsv[1]), int(hsv[2])

def hsv_to_bgr_tuple(hsv):
    """Convert HSV tuple to BGR tuple."""
    color = np.uint8([[list(hsv)]])
    bgr = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])

def adjust_color_by_uncertainty(base_bgr, uncertainty, s_min=60, s_max=255, v_min=80, v_max=255):
    """
    Map uncertainty [0,1] to saturation/value with more adaptive range.
    Lower uncertainty -> more saturated/brighter. Full saturation for uncertainty < 0.4.
    More linear/power change for uncertainty >= 0.4.
    """
    u = float(np.clip(uncertainty, 0.0, 1.0))
    h, s0, v0 = bgr_to_hsv_tuple(base_bgr)
    
    # Full saturation for uncertainty < 0.4
    if u < 0.4:
        s = s_max
        v = v_max
    else:
        # More linear/power change for uncertainty >= 0.4
        # Remap uncertainty from [0.4, 1.0] to [0.0, 1.0] for better distribution
        u_remapped = (u - 0.4) / 0.6  # Maps [0.4, 1.0] to [0.0, 1.0]
        
        # Use a gentler power curve for more linear-like behavior
        u_adjusted = u_remapped ** 0.8  # Gentler than 0.7 for more linear change
        
        s = int(round(s_min + (1.0 - u_adjusted) * (s_max - s_min)))
        v = int(round(v_min + (1.0 - u_adjusted) * (v_max - v_min)))
    
    return hsv_to_bgr_tuple((h, s, v))

def draw_contour_with_landmarks(overlay, landmarks, base_bgr, ere_threshold=6, thickness=6):
    """
    Draw contour lines connecting landmarks with uncertainty-based coloring.
    No oversampling - direct lines between consecutive points.
    """
    if len(landmarks) < 2:
        return
    
    # Draw direct lines between consecutive landmarks (no oversampling)
    for i in range(len(landmarks)):
        j = (i + 1) % len(landmarks)  # Connect last point to first
        
        # Get uncertainty for this segment
        _, _, ere1 = landmarks[i]
        _, _, ere2 = landmarks[j]
        avg_uncertainty = (ere1 + ere2) / 2.0
        uncertainty_normalized = min(avg_uncertainty / ere_threshold, 1.0)
        
        # Adjust color based on uncertainty
        color = adjust_color_by_uncertainty(base_bgr, uncertainty_normalized)
        
        # Get start and end points
        x1, y1, _ = landmarks[i]
        x2, y2, _ = landmarks[j]
        
        # Draw direct line between points (no oversampling)
        cv2.line(overlay, (int(x1), int(y1)), (int(x2), int(y2)), 
                color, thickness, lineType=cv2.LINE_AA)

def overlay_landmarks(vis_frame, landmarks_tensor, app_settings, ere_threshold=6):
    # Check if landmarks are from 4-chamber pipeline (512x512) or old pipeline (112x112)
    # 4-chamber landmarks have variable total: LV (24) + others (12 each) = 60 total
    if len(landmarks_tensor) >= 48:
        # 4-chamber landmarks are in 512x512 space
        scale_x = vis_frame.shape[1] / 512
        scale_y = vis_frame.shape[0] / 512
        
        # New bright colors (not red/yellow/green)
        chamber_colors = [
            (255, 0, 255),    # Chamber 1 - Magenta
            (0, 255, 255),    # Chamber 2 - Cyan  
            (255, 128, 0),    # Chamber 3 - Orange
            (128, 0, 255)     # Chamber 4 - Purple
        ]
        
        overlay = vis_frame.copy()
        
        # Process each chamber - LV (label 50) has 24 points, others have 12
        chamber_sizes = [24, 12, 12, 12]  # LV, LA, RA, RV
        start_idx = 0
        
        for chamber_idx in range(4):
            end_idx = start_idx + chamber_sizes[chamber_idx]
            chamber_landmarks = landmarks_tensor[start_idx:end_idx]
            start_idx = end_idx
            
            # Scale landmarks to display coordinates
            scaled_landmarks = []
            for lm in chamber_landmarks:
                x, y, ere_val = lm
                scaled_x = int(x.item() * scale_x)
                scaled_y = int(y.item() * scale_y)
                scaled_landmarks.append((scaled_x, scaled_y, ere_val.item()))
            
            # Get base color for this chamber
            base_color = chamber_colors[chamber_idx]
            
            # Draw contour lines connecting landmarks
            draw_contour_with_landmarks(overlay, scaled_landmarks, base_color, ere_threshold, thickness=6)
            
            # Draw ALL landmark points as small solid dots
            for x, y, ere_val in scaled_landmarks:
                # Calculate uncertainty for color adjustment (use ere_threshold as max for normalization)
                uncertainty = min(ere_val / ere_threshold, 1.0)
                color = adjust_color_by_uncertainty(base_color, uncertainty)
                
                # Draw slightly larger solid dot
                dot_radius = max(3, app_settings.landmark_radius // 2 + 1)  # Slightly larger dots
                cv2.circle(overlay, (x, y), dot_radius, color, -1, lineType=cv2.LINE_AA)
        
        # Blend overlay with original frame
        cv2.addWeighted(overlay, app_settings.landmark_alpha, vis_frame, 1 - app_settings.landmark_alpha, 0, vis_frame)
        
    else:
        # Old landmark logic (fallback)
        scale_x = vis_frame.shape[1] / 112
        scale_y = vis_frame.shape[0] / 112
        overlay = vis_frame.copy()

        for c, lm in enumerate(landmarks_tensor):
            x, y, ere_val = lm
            if c in [42, 45, 46]:
                ere_val = ere_val / 3
            if ere_val <= ere_threshold:
                color = app_settings.get_color_for_ere(ere_val, max_ere=ere_threshold)
                scaled_x = int(x.item() * scale_x)
                scaled_y = int(y.item() * scale_y)
                cv2.circle(overlay, (scaled_x, scaled_y), app_settings.landmark_radius, color, -1)

        cv2.addWeighted(overlay, app_settings.landmark_alpha, vis_frame, 1 - app_settings.landmark_alpha, 0, vis_frame)

    return vis_frame

def draw_text_on_frame(input_frame, text):
    # Define the font and scale
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = input_frame.shape[0] / 400
    font_color = (255, 255, 255)  # White color
    thickness = int(font_scale * 2)

    # Get the text size
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    # Calculate the position to the lower right corner
    text_x = input_frame.shape[1] - text_size[0] - 20  # 20 pixels from the right
    text_y = input_frame.shape[0] - text_size[1] - 20  # 20 pixels from the bottom

    # Draw the text on the frame
    cv2.putText(input_frame, text, (text_x, text_y), font, font_scale, font_color, thickness)

    return input_frame
    