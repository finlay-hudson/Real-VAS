import numpy as np


def ease_in_out(progress):
    """A non-linear easing function for smoother starts and stops."""
    return (1 - np.cos(progress * np.pi)) / 2


# --- MOTION COMPONENT FUNCTIONS ---

def no_motion(progress):
    """Returns a neutral value (0 for pan, 1 for zoom)."""
    return 0 if 'pan' in no_motion.__name__ else 1


def zoom_in(progress, start_zoom=1.0, end_zoom=1.4):
    """Zooms in from start_zoom to end_zoom."""
    return start_zoom + (end_zoom - start_zoom) * progress


def pan_left_to_right(progress, pan_distance=200):
    """Pans horizontally from left to right."""
    start_pan = -pan_distance / 2
    return start_pan + (pan_distance * progress)


def pan_right_to_left(progress, pan_distance=200):
    """Pans horizontally from right to left."""
    # Start on the right and subtract the distance as progress increases.
    start_pan = pan_distance / 2
    return start_pan - (pan_distance * progress)


def pan_top_to_bottom(progress, pan_distance=150):
    """Pans vertically from top to bottom."""
    start_pan = -pan_distance / 2
    return start_pan + (pan_distance * progress)


def pan_bottom_to_top(progress, pan_distance=150):
    """Pans vertically from top to bottom."""
    start_pan = pan_distance / 2
    return start_pan + (pan_distance * progress)


def rotate(progress, max_angle=5):
    """Rotates from -max_angle to +max_angle in degrees."""
    start_angle = -max_angle
    return start_angle + (2 * max_angle * progress)


def no_rotation(progress):
    """A neutral rotation value."""
    return 0


PRESETS = {
    'slow_zoom_and_pan_left': {
        'zoom': lambda p: zoom_in(p, end_zoom=1.4),
        'pan_x': lambda p: pan_left_to_right(p, pan_distance=250),
        'pan_y': no_motion,
        'rotate': no_rotation,
    },
    'slow_zoom_and_pan_right': {
        'zoom': lambda p: zoom_in(p, end_zoom=1.4),
        'pan_x': lambda p: pan_right_to_left(p, pan_distance=250),
        'pan_y': no_motion,
        'rotate': no_rotation,
    },
    'zoom_in_tilt_down': {
        'zoom': lambda p: zoom_in(p, end_zoom=1.5),
        'pan_x': no_motion,
        'pan_y': lambda p: pan_top_to_bottom(p, pan_distance=200),
        'rotate': no_rotation,
    },
    # 'static_shot': {
    #     'zoom': no_motion,
    #     'pan_x': no_motion,
    #     'pan_y': no_motion,
    #     'rotate': no_rotation,
    # },
    'pan_left': {
        'zoom': lambda p: zoom_in(p, end_zoom=1.1),
        'pan_x': lambda p: pan_left_to_right(p, pan_distance=250),
        'pan_y': no_motion,
        'rotate': no_rotation,
    },
    'pan_right': {
        'zoom': lambda p: zoom_in(p, end_zoom=1.1),
        'pan_x': lambda p: pan_right_to_left(p, pan_distance=250),
        'pan_y': no_motion,
        'rotate': no_rotation,
    },
    'pan_on_zoomed_canvas': {
        'zoom': lambda p: 1.25,  # A constant 25% zoom
        'pan_x': lambda p: pan_left_to_right(p, pan_distance=380),  # Pan 380px (safely within our 384px limit)
        'pan_y': no_motion,
        'rotate': no_rotation,
    },
    # --- New preset demonstrating rotation ---
    'dutch_angle_pan': {
        'zoom': lambda p: 1.3,  # Constant zoom
        'pan_x': lambda p: pan_left_to_right(p, pan_distance=350),
        'pan_y': no_motion,
        'rotate': lambda p: rotate(p, max_angle=5)  # Rotates from -5 to +5 degrees
    }
}
