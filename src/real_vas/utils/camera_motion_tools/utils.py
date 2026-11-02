import cv2
import numpy as np
from PIL import Image

from real_vas.utils.camera_motion_tools.components import ease_in_out, PRESETS

from real_vas.utils.vis_utils import overlay_mask_over_image


def get_combined_bounding_box(mask, padding=0):
    """
    Finds contours in a mask and returns a single bounding box
    that encompases all of them.
    Returns (x, y, w, h) or None if no contours are found.
    """
    # Find all separate contoured areas in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Combine all points from all contours into a single list
    all_points = np.vstack(contours)

    # Get the bounding box of the combined points
    x, y, w, h = cv2.boundingRect(all_points)

    # Add optional padding
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = w + 2 * padding
    h = h + 2 * padding

    return x, y, w, h


# --- The REFINED motion component ---
# TODO this is needed if we want to update the motion to be based off the bg mask
# def run_motion_component(progress, active_preset, frame, fg_mask, bg_mask):
#     # --- CONFIGURATION ---
#     # How strongly the camera corrects itself. Lower value = smoother/slower correction.
#     # Good values are between 0.05 and 0.2.
#     SMOOTHING_FACTOR = 0.1
#
#     # The central area of the frame to keep the subject in (e.g., 0.8 = middle 80%)
#     SAFE_ZONE_SCALE = 0.8
#     # --- END CONFIGURATION ---
#
#     height, width = frame.shape[:2]
#     eased_progress = ease_in_out(progress)
#
#     # 1. Get the ideal transformation from the preset
#     ideal_zoom = active_preset['zoom'](eased_progress)
#     ideal_pan_x = active_preset['pan_x'](eased_progress)
#     ideal_pan_y = active_preset['pan_y'](eased_progress)
#     ideal_angle_deg = active_preset['rotate'](eased_progress)
#
#     # 2. Find the center of the background object
#     bbox = get_combined_bounding_box(bg_mask)
#
#     corrected_pan_x = ideal_pan_x
#     corrected_pan_y = ideal_pan_y
#
#     if bbox is not None:
#         x, y, w, h = bbox
#         # Get the current center point of the object
#         center_point = np.array([[[x + w / 2, y + h / 2]]], dtype=np.float32)
#
#         # --- Create the "ideal" transformation matrix (same as before) ---
#         H_zoom_ideal = np.array(
#             [[ideal_zoom, 0, (1 - ideal_zoom) * width / 2], [0, ideal_zoom, (1 - ideal_zoom) * height / 2], [0, 0, 1]])
#         H_pan_ideal = np.array([[1, 0, ideal_pan_x], [0, 1, ideal_pan_y], [0, 0, 1]])
#         H_rot_ideal = cv2.getRotationMatrix2D((width / 2, height / 2), ideal_angle_deg, 1)
#         H_rot_ideal = np.vstack([H_rot_ideal, [0, 0, 1]])
#         H_ideal = H_pan_ideal @ H_rot_ideal @ H_zoom_ideal
#
#         # --- Predict the new position of the object's CENTER ---
#         transformed_center = cv2.perspectiveTransform(center_point, H_ideal)
#         pred_cx, pred_cy = transformed_center[0][0]
#
#         # --- Calculate the boundaries of our "safe zone" ---
#         safe_x_min = (width - width * SAFE_ZONE_SCALE) / 2
#         safe_x_max = width - safe_x_min
#         safe_y_min = (height - height * SAFE_ZONE_SCALE) / 2
#         safe_y_max = height - safe_y_min
#
#         # --- Calculate correction needed to bring center back to the SAFE ZONE edge ---
#         pan_correction_x = 0
#         if pred_cx < safe_x_min:
#             pan_correction_x = safe_x_min - pred_cx
#         elif pred_cx > safe_x_max:
#             pan_correction_x = safe_x_max - pred_cx
#
#         pan_correction_y = 0
#         if pred_cy < safe_y_min:
#             pan_correction_y = safe_y_min - pred_cy
#         elif pred_cy > safe_y_max:
#             pan_correction_y = safe_y_max - pred_cy
#
#         # --- KEY CHANGE: Apply only a FRACTION of the correction ---
#         corrected_pan_x += pan_correction_x * SMOOTHING_FACTOR
#         corrected_pan_y += pan_correction_y * SMOOTHING_FACTOR
#
#     # 4. Set final zoom and clamp the *gently corrected* pan values
#     current_zoom = max(ideal_zoom, 1.0)
#     max_pan_x = (width - width / current_zoom) / 2
#     max_pan_y = (height - height / current_zoom) / 2
#
#     current_pan_x = np.clip(corrected_pan_x, -max_pan_x, max_pan_x)
#     current_pan_y = np.clip(corrected_pan_y, -max_pan_y, max_pan_y)
#
#     # 5. Create and apply the final, corrected transformation matrix (same as before)
#     H_zoom = np.array(
#         [[current_zoom, 0, (1 - current_zoom) * width / 2], [0, current_zoom, (1 - current_zoom) * height / 2],
#          [0, 0, 1]])
#     H_pan = np.array([[1, 0, current_pan_x], [0, 1, current_pan_y], [0, 0, 1]])
#     H_rot = cv2.getRotationMatrix2D((width / 2, height / 2), ideal_angle_deg, 1)
#     H_rot = np.vstack([H_rot, [0, 0, 1]])
#     H = H_pan @ H_rot @ H_zoom
#
#     # Apply the transformation
#     output_frame = cv2.warpPerspective(frame, H, (width, height), flags=cv2.INTER_LINEAR,
#                                        borderMode=cv2.BORDER_CONSTANT)
#     output_fg_mask = cv2.warpPerspective(fg_mask, H, (width, height), flags=cv2.INTER_NEAREST,
#                                          borderMode=cv2.BORDER_CONSTANT)
#     output_bg_mask = cv2.warpPerspective(bg_mask, H, (width, height), flags=cv2.INTER_NEAREST,
#                                          borderMode=cv2.BORDER_CONSTANT)
#
#     return output_frame, output_fg_mask, output_bg_mask, current_zoom


def run_motion_component(progress, active_preset, frame, fg_mask, bg_mask):
    eased_progress = ease_in_out(progress)

    # 1. Get the ideal zoom and pan from the preset
    ideal_zoom = active_preset['zoom'](eased_progress)
    ideal_pan_x = active_preset['pan_x'](eased_progress)
    ideal_pan_y = active_preset['pan_y'](eased_progress)

    # 2. Set the zoom and calculate the resulting safe pan limits
    current_zoom = max(ideal_zoom, 1.0)  # Ensure zoom is never less than 1

    height, width = frame.shape[:2]
    max_pan_x = (width - width / current_zoom) / 2
    max_pan_y = (height - height / current_zoom) / 2

    # 3. Clamp the pan values to the safe range
    current_pan_x = np.clip(ideal_pan_x, -max_pan_x, max_pan_x)
    current_pan_y = np.clip(ideal_pan_y, -max_pan_y, max_pan_y)

    # --- Create and apply the final transformation matrix ---
    H_zoom = np.array([[current_zoom, 0, (1 - current_zoom) * width / 2],
                       [0, current_zoom, (1 - current_zoom) * height / 2],
                       [0, 0, 1]])

    H_pan = np.array([[1, 0, current_pan_x],
                      [0, 1, current_pan_y],
                      [0, 0, 1]])

    current_angle_deg = active_preset['rotate'](eased_progress)
    # Create a rotation matrix that rotates around the center of the frame
    H_rot = cv2.getRotationMatrix2D((width / 2, height / 2), current_angle_deg, 1)
    # getRotationMatrix2D returns a 2x3 matrix, so we need to convert it to a 3x3
    H_rot = np.vstack([H_rot, [0, 0, 1]])

    # Combine transformations: Pan -> Rotate -> Zoom
    H = H_pan @ H_rot @ H_zoom

    # # This order (Pan @ Zoom) is crucial for the "pan across a zoomed canvas" effect
    # H = H_pan @ H_zoom

    # --- Apply the transformation to the image and mask ---
    output_frame = cv2.warpPerspective(frame, H, (width, height),
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    output_fg_mask = cv2.warpPerspective(fg_mask, H, (width, height),
                                         flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

    output_bg_mask = cv2.warpPerspective(bg_mask, H, (width, height),
                                         flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

    return output_frame, output_fg_mask, output_bg_mask, current_zoom, H


def create_camera_motion(frames, fg_masks, bg_masks, selected_presets='slow_zoom_and_pan_left'):
    active_preset = PRESETS[selected_presets]
    updated_frames = []
    updated_fg_masks = []
    updated_bg_masks = []
    visibility_scores = []
    all_transformation_matrix = []
    for frame_idx, (frame, fg_mask, bg_mask) in enumerate(zip(frames, fg_masks, bg_masks)):
        progress = frame_idx / (len(frames) - 1) if len(frames) > 1 else 0
        output_frame, output_fg_mask, output_bg_mask, current_zoom, transformation_matrix = run_motion_component(
            progress, active_preset, frame, fg_mask, bg_mask)
        updated_frames.append(output_frame)
        updated_fg_masks.append(output_fg_mask)
        updated_bg_masks.append(output_bg_mask)
        all_transformation_matrix.append(transformation_matrix)

        # Calculate how much of the background mask is still visible
        original_area = np.count_nonzero(bg_mask)
        warped_area = np.count_nonzero(output_bg_mask)

        if original_area > 0:
            # Normalize the warped area by the square of the zoom factor
            # Add a small epsilon (1e-6) to prevent division by zero if zoom is 0
            zoom_factor_squared = current_zoom ** 2 + 1e-6
            normalized_area = warped_area / zoom_factor_squared
            # Now calculate the visibility as a ratio of the original area
            visibility = normalized_area / original_area
        else:
            visibility = 0.0

        # The result might be slightly > 1.0 due to interpolation, so we clip it
        final_score = min(round(visibility, 2), 1.0)  # Also round as warp affine isnt perfect
        visibility_scores.append(final_score)

    return np.array(updated_frames), np.array(updated_fg_masks), np.array(updated_bg_masks), np.array(
        visibility_scores), np.array(all_transformation_matrix)


def save_outputs(out_dir, frames, fg_masks, bg_masks):
    frame_out_dir = out_dir / "frames"
    frame_out_dir.mkdir(exist_ok=True)
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(frame_out_dir / f"{i}.jpg")

    fg_mask_out_dir = out_dir / "fg_masks"
    fg_mask_out_dir.mkdir(exist_ok=True)
    for i, fg_mask in enumerate(fg_masks):
        Image.fromarray(fg_mask).save(fg_mask_out_dir / f"{i}.png")

    bg_mask_out_dir = out_dir / "bg_masks"
    bg_mask_out_dir.mkdir(exist_ok=True)
    for i, bg_mask in enumerate(bg_masks):
        Image.fromarray(bg_mask).save(bg_mask_out_dir / f"{i}.png")

    debug_out_dir = out_dir / "debug"
    debug_out_dir.mkdir(exist_ok=True)
    for i, (f, fg_m, bg_m) in enumerate(zip(frames, fg_masks, bg_masks)):
        out_img = f.copy()
        out_img = overlay_mask_over_image(out_img, fg_m, color=(255, 0, 0), alpha=0.5)
        out_img = overlay_mask_over_image(out_img, bg_m, color=(0, 255, 0), alpha=0.5)
        Image.fromarray(out_img).save(debug_out_dir / f"{i}.jpg")
