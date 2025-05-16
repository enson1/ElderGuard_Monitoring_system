# --- START OF REFACTORED detection.py ---

import cv2
import time
import math
import logging
import argparse
import telepot # Keep for standalone runner only
import threading
import numpy as np
import datetime
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' # Suppresses a common harmless warning
import torch
from ultralytics import YOLO
from collections import deque, defaultdict
import traceback
import uuid
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not installed. MediaPipe pose estimation will not be available.")
    logging.warning("MediaPipe not installed. MediaPipe pose estimation will not be available.")


# Set up logging (remains global for the file)
logging.basicConfig(filename='health_monitor.log', level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

# --- Telegram Bot Config (ONLY for Standalone Runner) ---
# !!! SECURITY WARNING: Move TOKEN/CHAT_ID to environment variables or secure config !!!
#     These are left here *only* to keep the standalone runner functional *as is*.
#     In app.py, you should handle alerting separately without these hardcoded values.
_STANDALONE_BOT_TOKEN = os.environ.get('TELEGRAM_TOKEN', '7724847539:AAG9nxThI_w_4ZRP8bcoQur3Ef1Lem1Nr3w') # Example: Use env var or default
_STANDALONE_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '641970655') # Example: Use env var or default
_standalone_bot = None
if _STANDALONE_BOT_TOKEN and _STANDALONE_CHAT_ID:
    try:
        _standalone_bot = telepot.Bot(_STANDALONE_BOT_TOKEN)
        print(f"Standalone Telegram Bot initialized for Chat ID: {_STANDALONE_CHAT_ID}")
    except Exception as e:
        print(f"Warning: Failed to initialize standalone Telegram bot: {e}")
        logging.warning(f"Failed to initialize standalone Telegram bot: {e}")
else:
    print("Warning: Standalone Telegram Bot TOKEN or CHAT_ID not configured.")
    logging.warning("Standalone Telegram Bot TOKEN or CHAT_ID not configured.")


# --- Core Constants ---
class KptIdx: # COCO Keypoint Indices
    NOSE = 0; L_EYE = 1; R_EYE = 2; L_EAR = 3; R_EAR = 4
    L_SHOULDER = 5; R_SHOULDER = 6; L_ELBOW = 7; R_ELBOW = 8; L_WRIST = 9
    R_WRIST = 10; L_HIP = 11; R_HIP = 12; L_KNEE = 13; R_KNEE = 14
    L_ANKLE = 15; R_ANKLE = 16
    COUNT = 17 # Total number of keypoints

class MPKptIdx: # MediaPipe Pose Landmark Indices
    NOSE = 0; LEFT_EYE_INNER = 1; LEFT_EYE = 2; LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4; RIGHT_EYE = 5; RIGHT_EYE_OUTER = 6; LEFT_EAR = 7
    RIGHT_EAR = 8; MOUTH_LEFT = 9; MOUTH_RIGHT = 10; LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12; LEFT_ELBOW = 13; RIGHT_ELBOW = 14; LEFT_WRIST = 15
    RIGHT_WRIST = 16; LEFT_PINKY = 17; RIGHT_PINKY = 18; LEFT_INDEX = 19
    RIGHT_INDEX = 20; LEFT_THUMB = 21; RIGHT_THUMB = 22; LEFT_HIP = 23
    RIGHT_HIP = 24; LEFT_KNEE = 25; RIGHT_KNEE = 26; LEFT_ANKLE = 27
    RIGHT_ANKLE = 28; LEFT_HEEL = 29; RIGHT_HEEL = 30; LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32
    COUNT = 33

MEDIAPIPE_TO_COCO_MAP = { # Map MP indices to COCO indices
    KptIdx.NOSE: MPKptIdx.NOSE, KptIdx.L_EYE: MPKptIdx.LEFT_EYE, KptIdx.R_EYE: MPKptIdx.RIGHT_EYE,
    KptIdx.L_EAR: MPKptIdx.LEFT_EAR, KptIdx.R_EAR: MPKptIdx.RIGHT_EAR,
    KptIdx.L_SHOULDER: MPKptIdx.LEFT_SHOULDER, KptIdx.R_SHOULDER: MPKptIdx.RIGHT_SHOULDER,
    KptIdx.L_ELBOW: MPKptIdx.LEFT_ELBOW, KptIdx.R_ELBOW: MPKptIdx.RIGHT_ELBOW,
    KptIdx.L_WRIST: MPKptIdx.LEFT_WRIST, KptIdx.R_WRIST: MPKptIdx.RIGHT_WRIST,
    KptIdx.L_HIP: MPKptIdx.LEFT_HIP, KptIdx.R_HIP: MPKptIdx.RIGHT_HIP,
    KptIdx.L_KNEE: MPKptIdx.LEFT_KNEE, KptIdx.R_KNEE: MPKptIdx.RIGHT_KNEE,
    KptIdx.L_ANKLE: MPKptIdx.LEFT_ANKLE, KptIdx.R_ANKLE: MPKptIdx.RIGHT_ANKLE,
}

# --- Default Fall Detection Parameters (can be overridden during initialization) ---
DEFAULT_POSE_HISTORY_FRAMES = 10
DEFAULT_VELOCITY_CALC_FRAME_DIFF = 4
DEFAULT_MIN_KPT_CONFIDENCE = 0.3
DEFAULT_FALL_VELOCITY_THRESHOLD = 0.4  # NEW - MUCH LOWER
DEFAULT_FALL_CONFIDENCE_THRESHOLD = 0.50 # Confidence needed to trigger 'is_fall'
DEFAULT_FALL_VELOCITY_CONFIDENCE_BOOST = 0.4 # Added confidence if velocity threshold met


# ==============================================================================
# Core Pose Estimation and Fall Detection Logic Component
# ==============================================================================
class EnhancedPoseEstimation:
    """
    Core component for pose estimation (YOLOv8 or MediaPipe) and RAW fall detection.
    Does NOT handle alerting, recording, or temporal smoothing for alerting.
    """
    def __init__(self,
                 model_type='yolov8',
                 device='auto',
                 # --- Configurable Parameters ---
                 pose_history_frames=DEFAULT_POSE_HISTORY_FRAMES,
                 velocity_calc_frame_diff=DEFAULT_VELOCITY_CALC_FRAME_DIFF,
                 min_kpt_confidence=DEFAULT_MIN_KPT_CONFIDENCE,
                 fall_velocity_threshold=DEFAULT_FALL_VELOCITY_THRESHOLD,
                 fall_confidence_threshold=DEFAULT_FALL_CONFIDENCE_THRESHOLD,
                 fall_velocity_boost=DEFAULT_FALL_VELOCITY_CONFIDENCE_BOOST
                ):
        """Initialize pose estimation model and parameters."""
        self.model_type = model_type
        self.device = device

        # --- Store Configurable Parameters ---
        self.pose_history_frames = pose_history_frames
        self.velocity_calc_frame_diff = velocity_calc_frame_diff
        self.min_kpt_confidence = min_kpt_confidence
        self.fall_velocity_threshold = fall_velocity_threshold
        self.fall_confidence_threshold = fall_confidence_threshold
        self.fall_velocity_boost = fall_velocity_boost
        # --- End Stored Parameters ---

        print(f"Initializing Core Pose Estimator: Type={model_type}, Device={self.device}")
        logging.info(f"Core Pose Estimator: Type={model_type}, Device={self.device}, KptConf={self.min_kpt_confidence:.2f}, FallVelThresh={self.fall_velocity_threshold:.2f}")

        self.model = None
        self.mp_pose = None # For mediapipe

        if model_type == 'yolov8':
            try:
                self.model = YOLO('yolov8n-pose.pt')
                if self.device != 'auto': self.model.to(self.device)
                print(f"YOLOv8-pose model loaded on {self.model.device}")
                logging.info(f"YOLOv8-pose model loaded on {self.model.device}")
            except Exception as e:
                print(f"Error loading YOLOv8-pose model: {e}"); logging.error(f"Error loading YOLOv8-pose model: {e}"); raise
        elif model_type == 'mediapipe':
            if not MEDIAPIPE_AVAILABLE: raise ImportError("MediaPipe selected but not installed.")
            print("Warning: Using MediaPipe. Fall detection logic remains 2D-based."); logging.warning("Using MediaPipe. Fall detection logic remains 2D-based.")
            try:
                self.mp_pose = mp.solutions.pose
                self.model = self.mp_pose.Pose(static_image_mode=False, model_complexity=1,
                                               enable_segmentation=False, min_detection_confidence=0.5,
                                               min_tracking_confidence=0.5)
                print(f"MediaPipe pose model loaded"); logging.info(f"MediaPipe pose model loaded")
            except Exception as e:
                print(f"Error loading MediaPipe pose model: {e}"); logging.error(f"Error loading MediaPipe pose model: {e}"); raise
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Pose history is maintained per instance
        self.pose_history = defaultdict(lambda: deque(maxlen=self.pose_history_frames))
        self.global_pose_history = deque(maxlen=self.pose_history_frames)
        self.last_non_fall_print_time = defaultdict(float) # For debug prints

    def process_frame(self, frame, timestamp, fps):
        """
        Process a single frame using MediaPipe.
        Returns structured results including RAW fall detection.
        """
        result = {'poses': [], 'fall_detected_frame': False} # Overall frame status
        if self.model_type != 'mediapipe' or not self.model:
            logging.warning("process_frame called on non-mediapipe instance or uninitialized model.")
            return result

        h, w, _ = frame.shape
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            pose_results = self.model.process(frame_rgb)
            # frame_rgb.flags.writeable = True # Not needed if not drawing here

            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                coco_keypoints = np.zeros((KptIdx.COUNT, 3), dtype=np.float32)
                total_visibility = 0; valid_kpt_count = 0
                projected_points_for_bbox = []

                for coco_idx, mp_idx in MEDIAPIPE_TO_COCO_MAP.items():
                    if 0 <= mp_idx < MPKptIdx.COUNT:
                        lm = landmarks[mp_idx]
                        px, py, conf = lm.x * w, lm.y * h, lm.visibility
                        coco_keypoints[coco_idx] = [px, py, conf]
                        if conf >= self.min_kpt_confidence:
                            projected_points_for_bbox.append((px, py))
                            total_visibility += conf
                            valid_kpt_count += 1

                box, box_height = None, frame.shape[0] * 0.8  # Use frame height from frame.shape
                if len(projected_points_for_bbox) >= 5:
                    valid_kpts_np = np.array(projected_points_for_bbox)
                    x_min, y_min = valid_kpts_np[:, 0].min(), valid_kpts_np[:, 1].min()
                    x_max, y_max = valid_kpts_np[:, 0].max(), valid_kpts_np[:, 1].max()
                    box = [x_min, y_min, x_max, y_max]; box_height = max(y_max - y_min, 1.0)

                pose_confidence = (total_visibility / valid_kpt_count) if valid_kpt_count > 0 else 0.0
                track_id = -1 # MediaPipe doesn't provide track IDs

                self.global_pose_history.append((timestamp, coco_keypoints.copy(), box_height))

                # Get RAW fall detection result for this pose
                is_fall_raw, fall_prob_raw = self._detect_fall_with_pose(
                    coco_keypoints, frame.shape, timestamp, fps, track_id, box_height
                )

                pose_data = {
                    'keypoints': coco_keypoints, 'box': box, 'confidence': pose_confidence,
                    'id': track_id, 'is_fall': is_fall_raw, 'fall_confidence': fall_prob_raw,
                    'model_origin': 'mediapipe'
                }
                result['poses'].append(pose_data)
                if is_fall_raw: result['fall_detected_frame'] = True

            return result
        except Exception as e:
            logging.error(f"Error in MediaPipe process_frame: {str(e)}", exc_info=True)
            return result

    def analyze_poses(self, pose_results, timestamp, fps, use_tracking=False, tracking_results=None):
        """
        Process YOLOv8 pose results.
        Returns structured results including RAW fall detection per pose.
        """
        result = {'poses': [], 'fall_detected_frame': False} # Overall frame status
        if self.model_type != 'yolov8':
            logging.warning("analyze_poses called on non-yolov8 instance.")
            return result

        try:
            # Validate input structure
            if not pose_results or not pose_results[0].keypoints or pose_results[0].keypoints.data is None:
                 logging.debug("analyze_poses received no valid keypoints data.")
                 return result
            keypoints_data = pose_results[0].keypoints.data.cpu().numpy()
            if keypoints_data.shape[0] == 0:
                logging.debug("analyze_poses: keypoints_data tensor was empty.")
                return result

            orig_shape = pose_results[0].orig_shape
            boxes_data, confs_data = (pose_results[0].boxes.xyxy.cpu().numpy(), pose_results[0].boxes.conf.cpu().numpy()) if pose_results[0].boxes else (None, None)
            track_ids_data, track_boxes_data = (None, None)

            # Handle Tracking Data (if enabled and available)
            is_tracking_valid = False
            if use_tracking and tracking_results and tracking_results[0].boxes and tracking_results[0].boxes.id is not None:
                track_ids_data = tracking_results[0].boxes.id.int().cpu().numpy()
                track_boxes_data = tracking_results[0].boxes.xyxy.cpu().numpy()
                if len(keypoints_data) == len(track_ids_data):
                    is_tracking_valid = True
                else:
                    logging.warning(f"Tracking mismatch: Poses={len(keypoints_data)}, Tracks={len(track_ids_data)}. Disabling tracking for frame.")

            num_persons = keypoints_data.shape[0]
            for i in range(num_persons):
                if i >= len(keypoints_data): continue # Safety bound check
                kpts = keypoints_data[i]
                if not isinstance(kpts, np.ndarray) or kpts.shape != (KptIdx.COUNT, 3):
                    logging.warning(f"analyze_poses: Skipping invalid kpts shape {kpts.shape if isinstance(kpts, np.ndarray) else type(kpts)} for person {i}.")
                    continue

                # Determine Box, Confidence, and Track ID
                box, track_id = None, None
                pose_confidence = np.mean(kpts[:, 2][kpts[:, 2] > 0.1]) if np.any(kpts[:, 2] > 0.1) else 0.0

                if is_tracking_valid and i < len(track_ids_data):
                    track_id = int(track_ids_data[i]) # Ensure track ID is int
                    if track_boxes_data is not None and i < len(track_boxes_data): box = track_boxes_data[i]
                elif boxes_data is not None and i < len(boxes_data):
                    box = boxes_data[i]
                    if confs_data is not None and i < len(confs_data): pose_confidence = float(confs_data[i]) # Ensure float
                elif len(kpts[kpts[:, 2] > self.min_kpt_confidence]) >= 5: # Fallback box from keypoints
                    valid_kpts = kpts[kpts[:, 2] > self.min_kpt_confidence]
                    box = np.array([valid_kpts[:, 0].min(), valid_kpts[:, 1].min(), valid_kpts[:, 0].max(), valid_kpts[:, 1].max()])

                # Calculate Box Height
                box_height = orig_shape[0] * 0.8 # Default - use orig_shape instead of undefined h
                if box is not None:
                    box_height = max(box[3] - box[1], 1.0)
                else: # Estimate from keypoints if no box
                    valid_kpts_y = kpts[kpts[:, 2] > self.min_kpt_confidence][:, 1]
                    if len(valid_kpts_y) > 1: box_height = max(valid_kpts_y.max() - valid_kpts_y.min(), 1.0)

                # Update Pose History
                current_pose_info = (timestamp, kpts.copy(), box_height) # Store copy
                actual_track_id_for_history = -1 # Use -1 for global history
                if is_tracking_valid and track_id is not None:
                    self.pose_history[track_id].append(current_pose_info)
                    actual_track_id_for_history = track_id # Use actual ID for fall check
                else:
                    self.global_pose_history.append(current_pose_info)

                # --- Get RAW Fall Detection Result ---
                is_fall_raw, fall_prob_raw = self._detect_fall_with_pose(
                    kpts, orig_shape, timestamp, fps, actual_track_id_for_history, box_height
                )

                pose_data = {
                    'box': box, 'keypoints': kpts, 'id': track_id, # ID is None if not tracking
                    'confidence': pose_confidence, 'is_fall': is_fall_raw,
                    'fall_confidence': fall_prob_raw, 'model_origin': 'yolov8'
                }
                result['poses'].append(pose_data)
                if is_fall_raw: result['fall_detected_frame'] = True # Update overall frame status

            return result
        except Exception as e:
            logging.error(f"Error in YOLOv8 analyze_poses: {str(e)}", exc_info=True)
            return result

    # --- Internal Helper: Fall Detection Logic ---

    # Add a helper to get history robustly
    def _get_pose_history(self, track_id):
        return self.pose_history[track_id] if track_id != -1 and track_id in self.pose_history else self.global_pose_history

    def _get_hip_midpoint_y(self, keypoints):
        """Calculates the Y coordinate of the hip midpoint using COCO indices."""
        l_hip_y, r_hip_y = None, None
        # Check keypoint indices validity FIRST before accessing
        if KptIdx.L_HIP < keypoints.shape[0] and keypoints[KptIdx.L_HIP, 2] >= self.min_kpt_confidence:
           l_hip_y = keypoints[KptIdx.L_HIP, 1]
        if KptIdx.R_HIP < keypoints.shape[0] and keypoints[KptIdx.R_HIP, 2] >= self.min_kpt_confidence:
            r_hip_y = keypoints[KptIdx.R_HIP, 1]

        if l_hip_y is not None and r_hip_y is not None: return (l_hip_y + r_hip_y) / 2.0
        elif l_hip_y is not None: return l_hip_y
        elif r_hip_y is not None: return r_hip_y
        else: return None
    
    # Add helper to check recent velocity from history
    def _was_recently_fast(self, track_id, timestamp, lookback_sec=1.5):
        """ Checks if high velocity occurred within the recent lookback window. """
        history = self._get_pose_history(track_id)
        if len(history) < 2: return False # Need at least one past frame

        # Iterate backwards through recent history
        for i in range(len(history) - 1, -1, -1):
             past_ts, past_kpts, past_box_h = history[i]
             if (timestamp - past_ts) > lookback_sec: break # Stop if too old

             # Rough velocity estimate based on hip movement between history[i] and history[i-1]
             # This avoids recalculating full normalized velocity for every past frame
             if i > 0:
                 prev_ts, prev_kpts, _ = history[i-1]
                 curr_hip_y = self._get_hip_midpoint_y(past_kpts)
                 prev_hip_y = self._get_hip_midpoint_y(prev_kpts)
                 delta_t = past_ts - prev_ts

                 if curr_hip_y is not None and prev_hip_y is not None and delta_t > 1e-6:
                      # Crude check: large pixel drop relative to box height in a short time?
                      # Use the *current* box_height as an approximation for normalization
                      current_height_estimate = past_box_h # Use height from that frame
                      if not current_height_estimate or current_height_estimate < 10:
                          current_height_estimate = 500 # Fallback if height is bad

                      # Simplified normalized velocity check using the lower threshold
                      rough_norm_vel = abs(curr_hip_y - prev_hip_y) / delta_t / current_height_estimate
                      if rough_norm_vel > self.fall_velocity_threshold: # Use the class threshold
                          return True # Found high velocity recently
        return False

    def _detect_fall_with_pose(self, keypoints, frame_shape, timestamp, fps, track_id, box_height):
        """
        Detects falls based on high velocity OR low relative height persisting
        after a recent high velocity event. (V8 Strategy)
        """
        try:
            valid_keypoints = keypoints[keypoints[:, 2] >= self.min_kpt_confidence]
            if len(valid_keypoints) < 5: return False, 0.0

            # --- Calculate Core Indicators ---
            high_downward_velocity = False
            low_body_ratio = False
            # is_horizontal = False # Less reliable for state, only use for small boost?
            # is_wide_box = False

            body_height_ratio = 1.0
            orientation_ratio = 0.0 # Still calculate for debug/potential use
            box_aspect_ratio = 0.0

            # Velocity (using the NEW LOWER threshold)
            normalized_velocity = self._calculate_normalized_velocity(keypoints, frame_shape, timestamp, fps, track_id, box_height)
            high_downward_velocity = normalized_velocity > self.fall_velocity_threshold # Uses new threshold (e.g., 0.4)

            # Body Height Ratio (Relative to frame)
            current_height_estimate = box_height
            if not current_height_estimate or current_height_estimate < 10:
                valid_kpts_y = valid_keypoints[:, 1]
                if len(valid_kpts_y) > 1: current_height_estimate = max(valid_kpts_y.max() - valid_kpts_y.min(), 10.0)
                else: current_height_estimate = None

            if current_height_estimate and frame_shape[0] > 0:
                 body_height_ratio = current_height_estimate / frame_shape[0]
                 low_body_ratio = body_height_ratio < 0.40 # Keep threshold around 0.35-0.40? Needs checking

            # Horizontal Torso Orientation (for debug/minor boost)
            is_horizontal = False
            shoulders_mid, hips_mid_y, hips_mid_x = None, None, None
            s_l_y, s_l_x, s_l_c = keypoints[KptIdx.L_SHOULDER]
            s_r_y, s_r_x, s_r_c = keypoints[KptIdx.R_SHOULDER]
            if s_l_c >= self.min_kpt_confidence and s_r_c >= self.min_kpt_confidence:
                shoulders_mid = [(s_l_x + s_r_x)/2, (s_l_y + s_r_y)/2]

            hips_mid_y = self._get_hip_midpoint_y(keypoints)
            h_l_x, h_l_c = keypoints[KptIdx.L_HIP, 0], keypoints[KptIdx.L_HIP, 2]
            h_r_x, h_r_c = keypoints[KptIdx.R_HIP, 0], keypoints[KptIdx.R_HIP, 2]
            if hips_mid_y is not None and h_l_c >= self.min_kpt_confidence and h_r_c >= self.min_kpt_confidence:
                hips_mid_x = (h_l_x + h_r_x) / 2

            if shoulders_mid and hips_mid_x is not None and hips_mid_y is not None:
                 dx = abs(shoulders_mid[0] - hips_mid_x); dy = abs(shoulders_mid[1] - hips_mid_y)
                 orientation_ratio = dx / (dy + 1e-6)
                 is_horizontal = orientation_ratio > 1.8 # Maybe slightly higher threshold?

            # Wide Box (for debug/minor boost)
            is_wide_box = False
            if current_height_estimate and current_height_estimate > 1e-6 and len(valid_keypoints)>1:
                box_width = valid_keypoints[:,0].max() - valid_keypoints[:,0].min()
                box_aspect_ratio = box_width/current_height_estimate
                is_wide_box = box_aspect_ratio > 1.6

            # --- Combine Indicators (V8 Logic) ---
            fall_confidence = 0.0
            recently_fast = self._was_recently_fast(track_id, timestamp) # Check history

            # Condition 1: High Velocity NOW (The fall event)
            if high_downward_velocity:
                 fall_confidence = 0.70 # High confidence on velocity trigger

            # Condition 2: Low Ratio + Recent Velocity (Persisting fallen state)
            # Only trigger this if velocity isn't high *right now* but was high recently
            elif low_body_ratio and recently_fast:
                 fall_confidence = 0.55 # Maintain fall state if low after recent velocity spike
                 # Add minor boosts for horizontal/wide in this state?
                 if is_horizontal: fall_confidence = min(1.0, fall_confidence + 0.1)
                 if is_wide_box: fall_confidence = min(1.0, fall_confidence + 0.1)

            # Condition 3: Low Ratio WITHOUT Recent Velocity (Potentially standing/crouching - low conf)
            elif low_body_ratio:
                 fall_confidence = 0.15 # Low base confidence if just low ratio

            # Maybe add small confidence if just horizontal/wide?
            elif is_horizontal: fall_confidence = max(fall_confidence, 0.10)
            elif is_wide_box: fall_confidence = max(fall_confidence, 0.10)


            # Final decision
            fall_detected = fall_confidence >= self.fall_confidence_threshold # Use class threshold (0.50)

            # --- Debug Print (V8) ---
            # Keep printing unconditionally for now
#            if fall_confidence > 0.1 or fall_detected or recently_fast: # Print more often
            print(f"DEBUG ID {track_id} @ {timestamp:.2f}: "
                  f"CONF={fall_confidence:.3f} (Thresh={self.fall_confidence_threshold:.2f}) | "
                  f"Fall={fall_detected} | RctFast={recently_fast} | " # Added Recently Fast flag
                  f"H={is_horizontal}({orientation_ratio:.2f}) | "
                  f"LowR={low_body_ratio}({body_height_ratio:.2f}) | "
                  f"Wide={is_wide_box}({box_aspect_ratio:.2f}) | "
                  f"Vel={normalized_velocity:.3f}(High={high_downward_velocity}, Thresh={self.fall_velocity_threshold:.2f})") # Show threshold used

            return fall_detected, fall_confidence

        except Exception as e:
            logging.error(f"Error in _detect_fall_with_pose V8 for track {track_id}: {str(e)}", exc_info=True)
            return False, 0.0

    def _calculate_normalized_velocity(self, keypoints, frame_shape, timestamp, fps, track_id, box_height):
        """Calculates normalized vertical velocity of the hip midpoint."""
        try:
            # Determine which history to use (tracked ID or global)
            history = self.pose_history[track_id] if track_id != -1 and track_id in self.pose_history else self.global_pose_history

            # Check if enough history exists for calculation
            if len(history) < self.velocity_calc_frame_diff + 1:
                return 0.0

            # Get current hip position
            current_hip_y = self._get_hip_midpoint_y(keypoints)
            if current_hip_y is None:
                return 0.0 # Cannot calculate if current hip is missing

            # Get past hip position and timestamp from history
            # The index is -1 (last element) minus the frame difference
            past_timestamp, past_keypoints, past_box_height = history[-1 - self.velocity_calc_frame_diff]
            past_hip_y = self._get_hip_midpoint_y(past_keypoints)
            if past_hip_y is None:
                return 0.0 # Cannot calculate if past hip is missing

            # Calculate delta Y and delta T
            delta_y = current_hip_y - past_hip_y
            delta_t = timestamp - past_timestamp

            # Avoid division by zero if timestamps are identical (unlikely but possible)
            if delta_t <= 1e-6:
                return 0.0

            # Calculate velocity in pixels per second
            vertical_velocity_pixels_per_sec = delta_y / delta_t

            # Determine effective height for normalization (prefer box_height, fallback to frame height fraction)
            effective_height = box_height if box_height and box_height > 10 else frame_shape[0] * 0.8
            if effective_height <= 1e-6:
                return 0.0 # Avoid division by zero

            # Normalize velocity by effective height
            normalized_velocity = vertical_velocity_pixels_per_sec / effective_height

            # We are interested in downward velocity (positive delta_y means lower on screen)
            # Return 0 if moving up, otherwise return the downward velocity value
            return max(0.0, normalized_velocity)

        except IndexError:
            # Handle cases where history might be shorter than expected after checks (race condition?)
            logging.warning(f"IndexError calculating velocity for track {track_id}, history len: {len(history)}")
            return 0.0
        except Exception as e:
            logging.error(f"Error calculating velocity for track {track_id}: {e}", exc_info=True)
            return 0.0

    # --- Drawing Utility ---
    def draw_skeleton(self, frame, pose_data, draw_smoothed_state=False):
        """
        Draw skeleton and bounding box using COCO format keypoints.
        Args:
            frame: The image frame to draw on.
            pose_data: A dictionary containing 'keypoints', 'box', 'id', 'fall_confidence'.
                       It can optionally contain 'is_fall_smoothed' if draw_smoothed_state is True.
            draw_smoothed_state: If True, use 'is_fall_smoothed' from pose_data for box color/label.
                                 Otherwise, use the raw 'is_fall' status (not usually passed directly,
                                 but calculated from fall_confidence).
        """
        keypoints = pose_data.get('keypoints')
        if keypoints is None or not isinstance(keypoints, np.ndarray) or keypoints.shape != (KptIdx.COUNT, 3):
            logging.warning("draw_skeleton received invalid keypoints.")
            return # Cannot draw without valid keypoints

        box = pose_data.get('box', None)
        raw_fall_conf = pose_data.get('fall_confidence', 0.0)
        track_id = pose_data.get('id', None) # Might be None

        # Determine the state to display based on draw_smoothed_state flag
        is_displayed_fall = False
        if draw_smoothed_state:
            is_displayed_fall = pose_data.get('is_fall_smoothed', False) # Use pre-calculated smoothed state
        else:
            # Calculate raw fall state based on confidence threshold if not drawing smoothed
            is_displayed_fall = raw_fall_conf >= self.fall_confidence_threshold

        # --- COCO Connections ---
        connections = [
             (KptIdx.NOSE, KptIdx.L_EYE), (KptIdx.L_EYE, KptIdx.L_EAR), (KptIdx.NOSE, KptIdx.R_EYE),
             (KptIdx.R_EYE, KptIdx.R_EAR), (KptIdx.L_SHOULDER, KptIdx.R_SHOULDER),
             (KptIdx.L_SHOULDER, KptIdx.L_ELBOW), (KptIdx.L_ELBOW, KptIdx.L_WRIST),
             (KptIdx.R_SHOULDER, KptIdx.R_ELBOW), (KptIdx.R_ELBOW, KptIdx.R_WRIST),
             (KptIdx.L_HIP, KptIdx.R_HIP), (KptIdx.L_SHOULDER, KptIdx.L_HIP),
             (KptIdx.R_SHOULDER, KptIdx.R_HIP), (KptIdx.L_HIP, KptIdx.L_KNEE),
             (KptIdx.L_KNEE, KptIdx.L_ANKLE), (KptIdx.R_HIP, KptIdx.R_KNEE), (KptIdx.R_KNEE, KptIdx.R_ANKLE)
        ]
        keypoint_color = (0, 255, 0); line_color = (0, 255, 255)

        # Draw keypoints
        for i, (x, y, conf) in enumerate(keypoints):
            if conf >= self.min_kpt_confidence: cv2.circle(frame, (int(x), int(y)), 4, keypoint_color, -1)

        # Draw connections
        for i, j in connections:
             if keypoints[i, 2] >= self.min_kpt_confidence and keypoints[j, 2] >= self.min_kpt_confidence:
                 cv2.line(frame, (int(keypoints[i, 0]), int(keypoints[i, 1])),
                          (int(keypoints[j, 0]), int(keypoints[j, 1])), line_color, 2)

        # Draw bounding box and labels
        if box is not None:
            x1, y1, x2, y2 = map(int, box)
            # Box color depends on the displayed fall state
            box_color = (0, 0, 255) if is_displayed_fall else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            label_items = []
            if track_id is not None: label_items.append(f"ID:{track_id}")
            # Label also depends on the displayed fall state
            if is_displayed_fall: label_items.append(f"FALL({raw_fall_conf:.2f})")
            label = " ".join(label_items)
            if label: cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)


# ==============================================================================
# Standalone Application Runner Component (Uses EnhancedPoseEstimation)
# ==============================================================================
class ElderlyMonitoringSystem:
    """
    Standalone application runner using EnhancedPoseEstimation.
    Handles video loop, temporal smoothing for alerts, alerting (Telegram),
    recording, and displaying output.
    """
    def __init__(self, video_path,
                 # Pose estimation config
                 pose_model_type='yolov8', use_tracking=True, device='auto',
                 # Fall detection config (passed to EnhancedPoseEstimation)
                 min_kpt_confidence=DEFAULT_MIN_KPT_CONFIDENCE,
                 fall_velocity_threshold=DEFAULT_FALL_VELOCITY_THRESHOLD,
                 fall_confidence_threshold=DEFAULT_FALL_CONFIDENCE_THRESHOLD,
                 fall_velocity_boost=DEFAULT_FALL_VELOCITY_CONFIDENCE_BOOST,
                 # Standalone app behavior config
                 output_dir='processed_videos', inactivity_threshold_sec=300,
                 fall_cooldown_sec=30, emergency_cooldown_sec=60,
                 emergency_fall_duration_sec=15,
                 fall_confirm_frames=3, fall_clear_frames=5): # Smoothing frames for alerting

        self.video_source = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened(): raise IOError(f"Error: Could not open video source: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not self.fps or self.fps <= 0: self.fps = 30.0; logging.warning("Defaulting FPS to 30.0.")
        logging.info(f"Standalone App: Video FPS: {self.fps:.2f}")

        # --- Initialize Core Pose Estimator ---
        self.pose_estimator = EnhancedPoseEstimation(
            model_type=pose_model_type, device=device,
            min_kpt_confidence=min_kpt_confidence,
            fall_velocity_threshold=fall_velocity_threshold,
            fall_confidence_threshold=fall_confidence_threshold,
            fall_velocity_boost=fall_velocity_boost
            # Using default history/velocity diff frames here, could also be args
        )
        self.pose_model_type = pose_model_type # Store for logic
        self.use_tracking = use_tracking and (pose_model_type == 'yolov8')
        # --- End Core Estimator Init ---

        # --- Standalone App State & Config ---
        self.output_dir = output_dir; os.makedirs(self.output_dir, exist_ok=True)
        self.inactivity_threshold = inactivity_threshold_sec
        self.last_activity_time = time.monotonic()
        self.frame_count = 0
        self.debug_mode = False
        self.loop_start_time = 0.0
        self.fps_tracker = deque(maxlen=max(10, int(self.fps)))

        # Alerting / Recording State (Specific to this standalone runner)
        self.bot = _standalone_bot # Use the bot initialized globally for standalone
        self.chat_id = _STANDALONE_CHAT_ID
        self.alert_sent_time = defaultdict(float)
        self.emergency_alert_sent_time = defaultdict(float)
        self.fall_cooldown = fall_cooldown_sec
        self.emergency_cooldown = emergency_cooldown_sec
        self.current_fall_active = defaultdict(bool) # SMOOTHED state for alerting
        self.fall_start_time = defaultdict(lambda: None)
        self.emergency_threshold = emergency_fall_duration_sec
        self.fall_confirm_counters = defaultdict(int) # For smoothing
        self.fall_clear_counters = defaultdict(int)  # For smoothing
        self.FALL_CONFIRM_FRAMES = fall_confirm_frames # Smoothing param
        self.FALL_CLEAR_FRAMES = fall_clear_frames   # Smoothing param
        self.fall_recording_buffers = defaultdict(lambda: deque(maxlen=int(self.fps * 15))) # 15 sec buffer
        self.recording_active = defaultdict(bool)
        self.recorded_video_path = defaultdict(lambda: None)
        # --- End Standalone State ---

        print(f"Standalone Monitor: Tracking: {'ON' if self.use_tracking else 'OFF'}")
        logging.info(f"Standalone Monitor: Tracking: {'ON' if self.use_tracking else 'OFF'}")

    # --- Helper methods (log, timestamp, alert, recording) ---
    def log_event(self, message, level=logging.INFO): print(message); logging.log(level, message)
    def get_current_timestamp_monotonic(self): return time.monotonic()
    def get_video_timestamp(self): return self.frame_count / self.fps if self.fps > 0 else 0.0

    def send_alert(self, frame_to_send, message, alert_id=-1, is_emergency=False):
        # (Keep existing send_alert logic using self.bot, self.chat_id etc.)
        if not self.bot or not self.chat_id:
            self.log_event("Standalone: Bot/ChatID not configured, cannot send alert.", level=logging.WARNING)
            return
        now = self.get_current_timestamp_monotonic()
        cooldown = self.emergency_cooldown if is_emergency else self.fall_cooldown
        last_alert_times = self.emergency_alert_sent_time if is_emergency else self.alert_sent_time
        if (now - last_alert_times.get(alert_id, 0.0)) < cooldown: return

        last_alert_times[alert_id] = now
        log_level = logging.WARNING if is_emergency else logging.INFO
        self.log_event(f"Standalone: Sending {'Emergency' if is_emergency else 'Fall'} Alert (ID: {alert_id}): {message.splitlines()[0]}", level=log_level)

        # Use threading to avoid blocking the main loop
        def send_alert_thread():
            try:
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                fn = f"alert_{'emg' if is_emergency else 'fall'}_id{alert_id}_{ts}.jpg"
                # Save image in output dir
                img_path = os.path.join(self.output_dir, fn)
                if not cv2.imwrite(img_path, frame_to_send):
                    self.log_event(f"Standalone: Failed image write: {img_path}", logging.ERROR); return

                # Send message and photo
                self.bot.sendMessage(self.chat_id, message)
                with open(img_path, 'rb') as photo: self.bot.sendPhoto(self.chat_id, photo=photo)

                # Send associated video if recorded
                vid_path = self.recorded_video_path.get(alert_id, None)
                if vid_path and os.path.exists(vid_path):
                    self.log_event(f"Standalone: Sending video for ID {alert_id}: {vid_path}", logging.INFO)
                    self.bot.sendMessage(self.chat_id, "Associated recording:")
                    try:
                        with open(vid_path, 'rb') as vf: self.bot.sendVideo(self.chat_id, video=vf, timeout=120)
                    except telepot.exception.TelegramError as te: self.log_event(f"Standalone: TG Error sending video: {te}", logging.ERROR)
                    except Exception as ve: self.log_event(f"Standalone: Error sending video: {ve}", logging.ERROR)

                # Clean up temporary image file
                if os.path.exists(img_path): os.remove(img_path)

            except telepot.exception.TelegramError as te: self.log_event(f"Standalone: TG API Error: {te}", logging.ERROR)
            except Exception as e: self.log_event(f"Standalone: Failed alert thread: {e}", logging.ERROR, exc_info=True)

        thread = threading.Thread(target=send_alert_thread, daemon=True); thread.start()

    def save_fall_recording(self, alert_id=-1):
        # (Keep existing save_fall_recording logic, saving to self.output_dir)
        buffer = self.fall_recording_buffers.get(alert_id, None)
        min_frames = int(self.fps * 2) # Require at least 2 seconds
        if not buffer or len(buffer) < min_frames:
             self.log_event(f"Standalone: Rec Buffer too short ID {alert_id} ({len(buffer) if buffer else 0} frames)", logging.DEBUG)
             return None
        dt = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        vp = os.path.join(self.output_dir, f"fall_rec_id{alert_id}_{dt}.mp4")
        out = None
        try:
            first_valid_frame = next((f for f in buffer if f is not None), None)
            if first_valid_frame is None: raise ValueError("No valid frames in buffer.")
            h, w = first_valid_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Use a common codec
            out = cv2.VideoWriter(vp, fourcc, self.fps, (w, h))
            if not out.isOpened(): raise IOError(f"Failed VideoWriter: {vp}")

            fc = 0
            for frame in list(buffer): # Iterate over a copy
                if frame is not None and frame.shape[:2] == (h, w):
                    out.write(frame); fc += 1
                else: self.log_event(f"Standalone: Skipping invalid frame in rec buffer ID {alert_id}.", logging.WARNING)
            out.release(); out = None # Ensure released before check

            if fc >= min_frames:
                self.log_event(f"Standalone: Recording ID {alert_id} saved: {vp} ({fc} frames)", logging.INFO)
                self.recorded_video_path[alert_id] = vp; return vp
            else:
                self.log_event(f"Standalone: Insufficient valid frames ({fc}) for rec ID {alert_id}. Not saved.", logging.WARNING)
                if os.path.exists(vp): os.remove(vp); return None
        except Exception as e:
            self.log_event(f"Standalone: Failed save recording ID {alert_id}: {e}", logging.ERROR, exc_info=True)
            if out is not None and out.isOpened(): out.release() # Ensure release on error
            if 'vp' in locals() and os.path.exists(vp):
                 try: os.remove(vp)
                 except OSError: pass
            return None

    # --- Main Processing Loop for Standalone Execution ---
    def process_video(self):
        self.log_event("Starting Standalone Interactive Video Processing...")
        self.last_activity_time = self.get_current_timestamp_monotonic()
        self.loop_start_time = time.monotonic()
        paused = False

        while True:
            frame_start_time = self.get_current_timestamp_monotonic()

            if not paused:
                ret, frame = self.cap.read()
                if not ret: self.log_event("End of video or read error."); break
                self.frame_count += 1
                display_frame = frame.copy() # Work on a copy for drawing
                current_monotonic_time = frame_start_time # Use frame start time for consistency
                current_video_time_sec = self.get_video_timestamp()

                # --- Get Raw Pose & Fall Detections ---
                analyzed_poses = None
                if self.use_tracking:
                     try: # Separate track/pose calls for clarity
                         tracking_results = self.pose_estimator.model.track(display_frame, persist=True, verbose=False)
                         # Get pose results on the same frame
                         pose_results_for_analysis = self.pose_estimator.model(display_frame, verbose=False)
                         analyzed_poses = self.pose_estimator.analyze_poses(
                             pose_results_for_analysis, current_monotonic_time, self.fps,
                             use_tracking=True, tracking_results=tracking_results)
                     except Exception as e:
                         self.log_event(f"Standalone Frame {self.frame_count}: Error YOLOv8 track/pose: {e}", logging.ERROR)
                         analyzed_poses = {'poses': [], 'fall_detected_frame': False} # Default empty result
                else: # MediaPipe or No Tracking
                    if self.pose_model_type == 'mediapipe':
                         analyzed_poses = self.pose_estimator.process_frame(display_frame, current_monotonic_time, self.fps)
                    else: # YOLOv8 without tracking
                         pose_results_for_analysis = self.pose_estimator.model(display_frame, verbose=False)
                         analyzed_poses = self.pose_estimator.analyze_poses(pose_results_for_analysis, current_monotonic_time, self.fps, use_tracking=False)
                # --- End Raw Detection ---

                activity_detected = False
                processed_poses_for_drawing = [] # Store data needed for drawing

                # --- Apply Temporal Smoothing & Trigger Alerts/Recording ---
                if analyzed_poses and analyzed_poses['poses']:
                    activity_detected = True # Any pose detection counts as activity
                    for pose_data in analyzed_poses['poses']:
                        person_id_raw = pose_data.get('id', -1) # Use -1 if no ID
                        person_id = int(person_id_raw) if person_id_raw is not None else -1
                        is_fall_raw = pose_data['is_fall'] # Get RAW fall status

                        # --- Smoothing Logic for Alerting/State ---
                        if is_fall_raw:
                            self.fall_confirm_counters[person_id] += 1
                            self.fall_clear_counters[person_id] = 0
                        else:
                            self.fall_clear_counters[person_id] += 1
                            self.fall_confirm_counters[person_id] = 0

                        # Check for state change based on smoothing
                        if not self.current_fall_active[person_id] and self.fall_confirm_counters[person_id] >= self.FALL_CONFIRM_FRAMES:
                            self.current_fall_active[person_id] = True # Start SMOOTHED fall
                            self.fall_start_time[person_id] = current_monotonic_time
                            self.fall_clear_counters[person_id] = 0 # Reset clear counter on fall start
                            self.recording_active[person_id] = True # Start recording buffer
                            self.log_event(f"Standalone: SMOOTHED Fall START - ID: {person_id} at {current_video_time_sec:.2f}s", logging.INFO)
                            alert_msg = f"🚨 Fall Detected!\nPerson ID: {person_id}\nTime: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}"
                            self.send_alert(frame, alert_msg, alert_id=person_id, is_emergency=False) # Send initial alert

                        elif self.current_fall_active[person_id] and self.fall_clear_counters[person_id] >= self.FALL_CLEAR_FRAMES:
                            self.current_fall_active[person_id] = False # End SMOOTHED fall
                            self.fall_confirm_counters[person_id] = 0 # Reset confirm counter
                            fall_duration = current_monotonic_time - (self.fall_start_time[person_id] or current_monotonic_time)
                            self.log_event(f"Standalone: SMOOTHED Fall END - ID: {person_id} at {current_video_time_sec:.2f}s (Duration: {fall_duration:.1f}s)", logging.INFO)
                            self.fall_start_time[person_id] = None
                            # Stop recording AFTER a delay or when buffer is full? For now, just save.
                            self.save_fall_recording(alert_id=person_id)
                            self.recording_active[person_id] = False # Stop buffering (or let it overwrite?)

                        # --- Emergency Alert Check (only if fall is currently active) ---
                        if self.current_fall_active[person_id] and self.fall_start_time[person_id]:
                            fall_duration = current_monotonic_time - self.fall_start_time[person_id]
                            if fall_duration >= self.emergency_threshold:
                                emergency_msg = f"🆘 EMERGENCY! Fall Duration > {self.emergency_threshold}s\nPerson ID: {person_id}\nTime: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}"
                                self.send_alert(frame, emergency_msg, alert_id=person_id, is_emergency=True)
                                # Don't reset fall start time here, let it clear normally

                        # Add frame to recording buffer if recording is active for this ID
                        if self.recording_active.get(person_id, False):
                            self.fall_recording_buffers[person_id].append(frame.copy())

                        # Prepare data for drawing (using the SMOOTHED state)
                        pose_data['is_fall_smoothed'] = self.current_fall_active[person_id]
                        processed_poses_for_drawing.append(pose_data)
                # --- End Smoothing & Alerting Loop ---

                # --- Inactivity Check ---
                if activity_detected: self.last_activity_time = current_monotonic_time
                else:
                    inactivity_duration = current_monotonic_time - self.last_activity_time
                    if inactivity_duration > self.inactivity_threshold:
                        # Implement inactivity alert if needed (similar to fall alert)
                        # self.log_event(f"Inactivity detected: {inactivity_duration:.1f}s", logging.WARNING)
                        self.last_activity_time = current_monotonic_time # Reset after alert

                # --- Draw Skeletons & Info ---
                for p_data in processed_poses_for_drawing:
                    # Draw using the SMOOTHED state for correct visual feedback
                    self.pose_estimator.draw_skeleton(display_frame, p_data, draw_smoothed_state=True)

                # Add overlay text (FPS, Time, Status)
                frame_end_time = self.get_current_timestamp_monotonic()
                proc_time_ms = (frame_end_time - frame_start_time) * 1000
                self.fps_tracker.append(1.0 / (frame_end_time - frame_start_time + 1e-6))
                avg_fps = sum(self.fps_tracker) / len(self.fps_tracker)
                status_text = f"Time: {current_video_time_sec:.2f}s | FPS: {avg_fps:.1f} | Proc: {proc_time_ms:.0f}ms"
                status_text += f" | Track: {'ON' if self.use_tracking else 'OFF'}"
                status_text += f" | Pose: {self.pose_model_type.upper()}"
                cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if paused: cv2.putText(display_frame, "PAUSED", (display_frame.shape[1]//2 - 50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                if self.debug_mode: cv2.putText(display_frame, "DEBUG ON", (10, display_frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            # --- Display Frame ---
            if 'display_frame' in locals(): # Handle case where pause happens before first frame
                cv2.imshow('Standalone Elderly Monitoring System', display_frame)

            # --- Handle User Input ---
            key = cv2.waitKey(1 if not paused else 0) & 0xFF
            if key == ord('q'): self.log_event("'q' pressed. Exiting."); break
            elif key == ord('p'): paused = not paused; self.log_event("Paused" if paused else "Resumed")
            elif key == ord('d'): self.debug_mode = not self.debug_mode; self.log_event(f"Debug mode {'ON' if self.debug_mode else 'OFF'}")
            # Add other controls if needed

        self.log_event("Standalone processing finished.")
        self.cleanup()

    # --- API Processing Method (kept separate, potentially removed if not needed) ---
    def process_video_for_api(self, progress_callback=None):
        """ Processes video, saves annotated video, returns RAW detection data. """
        self.log_event(f"Starting API processing & annotation for {self.video_source}")
        detections = [] # Stores RAW fall events
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.cap else 0

        output_filename, output_filepath = None, None
        out_writer = None
        # --- Setup Output Video Writer ---
        try:
            original_basename = os.path.basename(self.video_source)
            name, ext = os.path.splitext(original_basename)
            output_filename = f"{name}_annotated_{uuid.uuid4().hex[:8]}{ext}"
            output_filepath = os.path.join(self.output_dir, output_filename)
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Or 'avc1' for H.264 if available
            out_writer = cv2.VideoWriter(output_filepath, fourcc, self.fps, (frame_width, frame_height))
            if not out_writer.isOpened(): raise IOError(f"Could not open video writer: {output_filepath}")
            self.log_event(f"API Mode: Output video writer initialized: {output_filepath}")
        except Exception as e:
             self.log_event(f"API Mode: Error initializing video writer: {e}", logging.ERROR); self.cleanup(); raise

        self.frame_count = 0
        last_progress_update = -1
        # Smoothing state specifically for drawing the *annotated video*
        api_draw_fall_state = defaultdict(bool)
        api_draw_fall_confirm = defaultdict(int)
        api_draw_fall_clear = defaultdict(int)

        while True:
            current_monotonic_time = time.monotonic()
            ret, frame = self.cap.read()
            if not ret: break
            self.frame_count += 1
            current_video_time_sec = self.get_video_timestamp()

            # --- Get Raw Pose & Fall Detections (same as standalone loop) ---
            analyzed_poses = None
            if self.use_tracking:
                 try:
                     tracking_results = self.pose_estimator.model.track(frame, persist=True, verbose=False)
                     pose_results_for_analysis = self.pose_estimator.model(frame, verbose=False)
                     analyzed_poses = self.pose_estimator.analyze_poses(
                         pose_results_for_analysis, current_monotonic_time, self.fps,
                         use_tracking=True, tracking_results=tracking_results)
                 except Exception as e: analyzed_poses = {'poses': [], 'fall_detected_frame': False}
            else:
                if self.pose_model_type == 'mediapipe': analyzed_poses = self.pose_estimator.process_frame(frame, current_monotonic_time, self.fps)
                else:
                     pose_results_for_analysis = self.pose_estimator.model(frame, verbose=False)
                     analyzed_poses = self.pose_estimator.analyze_poses(pose_results_for_analysis, current_monotonic_time, self.fps, use_tracking=False)
            # --- End Raw Detection ---

            processed_poses_for_drawing = []
            if analyzed_poses and analyzed_poses['poses']:
                 for pose_data in analyzed_poses['poses']:
                    person_id_raw = pose_data.get('id', -1)
                    person_id = int(person_id_raw) if person_id_raw is not None else -1
                    is_fall_raw = pose_data['is_fall']
                    fall_conf_raw = pose_data['fall_confidence']
                    box_raw = pose_data.get('box')

                    # --- Record RAW Detection Event ---
                    if is_fall_raw: # Using raw detection for the API output data
                        detection_event = {
                            "type": "fall", "frame": self.frame_count, "timestamp_sec": round(current_video_time_sec, 2),
                            "confidence": round(fall_conf_raw, 3), "person_id": person_id,
                            "bounding_box": [int(b) for b in box_raw] if box_raw is not None else None }
                        # Simple check to avoid flooding with identical events per frame
                        if not detections or abs(detections[-1]['timestamp_sec'] - current_video_time_sec) > 0.1 or detections[-1]['person_id'] != person_id:
                            detections.append(detection_event)
                            # logging.debug(f"API Frame {self.frame_count}: Raw Fall appended: ID {person_id}") # Debug

                    # --- Apply Smoothing FOR DRAWING the annotated video ---
                    # (This makes the saved video look consistent with the standalone app's view)
                    if is_fall_raw:
                        api_draw_fall_confirm[person_id] += 1
                        api_draw_fall_clear[person_id] = 0
                        if api_draw_fall_confirm[person_id] >= self.FALL_CONFIRM_FRAMES: # Use fixed smoothing values here or pass them in
                            api_draw_fall_state[person_id] = True
                    else:
                        api_draw_fall_clear[person_id] += 1
                        api_draw_fall_confirm[person_id] = 0
                        if api_draw_fall_clear[person_id] >= self.FALL_CLEAR_FRAMES:
                            api_draw_fall_state[person_id] = False

                    # Prepare data for drawing annotated video
                    pose_data['is_fall_smoothed'] = api_draw_fall_state[person_id]
                    processed_poses_for_drawing.append(pose_data)
                    # --- End Smoothing for Drawing ---

            # Draw annotations onto a copy for the output video
            annotated_frame = frame.copy()
            for p_data in processed_poses_for_drawing:
                self.pose_estimator.draw_skeleton(annotated_frame, p_data, draw_smoothed_state=True)
            # Add timestamp to annotated video
            cv2.putText(annotated_frame, f"T: {current_video_time_sec:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Write frame to output video
            if out_writer: out_writer.write(annotated_frame)

            # Update Progress
            if progress_callback and total_frames > 0:
                progress = int((self.frame_count / total_frames) * 100)
                if progress > last_progress_update:
                    progress_callback(progress); last_progress_update = progress

        # --- End Loop ---
        if progress_callback and last_progress_update < 100: progress_callback(100) # Ensure 100%
        if out_writer: out_writer.release(); self.log_event(f"API Mode: Finished writing annotated video: {output_filepath}")
        else: output_filename = None # No file saved if writer failed

        self.log_event(f"API Mode: processing finished. Found {len(detections)} raw fall events.")
        self.cleanup()
        return detections, output_filename # Return RAW detections and filename


    def cleanup(self):
        """Release resources."""
        if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
             self.cap.release()
             self.log_event("Video capture released.")
        # Only destroy windows if NOT in API mode (or check if window exists)
        if hasattr(self, 'process_video_for_api') and callable(getattr(self, 'process_video_for_api')):
             pass # Assume API mode doesn't create windows
        else:
             cv2.destroyAllWindows()


# --- Main function for Standalone Execution ---
def main():
    parser = argparse.ArgumentParser(description='Standalone Elderly Monitoring System Runner')
    parser.add_argument('--video', type=str, required=True, help='Path to video file or camera index')
    parser.add_argument('--pose-model', type=str, default='yolov8', choices=['yolov8', 'mediapipe'], help='Pose estimation model')
    parser.add_argument('--no-tracking', action='store_true', help='Disable YOLOv8 tracking')
    parser.add_argument('--device', type=str, default='auto', help="Device ('cpu', 'cuda', 'mps', 'auto')")
    parser.add_argument('--output-dir', type=str, default='processed_videos', help='Directory for output files')
    # Add args for configurable thresholds? Example:
    # parser.add_argument('--fall-vel-thresh', type=float, default=DEFAULT_FALL_VELOCITY_THRESHOLD)
    # parser.add_argument('--fall-conf-thresh', type=float, default=DEFAULT_FALL_CONFIDENCE_THRESHOLD)
    # parser.add_argument('--fall-confirm-frames', type=int, default=3) # Example smoothing frames
    # parser.add_argument('--fall-clear-frames', type=int, default=5)   # Example smoothing frames

    args = parser.parse_args()

    # --- Determine Video Source ---
    video_source = args.video; is_camera = False
    try: video_source = int(args.video); is_camera = True; print(f"Using camera: {video_source}")
    except ValueError: print(f"Using video file: {video_source}")
    if not is_camera and not os.path.exists(video_source): print(f"Error: File not found: {video_source}"); return

    # --- Determine Device ---
    selected_device = args.device
    if selected_device == 'auto': selected_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {selected_device}")

    monitor = None
    try:
        # --- Instantiate Standalone System ---
        monitor = ElderlyMonitoringSystem(
            video_source,
            pose_model_type=args.pose_model,
            use_tracking=(not args.no_tracking),
            device=selected_device,
            output_dir=args.output_dir
            # Pass other args if added:
            # fall_velocity_threshold=args.fall_vel_thresh,
            # fall_confidence_threshold=args.fall_conf_thresh,
            # fall_confirm_frames=args.fall_confirm_frames,
            # fall_clear_frames=args.fall_clear_frames
        )
        print("\nStarting Standalone System..."); print("-" * 40)
        print("Controls: 'q': Quit | 'p': Pause/Resume | 'd': Toggle Debug Log"); print("-" * 40)
        monitor.process_video() # Run the interactive loop

    except (ImportError, IOError, ValueError) as e: print(f"Initialization Error: {e}"); logging.critical(f"Init Error: {e}")
    except Exception as e: print(f"\n--- UNEXPECTED ERROR ---"); print(f"Type: {type(e).__name__}\nDetails: {e}"); logging.critical(f"Unhandled exception: {e}", exc_info=True); traceback.print_exc(); print("--- END ERROR ---\n")
    finally:
        if monitor: monitor.cleanup()
        else: cv2.destroyAllWindows(); print("Cleanup after init fail.")

if __name__ == "__main__":
    main() # Execute standalone runner

# --- END OF REFACTORED detection.py ---
