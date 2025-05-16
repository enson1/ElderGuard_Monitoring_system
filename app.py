import cv2
import time
import threading
import logging
import os
import datetime
import uuid
import numpy as np
import json
import base64 # Added for FR
import dlib # Added for FR
import re # Added for FR
from flask import Flask, render_template, Response, request, redirect, url_for, flash, send_from_directory, jsonify, current_app, session, stream_with_context # Added session, stream_with_context for FR
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash # Added for FR
from collections import defaultdict, deque
import torch
from flask_socketio import SocketIO # Keep for existing, FR does not use it directly
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import desc, func
from sqlalchemy.exc import IntegrityError # Added for FR
import socket

# --- Email Imports ---
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication # For attachments
from email.utils import formatdate


# --- Setup Logging FIRST ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Import the refactored classes ---
try:
    # Only import the core class needed by the app
    from detection import EnhancedPoseEstimation, KptIdx, MEDIAPIPE_AVAILABLE, DEFAULT_FALL_VELOCITY_THRESHOLD, DEFAULT_FALL_CONFIDENCE_THRESHOLD, DEFAULT_MIN_KPT_CONFIDENCE
    from detection import _standalone_bot as bot # Rename to avoid confusion
    from detection import _STANDALONE_CHAT_ID as CHAT_ID # Rename
    from clap_tracker import ClapTracker # <<< Import the new ClapTracker class
    # --- Import for ASL Feature --- # ADDED
    from sign_detector_logic import SignDetector
except ImportError as e:
    print(f"Error importing from detection.py or clap_tracker.py: {e}")
    logging.critical(f"Error importing modules: {e}", exc_info=True)
    exit()

# --- Configuration ---
CAMERA_INDEX = 0
POSE_MODEL_TYPE = 'yolov8' # Or 'mediapipe'
USE_TRACKING = True # Note: Tracking often disabled for file processing in background task
DEVICE = 'auto' # 'cpu', 'cuda', 'mps', etc.
FRAME_WIDTH = 640 # Output width for snippets/full video
FRAME_HEIGHT = 480 # Output height
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed' # For full videos AND snippets
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv'}
CONFIG_FILE = 'config.json'
ALERT_COOLDOWN_SECONDS = 10 # Cooldown between Telegram alerts for the *same* event
DATABASE_FILE = 'clap_history.db'
SNIPPET_BUFFER_SECONDS = 8 # Total seconds for snippet

# --- ASL Configuration (from sign_app.py) --- # ADDED
ASL_MODEL_PATH = 'custom_cnn_sign_language_best.keras'
ASL_TRAIN_PATH_FOR_LABELS = r"C:\\Users\\enson\\Downloads\\asl_alphabet_train" # <--- CHANGE IF NEEDED for ASL
ASL_IMG_HEIGHT = 64
ASL_IMG_WIDTH = 64
ASL_CONF_THRESHOLD = 0.85
ASL_STABILITY_THRESHOLD_TIME = 0.7
ASL_ROI_X, ASL_ROI_Y, ASL_ROI_W, ASL_ROI_H = 100, 100, 250, 250

# --- Face Recognition (FR) Configuration (from face_recong_app.py) --- # ADDED_FR
FR_DLIB_LANDMARK_MODEL_PATH = "shape_predictor_68_face_landmarks.dat"
FR_FACE_RECOGNITION_MODEL_PATH = "dlib_face_recognition_resnet_model_v1.dat"
FR_DNN_MODEL_FILE = "deploy.prototxt"
FR_DNN_WEIGHTS_FILE = "res10_300x300_ssd_iter_140000.caffemodel"
FR_DNN_CONFIDENCE_THRESHOLD = 0.6
FR_SIMILARITY_THRESHOLD = 0.5 # Similarity threshold for face recognition
FR_POSES = ["front", "left", "right", "chin_up", "chin_down", "mouth_open"]
FR_MAX_IMAGES_TO_CAPTURE = 5 # Number of images for each pose during registration
FR_CAPTURE_DELAY_SECONDS = 3 # Delay for auto-capture during registration
FR_MOUTH_OPEN_THRESHOLD = 15 # Threshold for detecting mouth open pose

# --- NEW: CameraService Class ---
class CameraService:
    def __init__(self, camera_index: int):
        self.camera_index: int = camera_index
        self._cap: cv2.VideoCapture | None = None
        self._active_user: str | None = None
        self._lock: threading.Lock = threading.Lock()
        self._current_fps: float = 15.0  # Default FPS
        self._actual_width: int = 0
        self._actual_height: int = 0

    def acquire(self, user_id: str, desired_width: int, desired_height: int) -> cv2.VideoCapture | None:
        with self._lock:
            if self._active_user is not None:
                if self._active_user == user_id:
                    # Already acquired by this user, ensure dimensions are compatible or re-init if necessary
                    # For simplicity, returning current cap. More complex logic could re-check/re-init.
                    logging.info(f"CameraService: '{user_id}' re-requesting already acquired camera.")
                    return self._cap
                logging.warning(f"CameraService: Acquisition failed for '{user_id}'. Camera in use by '{self._active_user}'.")
                return None

            logging.info(f"CameraService: Attempting to acquire camera for '{user_id}' with index {self.camera_index}.")
            try:
                # Ensure any previous instance is released internally before creating a new one.
                if self._cap:
                    self._cap.release()
                    self._cap = None

                cap_instance = cv2.VideoCapture(self.camera_index)
                time.sleep(0.5) # Some cameras need a moment
                if not cap_instance.isOpened():
                    logging.error(f"CameraService: Failed to open camera index {self.camera_index} for '{user_id}'.")
                    return None

                cap_instance.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
                cap_instance.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
                
                # Verify settings
                self._actual_width = int(cap_instance.get(cv2.CAP_PROP_FRAME_WIDTH))
                self._actual_height = int(cap_instance.get(cv2.CAP_PROP_FRAME_HEIGHT))
                logging.info(f"CameraService: Requested {desired_width}x{desired_height}, got {self._actual_width}x{self._actual_height}.")

                fps_read = cap_instance.get(cv2.CAP_PROP_FPS)
                if not fps_read or fps_read <= 0:
                    logging.warning(f"CameraService: Could not get valid FPS for '{user_id}'. Defaulting to 15.0.")
                    self._current_fps = 15.0
                else:
                    self._current_fps = fps_read
                    logging.info(f"CameraService: Camera FPS for '{user_id}': {self._current_fps:.2f}")
                
                self._cap = cap_instance
                self._active_user = user_id
                logging.info(f"CameraService: Camera acquired successfully by '{user_id}'.")
                return self._cap
            except Exception as e:
                logging.error(f"CameraService: Exception during camera acquisition for '{user_id}': {e}", exc_info=True)
                if self._cap: # Check self._cap not cap_instance which might be local
                    self._cap.release()
                self._cap = None
                self._active_user = None
                return None

    def release(self, user_id: str) -> bool:
        with self._lock:
            if self._active_user == user_id:
                logging.info(f"CameraService: Releasing camera from '{user_id}'. Active user was: {self._active_user}") # MODIFIED LOG
                if self._cap:
                    self._cap.release()
                    self._cap = None
                self._active_user = None
                self._current_fps = 15.0 # Reset FPS
                self._actual_width = 0
                self._actual_height = 0
                time.sleep(0.2) # Give time for resource to free up
                return True
            elif self._active_user is None and self._cap is None:
                logging.info(f"CameraService: Release called by '{user_id}', but camera was already free.")
                return True # Idempotent release
            else:
                logging.warning(f"CameraService: '{user_id}' attempted to release camera held by '{self._active_user or 'None'}'.")
                return False

    def is_active_by(self, user_id: str) -> bool:
        with self._lock:
            return self._active_user == user_id

    def is_active(self) -> bool:
        with self._lock:
            return self._active_user is not None

    def get_active_user(self) -> str | None:
        with self._lock:
            return self._active_user
            
    def get_capture_for_user(self, user_id: str) -> cv2.VideoCapture | None:
        with self._lock:
            if self._active_user == user_id:
                return self._cap
            return None

    def get_fps(self) -> float:
        with self._lock:
            # Ensure FPS is current if camera is active
            if self._cap and self._active_user:
                fps_read = self._cap.get(cv2.CAP_PROP_FPS)
                if fps_read and fps_read > 0:
                    self._current_fps = fps_read
            return self._current_fps
            
    def get_dimensions(self) -> tuple[int, int]:
        with self._lock:
            if self._cap and self._active_user:
                self._actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self._actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return self._actual_width, self._actual_height


# --- NEW: ProcessingManager Class ---
class ProcessingManager:
    def __init__(self):
        self._status: dict = {
            "status": "Idle", "current_file": None, "output_file": None,
            "error": None, "task_id": None,
            "fall_occurred_overall": False,
            "fall_snippets": [],
            "analysis_duration_seconds": None,
            "fall_event_count": 0
        }
        self._lock: threading.RLock = threading.RLock()

    def get_status(self) -> dict:
        with self._lock:
            return self._status.copy()

    def update_status(self, **kwargs) -> None:
        with self._lock:
            # Prevent overwriting output_file if status is already Completed or Error from a previous task
            current_task_id = self._status.get("task_id")
            new_task_id = kwargs.get("task_id")
            if new_task_id and current_task_id and new_task_id != current_task_id:
                 # If it's a new task, allow full update. Otherwise, be careful.
                 if "output_file" in kwargs and self._status.get("status") in ["Completed", "Error"]:
                     if kwargs.get("output_file") is None and self._status.get("output_file") is not None:
                         # Don't clear output_file if existing and new is None, unless task ID changes
                         pass # Keep existing output_file for the old task
            
            self._status.update(kwargs)
            logging.debug(f"ProcessingManager: Status updated - {kwargs}. Current full status: {self._status}")


    def set_new_task(self, task_id: str, input_filename: str) -> None:
        with self._lock:
            self._status = {
                "status": "Queued",
                "current_file": input_filename,
                "output_file": None, # Explicitly None for new task
                "error": None,
                "fall_occurred_overall": False,
                "fall_snippets": [],
                "task_id": task_id,
                "analysis_duration_seconds": None,
                "fall_event_count": 0
            }
            logging.info(f"ProcessingManager: New task set - ID: {task_id}, File: {input_filename}")

    def is_busy(self) -> bool:
        with self._lock:
            # Considers file processing busy. Webcam/Clap/ASL are handled by CameraService state.
            return self._status["status"] not in ["Idle", "Completed", "Error"]

    def get_current_task_id(self) -> str | None:
        with self._lock:
            return self._status.get("task_id")

    def set_error(self, error_message: str, task_id_check: str | None = None) -> None:
        with self._lock:
            current_task_id = self._status.get("task_id")
            if task_id_check is None or current_task_id == task_id_check:
                self._status.update(
                    status="Error", error=error_message,
                    # current_file might remain to show what failed, or set to None
                    # output_file=None, # Keep output_file if it was partially created and might be relevant
                    fall_occurred_overall=False, # Reset results for this task
                    fall_snippets=[],
                    fall_event_count=0,
                    analysis_duration_seconds=None
                )
                logging.error(f"ProcessingManager: Error set for task {current_task_id or 'N/A'} - {error_message}")
            else:
                logging.warning(f"ProcessingManager: set_error called for task {task_id_check} but current task is {current_task_id}. Ignoring error set for old task.")


    def set_completed(self, task_id_check: str, fall_occurred: bool, snippets: list, event_count: int, duration: float, output_filename: str | None) -> None:
        with self._lock:
            if self._status.get("task_id") == task_id_check:
                self._status.update(
                    status="Completed", current_file=None,
                    output_file=output_filename or self._status.get("output_file"), # Preserve if already set, update if new
                    fall_occurred_overall=fall_occurred,
                    fall_snippets=snippets,
                    fall_event_count=event_count,
                    analysis_duration_seconds=duration,
                    error=None # Clear any previous transient error for this task
                )
                logging.info(f"ProcessingManager: Task {task_id_check} completed. Output: {self._status.get('output_file')}, Falls: {event_count}, Duration: {duration:.2f}s")
            else:
                 logging.warning(f"ProcessingManager: set_completed called for task {task_id_check} but current task is {self._status.get('task_id')}. Ignoring.")

    def reset_to_idle_if_not_busy(self) -> None:
        with self._lock:
            if not self.is_busy(): # Check if truly idle (no file processing)
                self._status = {
                    "status": "Idle", "current_file": None, "output_file": None,
                    "error": None, "task_id": None,
                    "fall_occurred_overall": False,
                    "fall_snippets": [],
                    "analysis_duration_seconds": None,
                    "fall_event_count": 0
                }
                logging.info("ProcessingManager: Status reset to Idle as system was not busy with a file.")
            else:
                logging.info(f"ProcessingManager: reset_to_idle_if_not_busy called, but system is busy with task {self._status.get('task_id')}. No change.")


    def get_specific_status_field(self, field_name: str, default_value=None):
        with self._lock:
            return self._status.get(field_name, default_value)

# --- Flask App Setup ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a_very_secure_default_key')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'a_default_very_secret_key_merged_app') # Updated key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///security_system_merged.db' # MERGED DATABASE
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
socketio = SocketIO(app, async_mode='threading')

# --- Instantiate Services ---
camera_service = CameraService(CAMERA_INDEX)
processing_manager = ProcessingManager()

# --- Global Variables ---
# output_frame, lock are for fall detection webcam MJPEG stream
output_frame = None 
lock = threading.Lock() 
# asl_output_frame, asl_lock are for ASL MJPEG stream
asl_output_frame = None 
asl_lock = threading.Lock() 

webcam_thread = None # Manages the fall detection thread object
asl_video_thread = None # Manages the ASL processing thread object


alert_preferences = {}
prefs_lock = threading.RLock() # Lock for alert_preferences
session_start_time = None # For clap_tracker

# ASL detector instance (model logic) - initialized later
asl_detector = None

# --- Global Variables for Face Recognition (FR) --- # ADDED_FR
landmark_detector_fr = None
face_recognizer_fr = None
dnn_detector_fr = None
known_face_data_fr = {"embeddings": [], "labels": []} # Stores loaded embeddings and corresponding usernames

# State for real-time face login recognition
login_recognition_state_fr = {
    "status": "initializing",  # e.g., "initializing", "no_face", "unknown_face", "recognized", "error"
    "username": None,          # Username if recognized
    "timestamp": 0             # Timestamp of the last recognition update
}

# State for the multi-step face registration process
registration_process_state_fr = {
    "status": "idle", # "idle", "camera_started", "capturing_pose_X", "processing", "completed", "error"
    "current_pose_index": 0,
    "total_images_captured": 0, # For the current pose
    "instruction": "Please start the camera for registration.",
    "error_message": None,
    "face_detected_for_auto_capture": False,
    "auto_capture_countdown": None,
    "auto_capture_start_time": None,
    "captured_embeddings_for_user": [] # Stores embeddings during one registration session
}
face_rec_state_lock = threading.Lock() # Lock for FR state variables
temp_registration_data = {} # Stores username and PIN during registration flow


# --- Global Variables ---
# output_frame, lock are for fall detection webcam MJPEG stream
output_frame = None 
lock = threading.Lock() 
# asl_output_frame, asl_lock are for ASL MJPEG stream
asl_output_frame = None 
asl_lock = threading.Lock() 

webcam_thread = None # Manages the fall detection thread object
asl_video_thread = None # Manages the ASL processing thread object


alert_preferences = {}
prefs_lock = threading.RLock() # Lock for alert_preferences
session_start_time = None # For clap_tracker

# ASL detector instance (model logic) - initialized later
asl_detector = None


# --- Database Model ---
class ClapSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    start_time = db.Column(db.DateTime, nullable=False, default=datetime.datetime.utcnow)
    end_time = db.Column(db.DateTime, nullable=False)
    duration_seconds = db.Column(db.Integer, nullable=False)
    clap_count = db.Column(db.Integer, nullable=False)
    target_claps = db.Column(db.Integer, nullable=False)
    calories_burned = db.Column(db.Float, nullable=True)

    def __repr__(self):
        return f'<ClapSession {self.id} - {self.clap_count} claps>'

# --- >>> NEW: FallEvent Database Model <<< ---
class FallEvent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.datetime.utcnow)
    source = db.Column(db.String(100), nullable=False) # e.g., "Live Webcam", "File: video.mp4"
    person_id = db.Column(db.String(50), nullable=True) # If tracking is used
    snippet_filename = db.Column(db.String(255), nullable=True) # Path to the saved video snippet
    # Optional: Add confidence score, duration if available
    # fall_confidence = db.Column(db.Float, nullable=True)
    # fall_duration_seconds = db.Column(db.Integer, nullable=True)

    def __repr__(self):
        return f'<FallEvent {self.id} at {self.timestamp} from {self.source}>'
# --- >>> END NEW <<< ---

# --- Pose Estimator (initialized later after loading prefs) ---
pose_estimator = None
use_tracking_effective = False
# --- Fall Detection Smoothing Parameters (loaded later) ---
FALL_CONFIRM_FRAMES = 3
FALL_CLEAR_FRAMES = 5

# --- NEW: Instantiate Clap Tracker ---
# clap_tracker = ClapTracker(camera_index=CAMERA_INDEX, target_claps=7)
# MODIFIED: ClapTracker now needs camera_service.acquire('clap_tracker')
# You MUST update ClapTracker class in clap_tracker.py:
# 1. Modify __init__ to accept camera_service, desired_width, desired_height.
# 2. In start(): call self.camera_service.acquire('clap_tracker', self.desired_width, self.desired_height)
# 3. In stop(): call self.camera_service.release('clap_tracker')
# 4. is_active() should return self.camera_service.is_active_by('clap_tracker')
# 5. Internal camera loop should use the cv2.VideoCapture object from acquire().
clap_tracker = ClapTracker(camera_service=camera_service, target_claps=7, desired_width=FRAME_WIDTH, desired_height=FRAME_HEIGHT) # Pass FRAME_WIDTH/HEIGHT or specific clap tracker dimensions
# --- END Instantiate Clap Tracker ---

# --- Pose Estimator Initialization ---
try:
    logging.info(f"Initializing Pose Estimator ({POSE_MODEL_TYPE}) on device: {DEVICE}...")
    pose_estimator = EnhancedPoseEstimation(model_type=POSE_MODEL_TYPE, device=DEVICE)
    use_tracking_effective = USE_TRACKING and (POSE_MODEL_TYPE == 'yolov8')
    logging.info(f"Pose Estimator Initialized. Tracking: {'ON' if use_tracking_effective else 'OFF'}")
except Exception as e:
     logging.error(f"Critical error initializing pose estimator: {e}", exc_info=True)

# --- Helper Functions ---
def send_telegram_alert(frame_to_send, message):
    """Sends alert via Telegram (if configured)."""
    if not bot or not CHAT_ID:
        logging.warning("Telegram bot/chat ID not configured in app.py/detection.py. Skipping alert.")
        return
    try:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_alert_dir = "temp_alerts"
        os.makedirs(temp_alert_dir, exist_ok=True)
        fn = os.path.join(temp_alert_dir, f"alert_flask_{ts}.jpg")

        if not cv2.imwrite(fn, frame_to_send):
            logging.error(f"Failed to write alert image: {fn}")
            return

        logging.info(f"Sending Telegram alert: {message.splitlines()[0]}")
        bot.sendMessage(CHAT_ID, message)
        with open(fn, 'rb') as photo:
            bot.sendPhoto(CHAT_ID, photo=photo)

        try:
            if os.path.exists(fn): os.remove(fn)
        except Exception as e_rem:
             logging.warning(f"Couldn't remove temp alert file {fn}: {e_rem}")
    except Exception as e:
        logging.error(f"Failed to send Telegram alert: {e}", exc_info=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_fall_snippet(frame_buffer, snippet_filename, fps, width, height, task_id):
    """Saves the buffered frames as a snippet video."""
    snippet_path = os.path.join(app.config['PROCESSED_FOLDER'], snippet_filename)
    print(f"[Task {str(task_id)[:8]}] Saving snippet: {snippet_filename} ({len(frame_buffer)} frames)")
    logging.info(f"[Task {str(task_id)[:8]}] Saving snippet: {snippet_filename} ({len(frame_buffer)} frames)")

    writer = None
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1') # H.264
        writer = cv2.VideoWriter(snippet_path, fourcc, fps, (width, height))
        if not writer.isOpened():
             logging.warning(f"[Task {str(task_id)[:8]}] Failed snippet writer with 'avc1'. Trying 'mp4v'.")
             fourcc = cv2.VideoWriter_fourcc(*'mp4v') # MPEG-4
             writer = cv2.VideoWriter(snippet_path, fourcc, fps, (width, height))
             if not writer.isOpened(): raise IOError(f"Could not open snippet writer: {snippet_path}")

        frames_written = 0
        for frame in frame_buffer:
            if frame is not None and frame.shape[1] == width and frame.shape[0] == height:
                 writer.write(frame)
                 frames_written += 1
            else:
                 logging.warning(f"[Task {str(task_id)[:8]}] Skipping invalid frame in snippet buffer.")

        if frames_written == 0:
            logging.error(f"[Task {str(task_id)[:8]}] No valid frames written to snippet {snippet_filename}.")
            return False

        return True
    except Exception as e:
        logging.error(f"[Task {str(task_id)[:8]}] Error saving snippet {snippet_filename}: {e}", exc_info=True)
        if os.path.exists(snippet_path):
             try: os.remove(snippet_path)
             except OSError: pass
        return False
    finally:
        if writer: writer.release()


# --- MODIFIED Preference Loading ---
def load_preferences():
    """Loads preferences from file and sets global fall detection parameters."""
    global alert_preferences, FALL_CONFIRM_FRAMES, FALL_CLEAR_FRAMES, pose_estimator
    default_prefs = {
         "fall_detection": {
             "enabled": True, "sensitivity": "Medium (Balanced)", "play_sound": True,
             # Store the specific thresholds in the config now
             "min_kpt_confidence": DEFAULT_MIN_KPT_CONFIDENCE,
             "velocity_threshold": DEFAULT_FALL_VELOCITY_THRESHOLD,
             "confidence_threshold": DEFAULT_FALL_CONFIDENCE_THRESHOLD,
             "velocity_boost": 0.4, # Assuming default, add if needed
             "confirm_frames": 3,
             "clear_frames": 5
         },
         "motion_detection": {"enabled": True},
         "inactivity_detection": {"enabled": True, "threshold": "30 minutes", "play_sound": True},
         "notifications": {"in_app": True, "email": True, "sms": False, "phone_call": False},
         "emergency_contacts": {
             "primary": {"name": "", "relationship": "", "phone": "", "email": ""},
             "secondary": {"name": "", "relationship": "", "phone": "", "email": ""}
         },
         "alert_schedule": {"mode": "247", "custom_start": "09:00", "custom_end": "17:00"},
         # --- >>> NEW: Email Alerting Defaults <<< ---
         "email_alerting": {
            "enabled": False, # Default OFF for safety
            "smtp_server": "smtp.gmail.com", # Example: Gmail
            "smtp_port": 587, # Common for TLS
            "smtp_user": "your_email@gmail.com",
            "smtp_password": "YOUR_APP_PASSWORD", # !!! USE APP PASSWORD for Gmail/Others !!!
            "sender_email": "your_email@gmail.com", # Can be same as user or different
            "recipient_emails": ["alert_recipient@example.com"] # List of recipients
        }
         # --- >>> END NEW <<< ---
    }
    try:
        with prefs_lock:
            if os.path.exists(CONFIG_FILE):
                 with open(CONFIG_FILE, 'r') as f:
                     loaded = json.load(f)
                 # Merge loaded prefs into defaults carefully
                 current_prefs = default_prefs.copy()
                 for key, value in loaded.items():
                      if isinstance(value, dict) and key in current_prefs and isinstance(current_prefs[key], dict):
                          # Ensure nested email alerting dict is also updated correctly
                          if key == "email_alerting":
                              current_prefs[key].update(value)
                          else:
                              current_prefs[key].update(value) # Update other nested dicts
                      else:
                           current_prefs[key] = value
                 alert_preferences = current_prefs
                 logging.info(f"Loaded preferences from {CONFIG_FILE}")
            else:
                 alert_preferences = default_prefs
                 logging.info(f"Config file {CONFIG_FILE} not found. Using default preferences.")
                 save_preferences() # Save defaults if file missing

            fall_prefs = alert_preferences.get("fall_detection", {})
            FALL_CONFIRM_FRAMES = fall_prefs.get("confirm_frames", 3)
            FALL_CLEAR_FRAMES = fall_prefs.get("clear_frames", 5)
            logging.info(f"Fall Smoothing Params: Confirm={FALL_CONFIRM_FRAMES}, Clear={FALL_CLEAR_FRAMES}")

            # --- >>> MODIFICATION START: SET Pose Estimator Parameters <<< ---
            if pose_estimator:
                pose_estimator.min_kpt_confidence = fall_prefs.get("min_kpt_confidence", DEFAULT_MIN_KPT_CONFIDENCE)
                pose_estimator.fall_velocity_threshold = fall_prefs.get("velocity_threshold", DEFAULT_FALL_VELOCITY_THRESHOLD)
                pose_estimator.fall_confidence_threshold = fall_prefs.get("confidence_threshold", DEFAULT_FALL_CONFIDENCE_THRESHOLD)
                # pose_estimator.fall_velocity_boost = fall_prefs.get("velocity_boost", 0.4) # Add if using
                logging.info(f"Applied loaded thresholds to Pose Estimator:")
                logging.info(f"  - Kpt Conf: {pose_estimator.min_kpt_confidence:.2f}")
                logging.info(f"  - Vel Thresh: {pose_estimator.fall_velocity_threshold:.2f}")
                logging.info(f"  - Fall Conf Thresh: {pose_estimator.fall_confidence_threshold:.2f}")
            else:
                 logging.warning("load_preferences called before pose_estimator was initialized.")
            # --- >>> MODIFICATION END <<< ---

    except Exception as e:
        logging.error(f"Error loading preferences: {e}. Using default preferences.", exc_info=True)
        alert_preferences = default_prefs
        FALL_CONFIRM_FRAMES = default_prefs["fall_detection"]["confirm_frames"]
        FALL_CLEAR_FRAMES = default_prefs["fall_detection"]["clear_frames"]
        # --- >>> MODIFICATION START: Apply defaults on error <<< ---
        if pose_estimator:
             pose_estimator.min_kpt_confidence = default_prefs["fall_detection"]["min_kpt_confidence"]
             pose_estimator.fall_velocity_threshold = default_prefs["fall_detection"]["velocity_threshold"]
             pose_estimator.fall_confidence_threshold = default_prefs["fall_detection"]["confidence_threshold"]
             logging.info("Applied DEFAULT thresholds to Pose Estimator due to load error.")
        # --- >>> MODIFICATION END <<< ---
        try: save_preferences()
        except Exception as se: logging.error(f"Error saving defaults after load error: {se}", exc_info=True)


# --- MODIFIED Preference Saving ---
def save_preferences():
    """Saves current preferences dictionary to file."""
    global alert_preferences, FALL_CONFIRM_FRAMES, FALL_CLEAR_FRAMES # ADDED FALL_... globals
    try:
        with prefs_lock:
             # Ensure the fall_detection key exists before saving
             if "fall_detection" not in alert_preferences:
                  alert_preferences["fall_detection"] = {} # Create if missing
             # Ensure email_alerting key exists before saving (might not if loaded old config)
             if "email_alerting" not in alert_preferences:
                  alert_preferences["email_alerting"] = { # Add defaults if missing
                        "enabled": False, "smtp_server": "smtp.gmail.com", "smtp_port": 587,
                        "smtp_user": "", "smtp_password": "", "sender_email": "", "recipient_emails": []
                  }

             # --- >>> MODIFICATION START: Update dict from pose_estimator <<< ---
             # This ensures live changes made via sliders are saved
             if pose_estimator:
                  alert_preferences["fall_detection"]["min_kpt_confidence"] = getattr(pose_estimator, 'min_kpt_confidence', DEFAULT_MIN_KPT_CONFIDENCE)
                  alert_preferences["fall_detection"]["velocity_threshold"] = getattr(pose_estimator, 'fall_velocity_threshold', DEFAULT_FALL_VELOCITY_THRESHOLD)
                  alert_preferences["fall_detection"]["confidence_threshold"] = getattr(pose_estimator, 'fall_confidence_threshold', DEFAULT_FALL_CONFIDENCE_THRESHOLD)
                  # alert_preferences["fall_detection"]["velocity_boost"] = getattr(pose_estimator, 'fall_velocity_boost', 0.4) # Add if using
                  # Also save smoothing params if they become adjustable later
                  alert_preferences["fall_detection"]["confirm_frames"] = FALL_CONFIRM_FRAMES
                  alert_preferences["fall_detection"]["clear_frames"] = FALL_CLEAR_FRAMES
             # --- >>> MODIFICATION END <<< ---

             with open(CONFIG_FILE, 'w') as f:
                 json.dump(alert_preferences, f, indent=4)
             logging.info(f"Saved preferences to {CONFIG_FILE}")
    except Exception as e:
        logging.error(f"Error saving preferences: {e}", exc_info=True)


# --- NEW: Email Sending Function ---
def send_email_alert(subject, body, recipients, attachment_path=None):
    """Sends an email alert with optional image attachment."""
    global alert_preferences
    with prefs_lock:
        email_config = alert_preferences.get("email_alerting", {})
        prefs_notifications = alert_preferences.get("notifications", {})

    # Check if email alerting is enabled globally in this config section AND in general notifications
    if not email_config.get("enabled", False) or not prefs_notifications.get("email", False):
        logging.info("Email alerting is disabled in configuration. Skipping email.")
        return

    # --- Configuration ---
    smtp_server = email_config.get("smtp_server")
    smtp_port = email_config.get("smtp_port", 587) # Default to 587 (TLS)
    smtp_user = email_config.get("smtp_user")
    smtp_password = email_config.get("smtp_password") # !!! SECURITY RISK in config file !!!
    sender_email = email_config.get("sender_email", smtp_user) # Use user email if sender not specified

    # --- Get Recipients ---
    # Prioritize specific recipients if passed, otherwise use config list or emergency contacts
    if not recipients:
        recipients = email_config.get("recipient_emails", [])
        # Optionally add emergency contacts if recipients list is empty
        # if not recipients:
        #     primary_email = alert_preferences.get("emergency_contacts", {}).get("primary", {}).get("email")
        #     secondary_email = alert_preferences.get("emergency_contacts", {}).get("secondary", {}).get("email")
        #     if primary_email: recipients.append(primary_email)
        #     if secondary_email: recipients.append(secondary_email)

    if not recipients:
        logging.warning("No valid email recipients configured or provided. Cannot send email alert.")
        return
    # Filter out empty strings or None values
    recipients = [r for r in recipients if r and isinstance(r, str)]
    if not recipients:
        logging.warning("Recipient list became empty after filtering. Cannot send email alert.")
        return


    if not all([smtp_server, smtp_port, smtp_user, smtp_password, sender_email]):
        logging.error("Email configuration incomplete (server, port, user, password, sender). Cannot send email.")
        return

    # --- Create the email message ---
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = ", ".join(recipients) # Join list for display, sendmail handles list
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    # --- Attach image if provided ---
    if attachment_path and os.path.exists(attachment_path):
        try:
            with open(attachment_path, "rb") as attachment:
                part = MIMEApplication(
                    attachment.read(),
                    Name=os.path.basename(attachment_path)
                )
            # After the file is closed
            part['Content-Disposition'] = f'attachment; filename="{os.path.basename(attachment_path)}"'
            msg.attach(part)
            logging.info(f"Attached image {os.path.basename(attachment_path)} to email.")
        except Exception as e_attach:
            logging.error(f"Failed to attach image {attachment_path} to email: {e_attach}", exc_info=True)

    # --- Send the email ---
    try:
        context = ssl.create_default_context()
        # Check if using standard SSL port first
        if smtp_port == 465:
             server = smtplib.SMTP_SSL(smtp_server, smtp_port, context=context)
             server.login(smtp_user, smtp_password)
             logging.info(f"Connected to SMTP server {smtp_server}:{smtp_port} using SSL.")
        else: # Assume TLS for other ports like 587
             server = smtplib.SMTP(smtp_server, smtp_port, timeout=20) # Add timeout
             server.starttls(context=context) # Secure the connection
             server.login(smtp_user, smtp_password)
             logging.info(f"Connected to SMTP server {smtp_server}:{smtp_port} using TLS.")

        server.sendmail(sender_email, recipients, msg.as_string())
        server.quit()
        logging.info(f"Email alert sent successfully to: {', '.join(recipients)}")

    except smtplib.SMTPAuthenticationError:
         logging.error(f"SMTP Authentication Error for user {smtp_user}. Check username/password (especially if using Gmail App Passwords).")
    except smtplib.SMTPException as e_smtp:
        logging.error(f"SMTP Error sending email: {e_smtp}", exc_info=True)
    except socket.gaierror:
         logging.error(f"Network Error: Could not resolve SMTP server hostname '{smtp_server}'. Check server name and network connection.")
    except Exception as e:
        logging.error(f"Failed to send email alert: {e}", exc_info=True)


# --- ASL Feature Helper Functions (Moved from sign_app.py) --- # ADDED
def initialize_asl_detector():
    """Initializes ONLY the SignDetector for ASL."""
    global asl_detector
    logging.info("ASL: Initializing Sign Detector...")
    asl_detector = SignDetector(
        model_path=ASL_MODEL_PATH,
        train_path_for_labels=ASL_TRAIN_PATH_FOR_LABELS,
        img_height=ASL_IMG_HEIGHT,
        img_width=ASL_IMG_WIDTH,
        conf_threshold=ASL_CONF_THRESHOLD,
        stability_threshold_time=ASL_STABILITY_THRESHOLD_TIME
    )
    if asl_detector.model is None:
         logging.error("ASL ERROR: Model failed to load in SignDetector.")
    else:
        logging.info("ASL: Sign Detector initialized.")

def release_asl_camera():
    """Releases the ASL camera via CameraService."""
    global asl_video_thread # Keep to manage thread if needed
    logging.info("ASL: Attempting to release camera via CameraService...")
    if camera_service.release('asl'):
        logging.info("ASL: Camera released successfully by CameraService.")
    else:
        logging.warning("ASL: CameraService reported failure or camera not held by ASL to release.")
    
    # output_frame should be cleared by the thread stopping or here
    with asl_lock:
        global asl_output_frame
        asl_output_frame = None
    
    # If the thread is managed globally and needs explicit joining after release
    # if asl_video_thread and asl_video_thread.is_alive():
    #     asl_video_thread.join(timeout=1.0) # Example


def generate_asl_frames():
    """Generator function for ASL - captures frames using CameraService."""
    global asl_output_frame, asl_lock, asl_detector # asl_camera_active removed
    global ASL_ROI_X, ASL_ROI_Y, ASL_ROI_W, ASL_ROI_H 

    frame_counter = 0
    last_log_time = time.time()

    # Acquire camera. This function runs in a thread, acquire should be managed by the thread starter.
    # For this refactor, assume acquire is done before this thread starts, and this function receives the cap object.
    # Or, the loop itself checks camera_service.is_active_by('asl') and gets cap.
    # Let's go with the latter for self-containment of the loop logic.

    local_cap_asl = None

    while True: # Loop driven by thread state, not a global like asl_camera_active
        if not camera_service.is_active_by('asl'):
            if local_cap_asl: # If we had a cap, but service says no longer active
                logging.info("ASL: CameraService reports ASL no longer active user. Stopping frame generation.")
                # camera_service.release('asl') # Release should be handled by the entity that acquired.
                                              # Or if this loop is the sole manager, then release here.
                                              # For now, assume stop_camera route handles release.
                local_cap_asl = None # Invalidate local reference
            time.sleep(0.2)
            continue

        if not local_cap_asl: # Try to get the capture object if we are supposed to be active
            local_cap_asl = camera_service.get_capture_for_user('asl')
            if not local_cap_asl:
                logging.warning("ASL: Active by service, but no capture object. Waiting...")
                time.sleep(0.5)
                continue
            else:
                logging.info("ASL: Successfully got capture object from CameraService.")


        ret, frame = local_cap_asl.read()
        if not ret or frame is None:
            logging.error("ASL Error: Failed to capture frame from ASL webcam via CameraService.")
            # Consider if this means camera was lost; may need re-acquire or rely on CameraService state.
            if not local_cap_asl.isOpened(): # Check if the stream died
                logging.error("ASL Error: Capture object no longer opened.")
                camera_service.release('asl') # Force release if cap died
                local_cap_asl = None
            time.sleep(0.5)
            continue

        frame = cv2.flip(frame, 1)
        # Ensure ROI coordinates are integers and within frame bounds
        roi_y_end = ASL_ROI_Y + ASL_ROI_H
        roi_x_end = ASL_ROI_X + ASL_ROI_W

        # Clamp ROI to be within frame dimensions to prevent errors
        actual_roi_y = max(0, ASL_ROI_Y)
        actual_roi_x = max(0, ASL_ROI_X)
        actual_roi_y_end = min(frame.shape[0], roi_y_end)
        actual_roi_x_end = min(frame.shape[1], roi_x_end)

        if actual_roi_y_end <= actual_roi_y or actual_roi_x_end <= actual_roi_x:
            logging.warning("ASL ROI is outside frame or has zero size, skipping ROI processing.")
            roi_gray = None
        else:
            roi = frame[actual_roi_y:actual_roi_y_end, actual_roi_x:actual_roi_x_end]
            if roi.size == 0:
                logging.warning("ASL ROI is empty after clamping, skipping ROI processing.")
                roi_gray = None
            else:
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        if asl_detector and roi_gray is not None:
            current_frame_pred, current_frame_conf = asl_detector.process_frame(roi_gray)
            cv2.rectangle(frame, (actual_roi_x, actual_roi_y), (actual_roi_x_end, actual_roi_y_end), (0, 255, 0), 2)
            display_pred_label = current_frame_pred if current_frame_pred else "-"
            display_conf_label = f"{current_frame_conf:.2f}" if current_frame_pred else "-"
            cv2.putText(frame, f"Detect: {display_pred_label} ({display_conf_label})", (actual_roi_x, actual_roi_y - 10 if actual_roi_y > 10 else 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            translated = asl_detector.get_translated_text()
            text_display_y = frame.shape[0] - 30 # Adjust Y position based on frame height
            cv2.putText(frame, f"Text: {translated}", (10, text_display_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        elif roi_gray is None:
            cv2.putText(frame, "ASL ROI Error", (10, frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else: # asl_detector not ready
             cv2.putText(frame, "Error: ASL Detector not ready", (10, frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        with asl_lock:
            (flag, encoded_image) = cv2.imencode(".jpg", frame)
            if flag:
                asl_output_frame = encoded_image.tobytes()

        frame_counter += 1
        current_time = time.time()
        if current_time - last_log_time >= 5.0:
            # logging.debug(f"ASL: Processed {frame_counter} frames in the last 5 seconds.") # Optional
            frame_counter = 0
            last_log_time = current_time
        time.sleep(0.01) # Reduce CPU usage slightly, adjust as needed

def asl_video_stream_loop():
    """Target function for the ASL thread that runs generate_asl_frames."""
    logging.info("ASL: Video stream loop started in background thread.")
    try:
        generate_asl_frames()
    except Exception as e:
        logging.error(f"ASL Error in video stream loop: {e}", exc_info=True)
    finally:
         logging.info("ASL: Video stream loop finished.")


# --- Background Task for Processing Uploaded Video (MODIFIED FOR SMOOTHING) --- # MODIFIED globals, variable names, logging
def process_uploaded_video_background(input_path, output_path, task_id):
    """Processes an uploaded video file, applies smoothing, saves full video AND fall snippets."""
    global pose_estimator, socketio, FRAME_WIDTH, FRAME_HEIGHT, SNIPPET_BUFFER_SECONDS, FALL_CONFIRM_FRAMES, FALL_CLEAR_FRAMES
    # processing_info removed, use processing_manager

    start_time = time.monotonic()
    logging.info(f"[Task {str(task_id)[:8]}] Starting processing: {input_path}")

    if pose_estimator is None:
        error_msg = "Pose estimator not initialized."
        logging.error(f"[Task {str(task_id)[:8]}] {error_msg}")
        processing_manager.set_error(error_msg, task_id_check=task_id)
        return

    # --- Local state for this task ---
    task_fall_occurred_overall = False # Was any SMOOTHED fall detected?
    detected_fall_snippets = []
    task_fall_event_count = 0 # Count of SMOOTHED fall events
    smoothed_fall_active = defaultdict(bool)
    fall_confirm_counters = defaultdict(int)
    fall_clear_counters = defaultdict(int)
    fall_start_frame_num = defaultdict(lambda: None)
    # MODIFIED: fall_recording_buffers initialization for dynamic maxlen based on file_fps
    fall_recording_buffers = defaultdict(lambda: deque()) # Maxlen set later

    # --- Update Status ---
    # Use processing_manager, current_file already set by set_new_task
    processing_manager.update_status(status="Processing", output_file=os.path.basename(output_path), task_id=task_id)


    cap_file = None
    writer_full = None
    # success = False # success determined by flow

    try:
        # Check for cancellation before even opening files
        if processing_manager.get_current_task_id() != task_id:
            logging.info(f"[Task {str(task_id)[:8]}] Task cancelled or superseded before starting. Aborting.")
            # No status update needed as current task is different or already handled
            return

        cap_file = cv2.VideoCapture(input_path)
        if not cap_file.isOpened(): raise IOError(f"Could not open video file: {input_path}")

        file_fps = cap_file.get(cv2.CAP_PROP_FPS)
        if not file_fps or file_fps <= 0: file_fps = 30.0
        file_width = int(cap_file.get(cv2.CAP_PROP_FRAME_WIDTH))
        file_height = int(cap_file.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_width = FRAME_WIDTH if FRAME_WIDTH else file_width
        out_height = FRAME_HEIGHT if FRAME_HEIGHT else file_height

        # MODIFIED: buffer_max_len_file and setting maxlen for fall_recording_buffers
        buffer_max_len_file = int(file_fps * SNIPPET_BUFFER_SECONDS)
        # Ensure existing keys (if any, though unlikely here) get updated maxlen
        # And new keys get the correct maxlen from the defaultdict factory
        for person_id_key in list(fall_recording_buffers.keys()): # Iterate over a copy of keys if modifying dict
            current_buffer = fall_recording_buffers[person_id_key]
            new_buffer = deque(maxlen=buffer_max_len_file)
            new_buffer.extend(list(current_buffer)[-buffer_max_len_file:]) # Preserve recent items if any
            fall_recording_buffers[person_id_key] = new_buffer
        fall_recording_buffers.default_factory = lambda: deque(maxlen=buffer_max_len_file)

        # --- Initialize Full Video Writer --- # MODIFIED: logging task_id[:8] to str(task_id)[:8]
        fourcc_full = cv2.VideoWriter_fourcc(*'avc1')
        writer_full = cv2.VideoWriter(output_path, fourcc_full, file_fps, (out_width, out_height))
        if not writer_full.isOpened():
             logging.warning(f"[Task {str(task_id)[:8]}] Failed full writer with 'avc1'. Trying 'mp4v'.")
             fourcc_full = cv2.VideoWriter_fourcc(*'mp4v')
             writer_full = cv2.VideoWriter(output_path, fourcc_full, file_fps, (out_width, out_height))
             if not writer_full.isOpened(): raise IOError(f"Could not open full video writer: {output_path}")

        # MODIFIED: logging task_id[:8] to str(task_id)[:8]
        logging.info(f"[Task {str(task_id)[:8]}] Input: {file_width}x{file_height}@{file_fps:.2f}FPS | Output: {out_width}x{out_height}")

        frame_num = 0
        tracking_persist = {}

        while True:
             # Check for cancellation inside loop
             if processing_manager.get_current_task_id() != task_id:
                 logging.info(f"[Task {str(task_id)[:8]}] Task cancelled during processing.")
                 # success = False # No need for local success flag, manager handles final state
                 break 

             ret, frame = cap_file.read()
             if not ret: break # End of video
             frame_num += 1
             current_monotonic_time = time.monotonic()

             # Prepare display frame
             display_frame = frame
             if (out_width != file_width or out_height != file_height):
                 interpolation = cv2.INTER_AREA if (out_width * out_height < file_width * file_height) else cv2.INTER_LINEAR
                 display_frame = cv2.resize(frame, (out_width, out_height), interpolation=interpolation)
             if display_frame is None: continue
             display_frame = display_frame.copy()

             # --- Pose Estimation ---
             analyzed_poses = None
             current_frame_use_tracking = False # Default OFF for file processing
             if POSE_MODEL_TYPE == 'yolov8' and current_frame_use_tracking:
                 try:
                      tracking_results = pose_estimator.model.track(display_frame, persist=True, verbose=False, tracker="botsort.yaml")
                      pose_results_for_analysis = pose_estimator.model(display_frame, verbose=False)
                      analyzed_poses = pose_estimator.analyze_poses(
                           pose_results_for_analysis, current_monotonic_time, file_fps,
                           use_tracking=True, tracking_results=tracking_results)
                 except Exception as e_track:
                      # MODIFIED: logging task_id[:8] to str(task_id)[:8]
                      logging.error(f"[Task {str(task_id)[:8]}] Error during tracking/pose: {e_track}", exc_info=True)
                      analyzed_poses = {'poses': [], 'fall_detected_frame': False}
             elif POSE_MODEL_TYPE == 'yolov8':
                 try:
                      pose_results_for_analysis = pose_estimator.model(display_frame, verbose=False)
                      analyzed_poses = pose_estimator.analyze_poses(pose_results_for_analysis, current_monotonic_time, file_fps, use_tracking=False)
                 except Exception as e_pose:
                      # MODIFIED: logging task_id[:8] to str(task_id)[:8]
                      logging.error(f"[Task {str(task_id)[:8]}] Error during pose estimation: {e_pose}", exc_info=True)
                      analyzed_poses = {'poses': [], 'fall_detected_frame': False}
             elif POSE_MODEL_TYPE == 'mediapipe':
                 analyzed_poses = pose_estimator.process_frame(display_frame, current_monotonic_time, file_fps)


             processed_poses_for_drawing = []
             if analyzed_poses and analyzed_poses['poses']:
                 for pose_data in analyzed_poses['poses']:
                     person_id_raw = pose_data.get('id', -1)
                     person_id = int(person_id_raw) if person_id_raw is not None else -1
                     is_fall_raw = pose_data['is_fall']

                     # --- Apply Smoothing Logic ---
                     if is_fall_raw:
                         fall_confirm_counters[person_id] += 1
                         fall_clear_counters[person_id] = 0
                     else:
                         fall_clear_counters[person_id] += 1
                         fall_confirm_counters[person_id] = 0

                     if not smoothed_fall_active[person_id] and fall_confirm_counters[person_id] >= FALL_CONFIRM_FRAMES:
                         # MODIFIED: logging task_id[:8] to str(task_id)[:8]
                         logging.info(f"[Task {str(task_id)[:8]}] SMOOTHED Fall START detected for ID {person_id} at frame {frame_num}")
                         smoothed_fall_active[person_id] = True
                         fall_start_frame_num[person_id] = frame_num
                         fall_clear_counters[person_id] = 0
                         task_fall_occurred_overall = True

                         fall_event_data = {
                             'person_id': person_id,
                             'source': f'File ({os.path.basename(input_path)})',
                             'timestamp': datetime.datetime.now().isoformat(),
                             'task_id': task_id
                         }
                         try:
                             socketio.emit('fall_detected_event', fall_event_data)
                             # MODIFIED: logging task_id[:8] to str(task_id)[:8]
                             logging.info(f"[Task {str(task_id)[:8]}] Emitted 'fall_detected_event' via WebSocket for ID {person_id}")
                         except Exception as e_emit:
                             # MODIFIED: logging task_id[:8] to str(task_id)[:8]
                             logging.error(f"[Task {str(task_id)[:8]}] Error emitting WebSocket event: {e_emit}")

                     elif smoothed_fall_active[person_id] and fall_clear_counters[person_id] >= FALL_CLEAR_FRAMES:
                         # MODIFIED: logging task_id[:8] to str(task_id)[:8]
                         logging.info(f"[Task {str(task_id)[:8]}] SMOOTHED Fall END detected for ID {person_id} at frame {frame_num}")
                         # smoothed_fall_active[person_id] = False # Moved to after saving snippet/DB
                         # fall_confirm_counters[person_id] = 0    # Moved to after saving snippet/DB

                         start_frame = fall_start_frame_num.get(person_id, frame_num - FALL_CONFIRM_FRAMES)
                         snippet_filename = f"snippet_{str(task_id)[:8]}_id{person_id}_{start_frame}_{frame_num}.mp4"
                         buffer_to_save = list(fall_recording_buffers[person_id])
                         
                         db_snippet_filename_file = None # For database
                         if buffer_to_save:
                              snippet_saved = save_fall_snippet(buffer_to_save, snippet_filename, file_fps, file_width, file_height, task_id)
                              if snippet_saved:
                                   detected_fall_snippets.append(snippet_filename)
                                   task_fall_event_count += 1
                                   db_snippet_filename_file = snippet_filename # Store filename for DB

                                   # --- >>> NEW: Save FallEvent to Database (for file processing) <<< ---
                                   try:
                                       event_db_timestamp = datetime.datetime.utcnow()

                                       with app.app_context():
                                           new_fall_event = FallEvent(
                                               timestamp=event_db_timestamp,
                                               source=f"File: {os.path.basename(input_path)}",
                                               person_id=str(person_id) if person_id != -1 else None,
                                               snippet_filename=db_snippet_filename_file
                                           )
                                           db.session.add(new_fall_event)
                                           db.session.commit()
                                           # MODIFIED: logging task_id[:8] to str(task_id)[:8]
                                           logging.info(f"[Task {str(task_id)[:8]}] Saved FallEvent to DB: ID {new_fall_event.id} from file, Snippet: {db_snippet_filename_file}")
                                   except Exception as e_db_file:
                                       # MODIFIED: logging task_id[:8] to str(task_id)[:8]
                                       logging.error(f"[Task {str(task_id)[:8]}] Failed to save FallEvent from file to DB: {e_db_file}", exc_info=True)
                                       with app.app_context(): db.session.rollback()
                                   # --- >>> END NEW <<< ---
                         else:
                              # MODIFIED: logging task_id[:8] to str(task_id)[:8]
                              logging.warning(f"[Task {str(task_id)[:8]}] Snippet buffer empty for ID {person_id} at fall end.")
                         
                         fall_start_frame_num[person_id] = None
                         fall_recording_buffers[person_id].clear()
                         smoothed_fall_active[person_id] = False # Moved here
                         fall_confirm_counters[person_id] = 0    # Moved here

                     if smoothed_fall_active.get(person_id, False):
                         fall_recording_buffers[person_id].append(frame.copy())

                     pose_data['is_fall_smoothed'] = smoothed_fall_active[person_id]
                     processed_poses_for_drawing.append(pose_data)

             # --- Draw Skeletons ---
             for p_data in processed_poses_for_drawing:
                 pose_estimator.draw_skeleton(display_frame, p_data, draw_smoothed_state=True)

             # Add Text Overlays
             video_time = frame_num / file_fps
             cv2.putText(display_frame, f"Video Time: {video_time:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
             frame_fall_status = any(smoothed_fall_active.values())
             status_msg = "Status: FALL DETECTED (Smoothed)" if frame_fall_status else "Status: Normal"
             status_color = (0, 0, 255) if frame_fall_status else (0, 255, 0)
             cv2.putText(display_frame, status_msg, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
             cv2.putText(display_frame, f"Frame: {frame_num}", (10, out_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

             if writer_full: writer_full.write(display_frame)

        # --- End of loop --- # MODIFIED: logging task_id[:8] to str(task_id)[:8]
        for person_id, is_active in smoothed_fall_active.items():
             if is_active:
                 logging.warning(f"[Task {str(task_id)[:8]}] Fall ongoing at video end for ID {person_id}. Saving final snippet.")
                 start_frame = fall_start_frame_num.get(person_id, frame_num - FALL_CONFIRM_FRAMES)
                 snippet_filename = f"snippet_{str(task_id)[:8]}_id{person_id}_{start_frame}_end.mp4"
                 buffer_to_save = list(fall_recording_buffers[person_id])
                 if buffer_to_save:
                      # MODIFIED: use file_width/height for saving snippet from file processing
                      snippet_saved = save_fall_snippet(buffer_to_save, snippet_filename, file_fps, file_width, file_height, task_id)
                      if snippet_saved:
                           detected_fall_snippets.append(snippet_filename)
                           task_fall_event_count += 1
                 else:
                      logging.warning(f"[Task {str(task_id)[:8]}] Snippet buffer empty for ID {person_id} at video end.")
        # success = True # No need for local success flag

    except Exception as e:
        error_msg = f"Error during video processing task {str(task_id)[:8]}: {e}"
        print(error_msg)
        logging.error(error_msg, exc_info=True)
        # success = False # No need
        # Check if this task is still the current one before setting its error
        if processing_manager.get_current_task_id() == task_id:
            processing_manager.set_error(str(e), task_id_check=task_id)

    finally:
        duration = time.monotonic() - start_time
        if cap_file: cap_file.release()
        if writer_full: writer_full.release()
        logging.info(f"[Task {str(task_id)[:8]}] Released video resources. Duration: {duration:.2f}s")

        current_task_status = processing_manager.get_specific_status_field("status")
        # Only finalize if this task is still the one in the manager
        if processing_manager.get_current_task_id() == task_id:
            if current_task_status == "Processing": # If no major error set status to Completed
                processing_manager.set_completed(
                    task_id_check=task_id,
                    fall_occurred=task_fall_occurred_overall,
                    snippets=detected_fall_snippets,
                    event_count=task_fall_event_count,
                    duration=duration,
                    output_filename=os.path.basename(output_path) if os.path.exists(output_path) else None
                )
                logging.info(f"[Task {str(task_id)[:8]}] Processing finished. Duration: {duration:.2f}s. Falls: {task_fall_event_count}. Snippets: {len(detected_fall_snippets)}")
            elif current_task_status == "Error":
                 logging.info(f"[Task {str(task_id)[:8]}] Processing ended with an error. Status already set.")
                 if os.path.exists(output_path) and not detected_fall_snippets: # Remove failed full video if no snippets
                     try: os.remove(output_path); logging.info(f"[Task {str(task_id)[:8]}] Removed incomplete/error output: {output_path}")
                     except Exception as e_rem: logging.error(f"[Task {str(task_id)[:8]}] Failed to remove error output {output_path}: {e_rem}")
            # If status was "Queued" and loop was exited due to cancellation, it's handled
        else:
            logging.info(f"[Task {str(task_id)[:8]}] Task ended, but manager holds a different task ({processing_manager.get_current_task_id()}). No final status update for this task.")
        
        logging.info(f"[Task {str(task_id)[:8]}] Final manager status for this task (if current): '{processing_manager.get_specific_status_field('status', 'N/A')}', Falls: {processing_manager.get_specific_status_field('fall_event_count', 0)}, Snippets: {len(processing_manager.get_specific_status_field('fall_snippets', []))}")


# --- Webcam Processing Function (MODIFIED) ---
def process_video_stream():
    """Processes the video stream from the webcam for FALL DETECTION using CameraService."""
    global output_frame, lock, pose_estimator, use_tracking_effective, FRAME_WIDTH, FRAME_HEIGHT
    global socketio, SNIPPET_BUFFER_SECONDS, FALL_CONFIRM_FRAMES, FALL_CLEAR_FRAMES, ALERT_COOLDOWN_SECONDS
    # webcam_active, cap, current_fps, processing_info, processing_lock removed

    if pose_estimator is None:
        logging.error("Fall Detection: Pose estimator not initialized. Webcam stream cannot start.")
        # processing_manager.update_status(status="Error", error="Pose estimator not initialized.")
        # No processing_manager update here as camera acquisition hasn't happened. Route handles initial status.
        return # CameraService acquire will fail or not be called

    local_cap = camera_service.get_capture_for_user('fall_detection')
    if not local_cap:
        logging.error("Fall Detection: Failed to get capture object from CameraService. Stream cannot start.")
        # Caller (start_webcam route) should have handled this via acquire()
        return

    processing_manager.update_status(status="Streaming", error=None) # Update status once camera is confirmed
    
    # Get FPS from camera_service
    current_fps_val = camera_service.get_fps()
    cam_width, cam_height = camera_service.get_dimensions()

    is_currently_falling = False
    fall_confirm_counter = 0
    fall_clear_counter = 0
    last_alert_time = 0
    last_email_alert_time = 0 # ADDED for email alert cooldown
    alert_cooldown = ALERT_COOLDOWN_SECONDS
    buffer_max_len_live = int(current_fps_val * SNIPPET_BUFFER_SECONDS) if current_fps_val > 0 else int(15.0 * SNIPPET_BUFFER_SECONDS)
    live_fall_buffer = deque(maxlen=buffer_max_len_live)
    logging.info(f"Fall Detection: Webcam buffer initialized with maxlen: {buffer_max_len_live} frames ({SNIPPET_BUFFER_SECONDS}s @ {current_fps_val:.1f} FPS)")
    frame_num = 0
    logging.info("Fall Detection: Webcam processing thread started main loop.")

    try:
        while camera_service.is_active_by('fall_detection'): # Loop while this user owns the camera
            if not local_cap or not local_cap.isOpened(): # Re-check local_cap from service if it can change
                logging.error("Fall Detection: Webcam capture is not open or became unavailable.")
                processing_manager.update_status(status="Error", error="Webcam disconnected or failed.")
                # camera_service.release('fall_detection') # Ensure release if cap dies
                break # Exit loop

            ret, frame = local_cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue
            frame_num += 1
            current_monotonic_time = time.monotonic()
            if frame is not None: live_fall_buffer.append(frame.copy())

            display_frame = frame
            cap_width = int(local_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap_height = int(local_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_width = FRAME_WIDTH if FRAME_WIDTH else cap_width
            out_height = FRAME_HEIGHT if FRAME_HEIGHT else cap_height
            if (out_width != cap_width or out_height != cap_height):
                interpolation = cv2.INTER_AREA if (out_width * out_height < cap_width * cap_height) else cv2.INTER_LINEAR
                display_frame = cv2.resize(frame, (out_width, out_height), interpolation=interpolation)
                if display_frame is None: continue
            display_frame = display_frame.copy()

            # --- Pose Estimation (Fall Detection) ---
            analyzed_poses = {'poses': []}
            fall_detected_this_frame = False
            try:
                if POSE_MODEL_TYPE == 'yolov8' and use_tracking_effective:
                     try:
                         tracking_results = pose_estimator.model.track(display_frame, persist=True, verbose=False, tracker="botsort.yaml")
                         analyzed_poses = pose_estimator.analyze_poses(tracking_results, current_monotonic_time, current_fps_val, use_tracking=True)
                     except Exception as e_track:
                          # MODIFIED: Logging prefixes like "Fall Detection:"
                          logging.error(f"Fall Detection: [Webcam Frame {frame_num}] Error during tracking/pose: {e_track}", exc_info=True)
                          analyzed_poses = {'poses': [], 'fall_detected_frame': False}
                elif POSE_MODEL_TYPE == 'yolov8':
                     try:
                          pose_results_for_analysis = pose_estimator.model(display_frame, verbose=False)
                          analyzed_poses = pose_estimator.analyze_poses(pose_results_for_analysis, current_monotonic_time, current_fps_val, use_tracking=False)
                     except Exception as e_pose:
                          # MODIFIED: Logging prefixes like "Fall Detection:"
                          logging.error(f"Fall Detection: [Webcam Frame {frame_num}] Error during pose estimation: {e_pose}", exc_info=True)
                          analyzed_poses = {'poses': [], 'fall_detected_frame': False}
                elif POSE_MODEL_TYPE == 'mediapipe':
                     analyzed_poses = pose_estimator.process_frame(display_frame, current_monotonic_time, current_fps_val)

                if analyzed_poses and analyzed_poses['poses']:
                    for pose in analyzed_poses['poses']:
                        if pose.get('is_fall', False):
                            fall_detected_this_frame = True
                            break
            except Exception as e_proc:
                 logging.error(f"Error processing webcam frame {frame_num}: {e_proc}", exc_info=True)
                 analyzed_poses = {'poses': [], 'fall_detected_frame': False}
                 fall_detected_this_frame = False

            # --- Temporal Smoothing & Alerting Logic ---
            if fall_detected_this_frame:
                fall_confirm_counter += 1
                fall_clear_counter = 0
            else:
                fall_clear_counter += 1
                fall_confirm_counter = 0

            current_time = time.time()
            if not is_currently_falling and fall_confirm_counter >= FALL_CONFIRM_FRAMES:
                 is_currently_falling = True
                 # MODIFIED: Logging prefixes like "Fall Detection:"
                 logging.info(f"Fall Detection: Webcam Fall Event Confirmed START ~ frame {frame_num}")
                 event_timestamp = datetime.datetime.utcnow() # Consistent timestamp for all actions
                 
                 # --- Save Snippet/Image for Attachments --- # MODIFIED: Variable names like snippet_filename_live
                 snippet_url_relative = None # For websocket
                 db_snippet_filename = None # For database
                 snippet_filename_live = f"webcam_fall_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{frame_num}.mp4"

                 try:
                     # timestamp_snip = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # Defined in snippet_filename_live
                     # snippet_filename = f"webcam_fall_{timestamp_snip}_{frame_num}.mp4" # Renamed to snippet_filename_live
                     buffer_to_save = list(live_fall_buffer)
                     if buffer_to_save:
                         # MODIFIED: Logging prefixes like "Fall Detection:"
                         logging.info(f"Fall Detection: Attempting to save webcam fall snippet: {snippet_filename_live}")
                         # MODIFIED: Using cap_width, cap_height for saving snippet for correct resolution
                         snippet_saved_status = save_fall_snippet(buffer_to_save, snippet_filename_live, current_fps_val, cap_width, cap_height, "webcam_fall")
                         if snippet_saved_status:
                             # MODIFIED: Logging prefixes like "Fall Detection:"
                             logging.info(f"Fall Detection: Successfully saved webcam snippet: {snippet_filename_live}")
                             snippet_url_relative = f"/processed/{snippet_filename_live}"
                             db_snippet_filename = snippet_filename_live # Store just the filename for DB
                             # MODIFIED: Logging prefixes like "Fall Detection:"
                             logging.info(f"Fall Detection: Generated snippet URL: {snippet_url_relative}")
                         else:
                             # MODIFIED: Logging prefixes like "Fall Detection:"
                             logging.error(f"Fall Detection: Failed to save webcam snippet: {snippet_filename_live}")
                     else:
                         # MODIFIED: Logging prefixes like "Fall Detection:"
                         logging.warning("Fall Detection: Webcam fall detected, but buffer was empty. No snippet saved.")
                 except Exception as e_save:
                     # MODIFIED: Logging prefixes like "Fall Detection:"
                     logging.error(f"Fall Detection: Error occurred during webcam snippet saving or URL generation: {e_save}", exc_info=True)
                     snippet_url_relative = None # Ensure None on error
                     db_snippet_filename = None  # Ensure None on error

                 # --- >>> NEW: Save FallEvent to Database <<< ---
                 try:
                     with app.app_context(): # Ensure app context for DB operations in thread
                         new_fall_event = FallEvent(
                             timestamp=event_timestamp,
                             source="Live Webcam",
                             person_id="-1", # Webcam currently doesn't have specific person IDs unless tracking is very robust
                             snippet_filename=db_snippet_filename # Store filename
                         )
                         db.session.add(new_fall_event)
                         db.session.commit()
                         # MODIFIED: Logging prefixes like "Fall Detection:"
                         logging.info(f"Fall Detection: Saved FallEvent to DB: ID {new_fall_event.id} from Live Webcam, Snippet: {db_snippet_filename}")
                 except Exception as e_db:
                     # MODIFIED: Logging prefixes like "Fall Detection:"
                     logging.error(f"Fall Detection: Failed to save FallEvent from Live Webcam to DB: {e_db}", exc_info=True)
                     with app.app_context(): db.session.rollback()
                 # --- >>> END NEW <<< ---

                 # --- Emit WebSocket event ---
                 fall_event_data = {
                     'source': 'Live Webcam',
                     'timestamp': event_timestamp.isoformat(), # Use consistent UTC timestamp
                     'snippet_url': snippet_url_relative # Use the relative URL
                 }
                 try:
                     socketio.emit('fall_detected_event', fall_event_data)
                     # MODIFIED: Logging prefixes like "Fall Detection:"
                     logging.info(f"Fall Detection: Emitted 'fall_detected_event' via WebSocket for Webcam (Snippet URL: {'Yes' if snippet_url_relative else 'No'})")
                 except Exception as e_emit:
                     # MODIFIED: Logging prefixes like "Fall Detection:"
                     logging.error(f"Fall Detection: Error emitting WebSocket event for Webcam: {e_emit}")

                 # --- Send Telegram Alert (Threaded) ---
                 if current_time - last_alert_time > alert_cooldown:
                     timestamp_str_telegram = event_timestamp.strftime("%Y-%m-%d %H:%M:%S")
                     alert_msg_telegram = f"🚨 FALL DETECTED (Webcam)\\nTime: {timestamp_str_telegram}\\nStatus: Fall Confirmed"
                     alert_frame_telegram = display_frame.copy() # Use a fresh copy
                     threading.Thread(target=send_telegram_alert, args=(alert_frame_telegram, alert_msg_telegram), daemon=True).start()
                     last_alert_time = current_time
                 else:
                      # MODIFIED: Logging prefixes like "Fall Detection:"
                      logging.info(f"Fall Detection: Webcam Telegram Alert skipped due to cooldown. Last sent: {last_alert_time}, Current: {current_time}, Cooldown: {alert_cooldown}")

                 # --- Prepare for Email Alert --- # MODIFIED: Variable names like alert_image_path_email
                 alert_subject = f"Fall Alert (Webcam) - {event_timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
                 alert_body = f"A fall event was detected by the webcam.\nDevice: Live Webcam\nTime: {event_timestamp.strftime('%Y-%m-%d %H:%M:%S')}\nStatus: Fall Confirmed by system."
                 alert_image_path_email = None # Renamed from alert_image_path
                 temp_alert_dir = "temp_alerts"
                 os.makedirs(temp_alert_dir, exist_ok=True)
                 # MODIFIED: Variable names like email_image_filename_live
                 email_image_filename_live = os.path.join(temp_alert_dir, f"webcam_email_alert_{event_timestamp.strftime('%Y%m%d_%H%M%S%f')}.jpg")
                 alert_frame_email = display_frame.copy()

                 if cv2.imwrite(email_image_filename_live, alert_frame_email):
                     alert_image_path_email = email_image_filename_live
                     # MODIFIED: Logging prefixes like "Fall Detection:"
                     logging.info(f"Fall Detection: Saved image for email alert: {alert_image_path_email}")
                 else:
                     # MODIFIED: Logging prefixes like "Fall Detection:"
                     logging.error(f"Fall Detection: Failed to write image {email_image_filename_live} for email alert.")
                     # alert_image_path_email remains None, send_email_alert should handle this

                 # --- Send Email Alert (Threaded) --- # MODIFIED: Logging prefixes like "Fall Detection:"
                 logging.info("Fall Detection: Checking conditions for email alert...") # <-- ADDED logging prefix
                 if current_time - last_email_alert_time > alert_cooldown:
                     # MODIFIED: Logging prefixes like "Fall Detection:"
                     logging.info("Fall Detection: Email cooldown passed.") # <-- ADDED logging prefix
                     with prefs_lock: # Get recipients safely
                        email_recipients = alert_preferences.get("email_alerting", {}).get("recipient_emails", [])
                        primary_email = alert_preferences.get("emergency_contacts", {}).get("primary", {}).get("email")
                        secondary_email = alert_preferences.get("emergency_contacts", {}).get("secondary", {}).get("email")
                        # Ensure emails are strings and not None before appending and checking existence
                        if primary_email and isinstance(primary_email, str) and primary_email not in email_recipients: email_recipients.append(primary_email)
                        if secondary_email and isinstance(secondary_email, str) and secondary_email not in email_recipients: email_recipients.append(secondary_email)
                        email_enabled_specific = alert_preferences.get("email_alerting", {}).get("enabled", False) # <-- ADDED Check specific flag
                        email_enabled_general = alert_preferences.get("notifications", {}).get("email", False) # <-- ADDED Check general flag

                     # MODIFIED: Logging prefixes like "Fall Detection:"
                     logging.info(f"Fall Detection: Email Specific Enabled: {email_enabled_specific}, General Enabled: {email_enabled_general}") # <-- ADDED logging prefix
                     # Filter out any None or non-string recipients that might have slipped through or were in original list
                     email_recipients = [r for r in email_recipients if r and isinstance(r, str)]
                     # MODIFIED: Logging prefixes like "Fall Detection:"
                     logging.info(f"Fall Detection: Found email recipients: {email_recipients}") # <-- ADDED logging prefix

                     # --- Check specific and general flags BEFORE checking recipient list --- # MODIFIED: Logging prefixes like "Fall Detection:"
                     if email_enabled_specific and email_enabled_general:
                         logging.info("Fall Detection: Email is enabled in config.") # <-- ADDED logging prefix
                         if email_recipients:
                              # MODIFIED: Logging prefixes like "Fall Detection:"
                              logging.info("Fall Detection: Attempting to start email alert thread...") # <-- ADDED logging prefix
                              # Send email in a separate thread
                              email_thread = threading.Thread(
                                  target=send_email_alert,
                                  args=(alert_subject, alert_body, email_recipients, alert_image_path_email), # Pass image path
                                  daemon=True
                              )
                              email_thread.start()
                              last_email_alert_time = current_time
                         else:
                              # MODIFIED: Logging prefixes like "Fall Detection:"
                              logging.warning("Fall Detection: Email alert not started because no recipients were found after check.") # <-- ADDED logging prefix
                     else:
                          # MODIFIED: Logging prefixes like "Fall Detection:"
                          logging.warning("Fall Detection: Email alert not started because it's disabled in config (specific or general).") # <-- ADDED logging prefix
                 else:
                     # MODIFIED: Logging prefixes like "Fall Detection:"
                     logging.info(f"Fall Detection: Webcam Email Alert skipped due to cooldown. Last sent: {last_email_alert_time:.2f}, Current: {current_time:.2f}, Cooldown: {alert_cooldown}") # <-- ADDED logging prefix & Detail

                 fall_clear_counter = 0

            elif is_currently_falling and fall_clear_counter >= FALL_CLEAR_FRAMES:
                 is_currently_falling = False
                 # MODIFIED: Logging prefixes like "Fall Detection:"
                 logging.info(f"Fall Detection: Webcam Fall Event Cleared END ~ frame {frame_num}")
                 fall_confirm_counter = 0
                 fall_clear_counter = 0
                 last_alert_time = 0
                 last_email_alert_time = 0 # ADDED to reset email cooldown when event clears

            # --- Drawing Skeletons ---
            if analyzed_poses and analyzed_poses['poses']:
                for pose in analyzed_poses['poses']:
                    pose_estimator.draw_skeleton(display_frame, pose)

            # --- Add Text Overlays ---
            frame_status_msg = "Status: FALL DETECTED" if is_currently_falling else "Status: Normal"
            text_color_status = (0, 0, 255) if is_currently_falling else (0, 255, 0)
            track_stat = f"Tracking: {'ON' if use_tracking_effective else 'OFF'}"
            pose_stat = f"Pose Model: {POSE_MODEL_TYPE.upper()}"
            font = cv2.FONT_HERSHEY_SIMPLEX; font_scale = 0.6; thickness = 2; y_pos = 30; text_color = (0, 255, 0)
            cv2.putText(display_frame, f"Webcam Live (FPS: {camera_service.get_fps():.1f})", (10, y_pos), font, font_scale, text_color, thickness); y_pos += 30
            cv2.putText(display_frame, frame_status_msg, (10, y_pos), font, font_scale, text_color_status, thickness); y_pos += 30
            cv2.putText(display_frame, track_stat, (10, y_pos), font, font_scale, text_color, thickness); y_pos += 30
            cv2.putText(display_frame, pose_stat, (10, y_pos), font, font_scale, text_color, thickness); y_pos += 30
            proc_info = f"Frame: {frame_num}"
            cv2.putText(display_frame, proc_info, (10, out_height - 15), font, font_scale, (255, 255, 0), thickness)

            # --- Encode and Update Output Frame --- # MODIFIED: Renamed buffer_img for clarity from cv2.imencode
            ret_encode, buffer_img = cv2.imencode(".jpg", display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ret_encode:
                with lock:
                    output_frame = buffer_img.tobytes()
            time.sleep(0.01)

    except Exception as e:
        # MODIFIED: Logging prefixes like "Fall Detection:"
        logging.error(f"Fall Detection: Exception in webcam processing loop: {e}", exc_info=True)
        processing_manager.update_status(status="Error", error=f"Runtime error: {e}")
        # camera_service.release('fall_detection') # Ensure release on error

    finally:
        logging.info("Fall Detection: Webcam processing thread finishing.")
        # Release is handled by /stop_webcam or main shutdown.
        # If loop exited due to camera_service.is_active_by turning false, release is already handled.
        # If exited due to error, stop_webcam or shutdown will handle it.
        
        with lock:
            output_frame = None # Clear the MJPEG frame buffer

        # Update manager status only if no other activity is ongoing (like file processing)
        # And if the camera was indeed stopped for fall_detection
        if not camera_service.is_active_by('fall_detection'):
            if not processing_manager.is_busy() and not clap_tracker.is_active() and not camera_service.is_active_by('asl'):
                 processing_manager.reset_to_idle_if_not_busy()
                 logging.info("Fall Detection: Webcam thread setting manager status to Idle (if appropriate).")
            else:
                 current_pm_status = processing_manager.get_specific_status_field('status')
                 logging.info(f"Fall Detection: Webcam thread stopped. Processing manager status remains: {current_pm_status} or another camera is active.")


# --- Database Helper Functions ---
def get_clap_history_stats():
    """Queries the database and calculates history stats."""
    stats = {
        "last_session": None,
        "best_session": None,
        "weekly_avg": 0.0,
        "streak": 0,
        "total_sessions": 0,
        "total_calories_today": 0.0,
        "total_calories_overall": 0.0,
    }
    try:
        with app.app_context():
             stats["total_sessions"] = db.session.query(ClapSession).count()
             if stats["total_sessions"] == 0:
                 return stats

             last = db.session.query(ClapSession).order_by(desc(ClapSession.end_time)).first()
             if last:
                 stats["last_session"] = {
                     "claps": last.clap_count,
                     "duration": last.duration_seconds,
                     "end_time": last.end_time.strftime("%Y-%m-%d %H:%M"),
                     "target": last.target_claps,
                     "calories": last.calories_burned
                 }

             best = db.session.query(ClapSession).order_by(desc(ClapSession.clap_count)).first()
             if best:
                 stats["best_session"] = {
                     "claps": best.clap_count,
                     "duration": best.duration_seconds,
                     "end_time": best.end_time.strftime("%Y-%m-%d %H:%M"),
                 }

             one_week_ago = datetime.datetime.utcnow() - datetime.timedelta(days=7)
             weekly_sessions = db.session.query(ClapSession).filter(ClapSession.end_time >= one_week_ago).all()
             if weekly_sessions:
                 total_weekly_claps = sum(s.clap_count for s in weekly_sessions)
                 stats["weekly_avg"] = round(total_weekly_claps / len(weekly_sessions), 1)

             all_sessions_for_streak = db.session.query(ClapSession).order_by(desc(ClapSession.end_time)).all()
             if all_sessions_for_streak:
                 streak_count = 0
                 unique_days = sorted(list(set(s.end_time.date() for s in all_sessions_for_streak)), reverse=True)
                 if unique_days:
                      today = datetime.datetime.utcnow().date()
                      yesterday = today - datetime.timedelta(days=1)
                      if unique_days[0] == today or unique_days[0] == yesterday:
                          streak_count = 1
                          expected_day = unique_days[0] - datetime.timedelta(days=1)
                          for i in range(1, len(unique_days)):
                              if unique_days[i] == expected_day:
                                  streak_count += 1
                                  expected_day -= datetime.timedelta(days=1)
                              else: break
                 stats["streak"] = streak_count

             total_cal_query = db.session.query(func.sum(ClapSession.calories_burned)).scalar()
             stats["total_calories_overall"] = round(float(total_cal_query or 0.0), 1)

             today_utc = datetime.datetime.utcnow().date()
             today_cal_query = db.session.query(func.sum(ClapSession.calories_burned)).filter(
                 func.date(ClapSession.end_time) == today_utc
             ).scalar()
             stats["total_calories_today"] = round(float(today_cal_query or 0.0), 1)

    except Exception as e:
        logging.error(f"Error querying clap history stats: {e}", exc_info=True)
        stats = { "last_session": None, "best_session": None, "weekly_avg": 0.0, "streak": 0, "total_sessions": 0, "total_calories_today": 0.0, "total_calories_overall": 0.0 }

    return stats

def initialize_database():
     """Creates the database tables if they don't exist."""
     with app.app_context():
         logging.info("Checking and creating database tables if necessary...")
         db.create_all()
         logging.info("Database tables checked/created.")


# --- Flask Routes ---
@app.route('/dashboard')
def index():
    """Home page - shows controls and status."""
    # global clap_tracker, webcam_active, processing_info # webcam_active, processing_info removed
    # current_status = processing_info.copy() # Replaced
    current_processing_status_dict = processing_manager.get_status()
    
    is_fall_detection_active = camera_service.is_active_by('fall_detection')
    is_clap_tracker_active = clap_tracker.is_active() # Relies on ClapTracker's updated is_active

    processed_video_url = None
    if current_processing_status_dict.get("status") == "Completed" and current_processing_status_dict.get("output_file"):
        processed_video_url = url_for('get_processed_file', filename=current_processing_status_dict["output_file"])

    initial_clap_status = None
    try:
        initial_clap_status = clap_tracker.get_status()
    except Exception as e:
        print(f"ERROR getting clap_tracker status in index route: {e}")
        logging.error(f"ERROR getting clap_tracker status in index route: {e}", exc_info=True)
        initial_clap_status = {"target_claps": 100, "clap_count": 0}

    return render_template('index.html',
                           webcam_active=is_fall_detection_active, # Fall detection specific
                           clap_tracker_active=is_clap_tracker_active,
                           processing_status=current_processing_status_dict["status"],
                           current_file=current_processing_status_dict["current_file"],
                           processed_video_url=processed_video_url,
                           error_message=current_processing_status_dict["error"],
                           initial_clap_status=initial_clap_status)

@app.route('/status')
def get_status():
    """API endpoint to get current processing status including results."""
    # global clap_tracker, webcam_active, processing_info, processing_lock, asl_camera_active # Removed
    
    current_processing_status_dict = processing_manager.get_status()
    is_fall_detection_active = camera_service.is_active_by('fall_detection')
    is_asl_camera_active = camera_service.is_active_by('asl')


    clap_status_data = None
    try:
        clap_status_data = clap_tracker.get_status()
    except Exception as e:
        logging.error(f"Error getting clap_tracker status in /status route: {e}", exc_info=True)
        clap_status_data = {"clap_count": 0, "target_claps": 0, "calories_burned": 0.0, "target_reached": False, "active": False}

    is_clap_active = clap_status_data.get('active', False)

    clap_history_stats = get_clap_history_stats()

    clap_status_data['history'] = clap_history_stats

    processed_video_url = None
    modal_video_url = None
    trigger_fall_modal = False
    is_completed = current_processing_status_dict.get("status") == "Completed"
    output_file = current_processing_status_dict.get("output_file")
    fall_occurred = current_processing_status_dict.get("fall_occurred_overall", False)
    snippets = current_processing_status_dict.get("fall_snippets", [])
    current_task_id = current_processing_status_dict.get("task_id")

    if is_completed and output_file:
        processed_video_url = url_for('get_processed_file', filename=output_file, _external=True)

    if is_completed and fall_occurred:
        trigger_fall_modal = True
        if snippets:
             modal_video_url = url_for('get_processed_file', filename=snippets[0], _external=True)
        elif processed_video_url:
            modal_video_url = processed_video_url
            logging.warning(f"[Status] Fall detected for task {current_task_id} but no snippet file found. Modal will show full video.")
        else:
             trigger_fall_modal = False
             logging.error(f"[Status] Fall detected for task {current_task_id} but no snippet or processed video URL found.")

    fall_count = current_processing_status_dict.get("fall_event_count", 0) if is_completed else 0
    duration = current_processing_status_dict.get("analysis_duration_seconds", None) if is_completed else None

    status_response = {
        "webcam_active": is_fall_detection_active,
        "asl_camera_active": is_asl_camera_active,
        "clap_tracker_active": is_clap_active, # From clap_tracker status
        "processing_status": current_processing_status_dict["status"],
        "current_file": current_processing_status_dict["current_file"],
        "output_file": output_file,
        "processed_video_url": processed_video_url,
        "error": current_processing_status_dict["error"],
        "trigger_fall_modal": trigger_fall_modal,
        "modal_video_url": modal_video_url,
        "task_id": current_task_id,
        "fall_event_count": fall_count,
        "analysis_duration_seconds": duration,
        "clap_status": clap_status_data
    }

    return jsonify(status_response)

@app.route('/start_webcam')
def start_webcam():
    """Starts the FALL DETECTION webcam processing thread."""
    global webcam_thread # Keep thread object global
    # global webcam_active, processing_info, clap_tracker, asl_camera_active # Removed

    if camera_service.is_active_by('fall_detection'):
        flash("Fall detection webcam is already active.", "warning")
        return redirect(url_for('index'))
    
    if camera_service.is_active(): # Check if any user has the camera
        active_cam_user = camera_service.get_active_user()
        flash(f"Cannot start fall detection. Camera is already in use by '{active_cam_user}'.", "danger")
        return redirect(url_for('index'))

    if processing_manager.is_busy():
         flash("Cannot start webcam while a file is processing.", "danger")
         return redirect(url_for('index'))

    # Attempt to acquire camera
    # Desired dimensions for fall detection
    cap_instance = camera_service.acquire('fall_detection', FRAME_WIDTH, FRAME_HEIGHT)
    if not cap_instance:
        flash("Failed to acquire camera for fall detection. It might be in use or unavailable.", "danger")
        processing_manager.update_status(status="Error", error="Failed to acquire camera for fall detection.") # Or reset to idle if appropriate
        return redirect(url_for('index'))

    processing_manager.update_status(status="Starting Webcam...", error=None) # Indicates attempt

    # webcam_active = True # Removed, use camera_service state
    webcam_thread = threading.Thread(target=process_video_stream, daemon=True)
    webcam_thread.start()
    flash("Attempting to start fall detection webcam stream...", "info")
    logging.info("Fall detection webcam stream requested to start.")
    time.sleep(1) # Give thread time to start and update status via processing_manager
    return redirect(url_for('index'))

@app.route('/stop_webcam')
def stop_webcam():
    """Signals the FALL DETECTION webcam processing thread to stop."""
    global webcam_thread # Keep thread object global
    # global webcam_active # Removed

    if camera_service.is_active_by('fall_detection'):
        logging.info("Fall detection webcam requested to stop.")
        if not camera_service.release('fall_detection'):
            logging.error("Failed to release camera from 'fall_detection' via CameraService.")
            flash("Error stopping webcam. Release failed.", "danger")
            # Fall through to attempt thread join anyway
        
        if webcam_thread is not None and webcam_thread.is_alive():
             webcam_thread.join(timeout=3.0)
             if webcam_thread.is_alive():
                 logging.warning("Fall detection webcam thread did not terminate in time.")
        webcam_thread = None
        flash("Fall detection webcam stopping.", "info")
        # The thread's finally block should update processing_manager status to Idle if appropriate
    else:
        flash("Fall detection webcam is not active.", "warning")
    return redirect(url_for('index'))

def generate_mjpeg_stream():
    """Yields frames for the FALL DETECTION MJPEG stream."""
    global output_frame, lock # current_fps removed, webcam_active removed
    logging.info("MJPEG stream generator started for Fall Detection.")
    
    stream_fps_val = 15.0 # Default

    while camera_service.is_active_by('fall_detection'): # Check if fall detection is the active user
        frame_bytes = None
        with lock: # output_frame and lock remain for this specific stream
            if output_frame is not None:
                frame_bytes = output_frame
        
        current_service_fps = camera_service.get_fps() # Get live FPS
        if current_service_fps > 0 : stream_fps_val = current_service_fps


        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            # Adjust sleep based on actual camera FPS to be responsive
            sleep_time = 1.0 / (stream_fps_val + 5) if stream_fps_val > 0 else 0.05 
            time.sleep(sleep_time)
        else:
            # If no frame, but camera is supposed to be active, sleep briefly
            time.sleep(0.05)

    logging.info("MJPEG stream generator for Fall Detection stopped.")

@app.route('/video_feed')
def video_feed():
    """Video streaming route for FALL DETECTION webcam."""
    if not camera_service.is_active_by('fall_detection'):
        placeholder_path = "static/placeholder.png"
        try:
            placeholder = cv2.imread(placeholder_path)
            if placeholder is None: raise FileNotFoundError
            target_h = FRAME_HEIGHT if FRAME_HEIGHT else 480
            target_w = FRAME_WIDTH if FRAME_WIDTH else 640
            placeholder = cv2.resize(placeholder, (target_w, target_h))
            ret, buffer = cv2.imencode(".jpg", placeholder)
            if not ret: raise ValueError("Could not encode placeholder")
            placeholder_bytes = buffer.tobytes()
        except Exception as e:
            print(f"Error loading/encoding placeholder: {e}")
            target_h = FRAME_HEIGHT if FRAME_HEIGHT else 480
            target_w = FRAME_WIDTH if FRAME_WIDTH else 640
            placeholder = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Feed Off", (target_w//2 - 50, target_h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode(".jpg", placeholder)
            placeholder_bytes = buffer.tobytes()

        return Response( (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + placeholder_bytes + b'\r\n'),
                          mimetype='multipart/x-mixed-replace; boundary=frame')

    return Response(generate_mjpeg_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_clap_tracker')
def start_clap_tracker():
    """Starts the CLAP TRACKER using the ClapTracker class."""
    global clap_tracker, session_start_time # webcam_active, processing_info, asl_camera_active removed

    if clap_tracker.is_active(): # Relies on ClapTracker's updated is_active
        return jsonify({"success": False, "message": "Clap Tracker is already active."}), 400
    
    if camera_service.is_active():
        active_cam_user = camera_service.get_active_user()
        return jsonify({"success": False, "message": f"Camera is in use by '{active_cam_user}'. Stop it first."}), 400
        
    if processing_manager.is_busy():
         return jsonify({"success": False, "message": "File processing is active."}), 400

    # ClapTracker's start method should now handle camera_service.acquire('clap_tracker')
    # and return success/failure based on that.
    # processing_manager.update_status(status="Clap Tracker Starting...", error=None) # ClapTracker might manage its own transient status if needed

    try:
        session_start_time = datetime.datetime.utcnow()
        start_success, message = clap_tracker.start() # Assume clap_tracker.start() now returns status
        if not start_success:
            logging.error(f"Failed to start Clap Tracker: {message}")
            # session_start_time = None # Reset if start failed
            return jsonify({"success": False, "message": message or "Failed to start Clap Tracker (e.g. camera acquisition failed)."}), 500

        logging.info(f"Clap Tracker session started at {session_start_time}")
        return jsonify({"success": True, "message": "Clap Tracker started."})
    except Exception as e:
         logging.error(f"Error starting clap tracker: {e}", exc_info=True)
         # processing_manager.update_status(status="Error", error=f"Clap tracker start error: {e}") # If manager tracks this
         return jsonify({"success": False, "message": f"Server error starting tracker: {e}"}), 500


# --- MODIFY /stop_clap_tracker route ---
@app.route('/stop_clap_tracker')
def stop_clap_tracker():
    """Stops the CLAP TRACKER and saves the session data."""
    global clap_tracker, session_start_time
    if clap_tracker.is_active(): # Relies on ClapTracker's updated is_active
        logging.info("Clap Tracker requested to stop.")
        final_status = {}
        save_success = False
        error_message = None

        try:
            final_status = clap_tracker.get_status() # Get status before stopping
            
            stop_success, message = clap_tracker.stop() # Assumes stop handles camera_service.release and returns status
            if not stop_success:
                logging.warning(f"Clap tracker stop method reported an issue: {message}")
                # Proceed with saving data if possible

            end_time = datetime.datetime.utcnow()
            start_time = session_start_time if session_start_time else end_time

            duration = int((end_time - start_time).total_seconds()) if session_start_time else 0
            count = final_status.get('clap_count', 0)
            target = final_status.get('target_claps', 0)
            calories = final_status.get('calories_burned', 0.0)

            # --- Save to Database ---
            if count > 0 or duration > 5: # Only save meaningful sessions
                 with app.app_context(): # Need app context for db operations outside request
                    new_session = ClapSession(
                        start_time=start_time, end_time=end_time, duration_seconds=duration,
                        clap_count=count, target_claps=target, calories_burned=calories
                    )
                    db.session.add(new_session)
                    db.session.commit()
                    logging.info(f"Saved clap session to DB: Start={start_time}, End={end_time}, Claps={count}")
                    save_success = True
            else:
                 logging.info(f"Skipping saving short/empty clap session (Claps={count}, Duration={duration}s).")
                 save_success = True # Considered success even if not saved

        except Exception as e:
            logging.error(f"Failed to save clap session to database: {e}", exc_info=True)
            error_message = f"Failed to save session data: {e}"
            with app.app_context(): db.session.rollback()
            save_success = False
        finally:
              session_start_time = None # Reset start time tracker

        if save_success:
            # flash("Clap Tracker stopping. Session data saved.", "info") # Replaced by JS
             return jsonify({"success": True, "message": "Tracker stopped and session data processed."})
        else:
            # flash("Clap Tracker stopped, but failed to save session data.", "error") # Replaced by JS
             return jsonify({"success": False, "message": error_message or "Failed to save session data."}), 500

    else:
        # flash("Clap Tracker is not active.", "warning") # Replaced by JS
        return jsonify({"success": False, "message": "Clap Tracker is not active."}), 400

    # REMOVED redirect

def generate_clap_mjpeg_stream():
    """Yields frames for the CLAP TRACKER MJPEG stream using ClapTracker instance."""
    global clap_tracker # ClapTracker manages its own camera via CameraService
    logging.info("Clap Tracker MJPEG stream generator started.")

    while clap_tracker.is_active(): # Relies on ClapTracker's is_active using CameraService
        frame_bytes = clap_tracker.get_frame() # Assumes get_frame works if active

        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.05)
        else:
            time.sleep(0.1)

    logging.info("Clap Tracker MJPEG stream generator stopped.")

@app.route('/clap_tracker_feed')
def clap_tracker_feed():
    """Video streaming route for CLAP TRACKER webcam using ClapTracker instance."""
    global clap_tracker
    if not clap_tracker.is_active(): # Relies on ClapTracker's is_active
        logging.debug("/clap_tracker_feed returning 204 No Content (tracker inactive)")
        return Response(status=204)

    logging.debug("/clap_tracker_feed starting MJPEG stream generator (tracker active)")
    return Response(generate_clap_mjpeg_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handles video file upload and starts background processing."""
    # global processing_info, webcam_active, clap_tracker, asl_camera_active # Removed

    if 'video' not in request.files:
        flash('No video file selected', 'danger')
        return redirect(url_for('index'))
    file = request.files['video']
    if file.filename == '':
        flash('No video file selected', 'danger')
        return redirect(url_for('index'))
    if not allowed_file(file.filename):
        flash('Invalid file type. Allowed types: {}'.format(ALLOWED_EXTENSIONS), 'danger')
        return redirect(url_for('index'))

    if processing_manager.is_busy():
         flash("Cannot upload, system is busy processing a file.", "warning")
         return redirect(url_for('index'))
    
    if camera_service.is_active():
        active_cam_user = camera_service.get_active_user()
        flash(f"Please stop the active camera feature ('{active_cam_user}') before uploading.", "warning")
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4().hex}_{filename}")
    base, ext = os.path.splitext(filename)
    output_filename = f"{base}_processed_{uuid.uuid4().hex[:8]}{ext}"
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
    task_id = uuid.uuid4().hex

    try:
        file.save(input_path)
        print(f"Video saved to {input_path}")

        processing_manager.set_new_task(task_id, os.path.basename(input_path))
        
        processing_thread = threading.Thread(target=process_uploaded_video_background,
                                             args=(input_path, output_path, task_id),
                                             daemon=True)
        processing_thread.start()

        flash(f"Video '{filename}' uploaded. Processing started.", "success")
        return redirect(url_for('index'))

    except Exception as e:
        error_msg = f"Error starting processing: {e}"
        print(error_msg)
        logging.error(error_msg, exc_info=True)
        processing_manager.set_error(f"Error during upload: {str(e)}") # Use processing_manager
        flash(f"An error occurred: {e}", "danger")
        return redirect(url_for('index'))

@app.route('/processed/<filename>')
def get_processed_file(filename):
    """Serves the processed video file (full or snippet)."""
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=False)

@app.route('/alert_preferences', methods=['GET'])
def get_alert_preferences():
    """API endpoint to get current alert preferences."""
    with prefs_lock:
        return jsonify(alert_preferences.copy())

@app.route('/alert_preferences', methods=['POST'])
def set_alert_preferences():
    """API endpoint to update alert preferences."""
    global alert_preferences
    new_prefs = request.json
    if not new_prefs:
        return jsonify({"error": "No data provided"}), 400
    try:
        with prefs_lock:
             alert_preferences.update(new_prefs)
             save_preferences()
        return jsonify({"message": "Preferences updated successfully"}), 200
    except Exception as e:
         error_msg = f"Failed to update preferences: {e}"
         print(error_msg)
         logging.error(error_msg, exc_info=True)
         return jsonify({"error": "Internal server error while updating preferences"}), 500

@app.route('/set_clap_target', methods=['POST'], endpoint='set_clap_target')
def set_clap_target():
    global clap_tracker
    data = request.get_json()

    if not data or 'target' not in data:
        return jsonify({"success": False, "error": "Missing target value"}), 400

    new_target = data['target']
    success, message = clap_tracker.set_target(new_target)

    if success:
        logging.info(f"API Call: Successfully set clap target to {new_target}")
        return jsonify({"success": True, "message": message}), 200
    else:
        logging.error(f"API Call: Failed to set clap target to {new_target}. Reason: {message}")
        return jsonify({"success": False, "error": message}), 400

# --- NEW: Live Fall Threshold Routes ---
@app.route('/get_fall_thresholds', methods=['GET'])
def get_fall_thresholds():
    """API endpoint to get current LIVE fall detection thresholds."""
    global pose_estimator
    if pose_estimator is None:
        logging.warning("/get_fall_thresholds called but pose_estimator is None.")
        # Return defaults from the config or hardcoded defaults
        with prefs_lock:
            fall_prefs = alert_preferences.get("fall_detection", {})
            return jsonify({
                 "velocity_threshold": fall_prefs.get("velocity_threshold", DEFAULT_FALL_VELOCITY_THRESHOLD),
                 "confidence_threshold": fall_prefs.get("confidence_threshold", DEFAULT_FALL_CONFIDENCE_THRESHOLD),
                 "min_kpt_confidence": fall_prefs.get("min_kpt_confidence", DEFAULT_MIN_KPT_CONFIDENCE),
                 # Add others if needed
             })

    try:
        # Read directly from the live instance
        thresholds = {
            "velocity_threshold": pose_estimator.fall_velocity_threshold,
            "confidence_threshold": pose_estimator.fall_confidence_threshold,
            "min_kpt_confidence": pose_estimator.min_kpt_confidence,
            # "velocity_boost": pose_estimator.fall_velocity_boost, # Add if using
        }
        return jsonify(thresholds)
    except AttributeError as e:
         logging.error(f"AttributeError getting thresholds from pose_estimator: {e}")
         return jsonify({"error": "Could not retrieve thresholds from estimator"}), 500
    except Exception as e:
         logging.error(f"Error in /get_fall_thresholds: {e}", exc_info=True)
         return jsonify({"error": "Internal server error"}), 500

@app.route('/set_fall_thresholds', methods=['POST'])
def set_fall_thresholds():
    """API endpoint to update LIVE fall detection thresholds."""
    global pose_estimator, alert_preferences # Need alert_preferences if calling save_preferences
    if pose_estimator is None:
        logging.error("/set_fall_thresholds called but pose_estimator is None.")
        return jsonify({"error": "Pose estimator not available"}), 503 # Service Unavailable

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    updated_count = 0
    try:
        # --- Update LIVE Pose Estimator Instance ---
        if 'velocity_threshold' in data:
            new_val = float(data['velocity_threshold'])
            if 0.05 <= new_val <= 1.5: # Add reasonable validation
                 pose_estimator.fall_velocity_threshold = new_val
                 logging.info(f"Live Fall Threshold Update: Velocity set to {new_val:.2f}")
                 updated_count += 1
            else: logging.warning(f"Invalid velocity_threshold received: {data['velocity_threshold']}")

        if 'confidence_threshold' in data:
            new_val = float(data['confidence_threshold'])
            if 0.05 <= new_val <= 0.95:
                 pose_estimator.fall_confidence_threshold = new_val
                 logging.info(f"Live Fall Threshold Update: Confidence set to {new_val:.2f}")
                 updated_count += 1
            else: logging.warning(f"Invalid confidence_threshold received: {data['confidence_threshold']}")

        if 'min_kpt_confidence' in data:
            new_val = float(data['min_kpt_confidence'])
            if 0.05 <= new_val <= 0.9:
                 pose_estimator.min_kpt_confidence = new_val
                 logging.info(f"Live Fall Threshold Update: Min Kpt Conf set to {new_val:.2f}")
                 updated_count += 1
            else: logging.warning(f"Invalid min_kpt_confidence received: {data['min_kpt_confidence']}")

        # --- Update Preferences Dictionary & Save ---
        # This makes the live changes persistent
        if updated_count > 0:
             save_preferences() # This function now reads from pose_estimator before saving

        return jsonify({"success": True, "message": f"{updated_count} threshold(s) updated and saved."})

    except ValueError as e:
         logging.error(f"ValueError setting thresholds: {e}")
         return jsonify({"error": "Invalid threshold value provided."}), 400
    except Exception as e:
        logging.error(f"Error setting fall thresholds: {e}", exc_info=True)
        return jsonify({"error": "Internal server error setting thresholds"}), 500

# --- >>> NEW: Event History Route <<< ---
@app.route('/event_history')
def event_history_page():
    # Query both FallEvents and ClapSessions
    fall_events = FallEvent.query.order_by(FallEvent.timestamp.desc()).all()
    clap_sessions = ClapSession.query.order_by(ClapSession.end_time.desc()).all()

    # Prepare events for display, adding a 'type' field
    # and ensuring a consistent timestamp field name for sorting
    combined_events = []
    for event in fall_events:
        combined_events.append({
            'type': 'fall',
            'id': event.id,
            'display_timestamp': event.timestamp, # Use this for sorting
            'timestamp_formatted': event.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
            'source': event.source,
            'person_id': event.person_id,
            'snippet_filename': event.snippet_filename,
            'snippet_url': url_for('get_processed_file', filename=event.snippet_filename) if event.snippet_filename else None
        })

    for session in clap_sessions:
        combined_events.append({
            'type': 'clap_session',
            'id': session.id,
            'display_timestamp': session.end_time, # Use this for sorting
            'timestamp_formatted': session.end_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            'clap_count': session.clap_count,
            'target_claps': session.target_claps,
            'duration_seconds': session.duration_seconds,
            'calories_burned': session.calories_burned,
            'source': 'Buddha Clap Tracker' # Or derive from somewhere if needed
        })

    # Sort all events by their display_timestamp in descending order (most recent first)
    combined_events.sort(key=lambda x: x['display_timestamp'], reverse=True)

    # Limit the number of events to display, e.g., last 100
    max_events_to_show = 100
    combined_events = combined_events[:max_events_to_show]

    return render_template('event_history.html', events=combined_events)
# --- >>> END NEW <<< ---

# --- ASL Feature Flask Routes (Moved from sign_app.py and prefixed) --- # ADDED
@app.route('/asl/start_camera', methods=['POST'])
def asl_start_camera_endpoint():
    global asl_video_thread, asl_detector # asl_video_capture, asl_camera_active, asl_lock removed
    # global webcam_active, clap_tracker, CAMERA_INDEX # Removed direct checks

    if camera_service.is_active():
        active_user = camera_service.get_active_user()
        return jsonify({'status': 'error', 'message': f"Camera is already active by '{active_user}'. Stop it first."}), 400
    
    if processing_manager.is_busy():
        return jsonify({'status': 'error', 'message': 'File processing is active. Stop it first.'}), 400

    # Desired dimensions for ASL camera. FRAME_WIDTH/HEIGHT or specific ASL_IMG_WIDTH/HEIGHT.
    # Using FRAME_WIDTH/HEIGHT for consistency, ASL logic can ROI from this.
    cap_asl = camera_service.acquire('asl', FRAME_WIDTH, FRAME_HEIGHT) 
    if not cap_asl:
        logging.critical("ASL CRITICAL ERROR: Cannot acquire camera via CameraService.")
        return jsonify({'status': 'error', 'message': 'ASL: Cannot acquire camera.'}), 500
    
    logging.info("ASL: Webcam acquired successfully by CameraService for ASL.")
    if asl_detector:
            asl_detector.reset_translated_text() 

    # Start the ASL processing thread if not already running for ASL
    # This check might need refinement if thread can exist but be idle.
    if asl_video_thread is None or not asl_video_thread.is_alive():
        asl_video_thread = threading.Thread(target=asl_video_stream_loop, daemon=True)
        asl_video_thread.start()
        logging.info("ASL: Video processing thread (re)started for ASL.")
    else:
        logging.info("ASL: Video processing thread already running.")

    return jsonify({'status': 'success', 'message': 'ASL Camera started.'})


@app.route('/asl/stop_camera', methods=['POST'])
def asl_stop_camera_endpoint():
    global asl_video_thread # Keep to join if necessary
    logging.info("ASL: Camera stop requested.")
    
    released = camera_service.release('asl')
    if released:
        logging.info("ASL: Camera released via CameraService.")
    else:
        logging.warning("ASL: Attempted to stop ASL camera, but CameraService reported it wasn't active by ASL or failed to release.")

    if asl_detector:
        asl_detector.reset_translated_text()
    
    # The generate_asl_frames loop should now stop because camera_service.is_active_by('asl') will be false.
    # Join the thread to ensure clean shutdown of its loop.
    if asl_video_thread and asl_video_thread.is_alive():
        logging.info("ASL: Joining ASL video thread...")
        asl_video_thread.join(timeout=2.0) # Give some time for the thread to exit cleanly
        if asl_video_thread.is_alive():
            logging.warning("ASL: video thread did not join in time.")
        else:
            logging.info("ASL: video thread joined successfully.")
        # asl_video_thread = None # Optionally reset thread var if it's always recreated

    return jsonify({'status': 'success', 'message': 'ASL Camera stopped attempt processed.' if released else 'ASL Camera was not active by ASL or release failed.'})


@app.route('/asl/video_feed')
def asl_video_feed():
    """Route for the ASL MJPEG video stream."""
    def streamer_asl(): 
        global asl_output_frame, asl_lock # asl_camera_active removed
        last_frame_time_asl = time.time() 
        while camera_service.is_active_by('asl'): # Stream only if ASL is the active camera user
            with asl_lock: # asl_output_frame and asl_lock remain for this stream
                is_active_stream_check = True # Redundant if outer loop controls by service
                frame_bytes_asl = asl_output_frame 
            if is_active_stream_check and frame_bytes_asl: # is_active_stream_check can be removed
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes_asl + b'\r\n')
                last_frame_time_asl = time.time()
                time.sleep(0.03) # Adjust sleep time as needed for ASL feed responsiveness
            else:
                # If camera is inactive or no frame, send nothing or a placeholder infrequently
                if not is_active_stream_check and (time.time() - last_frame_time_asl > 1.0):
                    # Optionally send a placeholder if camera is off
                    # logging.debug("ASL Video feed: camera not active or no frame, sending placeholder logic could go here")
                    pass # Currently sends nothing if inactive
                elif time.time() - last_frame_time_asl > 1.0: # Log if active but no frame for a while
                    # logging.debug("ASL Video feed waiting for active camera/frame...") # Optional
                    pass
                time.sleep(0.1) # Wait longer if inactive or no frame
    return Response(streamer_asl(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/asl/get_text')
def asl_get_text():
    global asl_detector # asl_camera_active removed
    if asl_detector and camera_service.is_active_by('asl'): # Check if ASL is using the camera
        stable_pred_asl = asl_detector.get_stable_prediction() 
        translated_asl = asl_detector.get_translated_text() 
        last_raw_pred_asl, last_raw_conf_asl = asl_detector.last_frame_prediction, asl_detector.last_frame_confidence # Renamed
        return jsonify({
            'raw_prediction': last_raw_pred_asl if last_raw_pred_asl else "-",
            'raw_confidence': f"{last_raw_conf_asl:.2f}" if last_raw_pred_asl else "-",
            'stable_candidate': stable_pred_asl if stable_pred_asl else "-",
            'translated_text': translated_asl
        })
    elif not camera_service.is_active_by('asl'): # If ASL not using camera
         return jsonify({
            'raw_prediction': "-", 'raw_confidence': "-",
            'stable_candidate': "-", 'translated_text': "ASL Camera stopped."
        })
    else: # asl_detector not ready but camera might be active (e.g. during init error)
        return jsonify({
            'raw_prediction': "-", 'raw_confidence': "-",
            'stable_candidate': "-", 'translated_text': "Error: ASL Detector not ready."
        }), 503

@app.route('/asl/clear_text', methods=['POST'])
def asl_clear_text_backend():
    global asl_detector
    if asl_detector:
        try:
            asl_detector.reset_translated_text()
            logging.info("ASL: Backend translated text cleared.")
            return jsonify({'status': 'success', 'message': 'ASL Text cleared on backend.'}), 200
        except AttributeError:
            logging.error("ASL Error: detector object does not have 'reset_translated_text' method.")
            return jsonify({'status': 'error', 'message': 'ASL Backend detector cannot clear text.'}), 500
        except Exception as e:
            logging.error(f"ASL Error clearing backend text: {e}", exc_info=True)
            return jsonify({'status': 'error', 'message': 'ASL Error clearing text on backend.'}), 500
    else:
        logging.warning("ASL: Clear text request received but detector not ready.")
        return jsonify({'status': 'error', 'message': 'ASL Detector not ready.'}), 503


# --- User Model (for Face Recognition) --- # ADDED_FR
class User(db.Model):
    __tablename__ = 'fr_user' # To avoid potential naming conflicts with other user tables if any
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    hashed_pin = db.Column(db.String(200), nullable=False) # Changed from password to PIN
    role = db.Column(db.String(50), nullable=True) # Or False if always required
    phone_number = db.Column(db.String(20), nullable=True)
    birthday = db.Column(db.String(10), nullable=True) # Store as YYYY-MM-DD string
    embedding = db.Column(db.Text, nullable=True) # Stored as JSON string of list

    def __repr__(self):
        return f'<User {self.username}>'

# --- Model Loading, Embedding Helpers for Face Recognition (FR) --- # ADDED_FR

def load_all_models_fr():
    global landmark_detector_fr, dnn_detector_fr, face_recognizer_fr
    logging.info("FR: Loading face recognition models...")
    models_loaded = True
    error_messages = []

    if os.path.exists(FR_DLIB_LANDMARK_MODEL_PATH):
        try:
            landmark_detector_fr = dlib.shape_predictor(FR_DLIB_LANDMARK_MODEL_PATH)
            logging.info("FR:  - Dlib Landmark model loaded.")
        except Exception as e:
            error_messages.append(f"FR: Failed to load Landmark model: {e}")
            logging.error(f"FR: Failed to load Landmark model: {e}", exc_info=True)
    else:
        error_messages.append(f"FR: Dlib Landmark model not found: {FR_DLIB_LANDMARK_MODEL_PATH}")

    if os.path.exists(FR_FACE_RECOGNITION_MODEL_PATH):
        try:
            face_recognizer_fr = dlib.face_recognition_model_v1(FR_FACE_RECOGNITION_MODEL_PATH)
            logging.info("FR:  - Dlib Recognition model loaded.")
        except Exception as e:
            error_messages.append(f"FR: Failed to load Recognition model: {e}")
            logging.error(f"FR: Failed to load Recognition model: {e}", exc_info=True)
    else:
        error_messages.append(f"FR: Dlib Face Recognition model not found: {FR_FACE_RECOGNITION_MODEL_PATH}")

    if os.path.exists(FR_DNN_MODEL_FILE) and os.path.exists(FR_DNN_WEIGHTS_FILE):
        try:
            dnn_detector_fr = cv2.dnn.readNetFromCaffe(FR_DNN_MODEL_FILE, FR_DNN_WEIGHTS_FILE)
            logging.info("FR:  - OpenCV DNN detector loaded.")
        except Exception as e:
            error_messages.append(f"FR: Failed to load DNN model: {e}")
            logging.error(f"FR: Failed to load DNN model: {e}", exc_info=True)
    else:
        error_messages.append(f"FR: OpenCV DNN model files not found ({FR_DNN_MODEL_FILE}, {FR_DNN_WEIGHTS_FILE}).")

    if error_messages:
        logging.error("--- FR: MODEL LOADING ERRORS ---")
        for msg in error_messages:
            logging.error(f" - {msg}")
        models_loaded = False
    
    if not all([landmark_detector_fr, face_recognizer_fr, dnn_detector_fr]):
        logging.critical("FR: CRITICAL - One or more essential FR models failed to load.")
        models_loaded = False
        # Potentially raise an error or prevent app startup if these are critical
    
    if models_loaded:
        logging.info("FR: All essential FR models loaded successfully.")
    return models_loaded

def normalize_embedding_fr(embedding):
    if isinstance(embedding, dlib.vector):
        embedding = np.array(embedding)
    return embedding

def compute_face_embedding_fr(image_bgr, rect):
    """Computes dlib 128D face descriptor for a given face rectangle in an image."""
    if not landmark_detector_fr or not face_recognizer_fr:
        logging.warning("FR: Landmark or face recognizer model not loaded for embedding computation.")
        return None
    try:
        # Ensure image is grayscale for landmark detection if models expect it, though dlib often handles BGR
        # For dlib, BGR is usually fine.
        shape = landmark_detector_fr(image_bgr, rect)
        embedding_vector = face_recognizer_fr.compute_face_descriptor(image_bgr, shape, 1) # 1 for num_jitters
        return normalize_embedding_fr(embedding_vector)
    except Exception as e:
        logging.error(f"FR: Error computing face embedding: {e}", exc_info=True)
        return None

def is_mouth_open_fr(image_bgr, rect):
    """Checks if the mouth is open based on landmark points."""
    if not landmark_detector_fr:
        logging.warning("FR: Landmark detector not loaded for mouth open check.")
        return False
    try:
        shape = landmark_detector_fr(image_bgr, rect)
        # Points for mouth: 60-67 (outer), 48-59 (inner). Using vertical distance between inner upper and lower lip.
        # Example: top_lip_center_y (around part 51 or 62), bottom_lip_center_y (around part 57 or 66)
        # Using specific points from dlib's 68-point model for inner lip
        top_lip_y = shape.part(62).y  # Inner upper lip
        bottom_lip_y = shape.part(66).y # Inner lower lip
        distance = abs(bottom_lip_y - top_lip_y)
        # logging.debug(f"FR: Mouth open distance: {distance}")
        return distance > FR_MOUTH_OPEN_THRESHOLD
    except Exception as e:
        logging.error(f"FR: Error in is_mouth_open_fr: {e}", exc_info=True)
        return False

def load_embeddings_from_db_fr(force_reload=False):
    global known_face_data_fr
    # Check if app context is available, useful if called outside a request
    with app.app_context():
        if not force_reload and known_face_data_fr.get("embeddings") and known_face_data_fr.get("labels"):
            logging.info("FR: Using cached known face data.")
            return True

        logging.info("FR: Loading known embeddings from database...")
        local_embeddings = []
        local_labels = []
        loaded_count = 0
        try:
            users_with_embeddings = User.query.filter(User.embedding.isnot(None)).all()
            
            for user_db in users_with_embeddings:
                try:
                    if user_db.embedding:
                        embedding_list = json.loads(user_db.embedding)
                        embedding_np = np.array(embedding_list, dtype=np.float32) # Ensure consistent type
                        if embedding_np.ndim == 1 and embedding_np.size == 128: # dlib embeddings are 128D
                            local_embeddings.append(embedding_np)
                            local_labels.append(user_db.username)
                            loaded_count += 1
                        else:
                            logging.warning(f"FR: Invalid embedding structure for user {user_db.username}. Size: {embedding_np.size}, Dim: {embedding_np.ndim}. Skipping.")
                    else:
                        logging.warning(f"FR: Null embedding for user {user_db.username}. Skipping.")
                except json.JSONDecodeError as e_parse:
                    logging.warning(f"FR: Could not parse embedding for user {user_db.username}: {e_parse}")
                except Exception as e_proc:
                    logging.error(f"FR: Error processing embedding for user {user_db.username}: {e_proc}", exc_info=True)
            
            with face_rec_state_lock: # Use the new dedicated lock
                known_face_data_fr["embeddings"] = local_embeddings
                known_face_data_fr["labels"] = local_labels
            logging.info(f"FR: Loaded {loaded_count} embeddings from database.")
            return True

        except Exception as e_db:
            logging.error(f"FR: ERROR querying database for embeddings: {e_db}", exc_info=True)
            with face_rec_state_lock: # Use the new dedicated lock
                known_face_data_fr = {"embeddings": [], "labels": []} # Reset on critical error
            return False

def decode_image_fr(data_url):
    """Decodes a base64 data URL to an OpenCV image."""
    try:
        if data_url is None or ',' not in data_url:
            logging.error("FR: Invalid data URL for image decoding.")
            return None
        encoded_data = data_url.split(',', 1)[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logging.error(f"FR: Error decoding image: {e}", exc_info=True)
        return None

# --- Face Recognition (FR) Video Generators --- # ADDED_FR

def generate_face_login_feed_fr():
    """Generates video frames for face login, performing recognition."""
    global login_recognition_state_fr
    user_id_cam = "face_login_stream" # Unique ID for CameraService
    desired_width, desired_height = 640, 480
    last_recognition_time = 0
    recognition_interval = 0.5 # Seconds between recognition attempts

    logging.info(f"FR: Attempting to acquire camera for {user_id_cam}")
    cap = camera_service.acquire(user_id_cam, desired_width, desired_height)

    if not cap:
        logging.error(f"FR: Could not acquire camera for {user_id_cam}. Login feed will not start.")
        # Yield a static "camera unavailable" image or message if desired
        # For now, just stops the generator.
        camera_service.release(user_id_cam) # Ensure release if acquire failed partially
        return

    logging.info(f"FR: Camera acquired for {user_id_cam}. Starting login video stream.")

    try:
        while camera_service.is_active_by(user_id_cam):
            cap_instance = camera_service.get_capture_for_user(user_id_cam)
            if not cap_instance: # Should not happen if is_active_by is true, but good check
                logging.warning(f"FR: Lost capture instance for {user_id_cam} despite being active.")
                time.sleep(0.1)
                continue

            ret, frame = cap_instance.read()
            if not ret:
                logging.warning(f"FR: Could not read frame for {user_id_cam}.")
                time.sleep(0.05) # Wait a bit before trying again
                continue

            current_time = time.time()
            frame = cv2.flip(frame, 1) # Flip horizontally
            display_frame = frame.copy()
            (h, w) = display_frame.shape[:2]

            # Face detection using DNN
            if dnn_detector_fr:
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                dnn_detector_fr.setInput(blob)
                detections = dnn_detector_fr.forward()

                best_conf = 0.0
                best_box_coords = None # Store (startX, startY, endX, endY)
                best_rect_dlib = None    # Store dlib.rectangle

                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > FR_DNN_CONFIDENCE_THRESHOLD and confidence > best_conf:
                        best_conf = confidence
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        # Ensure box is within frame boundaries
                        startX, startY = max(0, startX), max(0, startY)
                        endX, endY = min(w - 1, endX), min(h - 1, endY)
                        
                        if startX < endX and startY < endY: # Valid box
                            best_box_coords = (startX, startY, endX, endY)
                            best_rect_dlib = dlib.rectangle(startX, startY, endX, endY)
                
                # Face Recognition part
                if best_rect_dlib and (current_time - last_recognition_time > recognition_interval):
                    last_recognition_time = current_time
                    embedding = compute_face_embedding_fr(frame, best_rect_dlib) # Use original frame for embedding

                    recognized_username_val = None
                    recognition_status_val = "no_face_detected" # Default if embedding fails or no match

                    if embedding is not None:
                        with face_rec_state_lock: # Use the new dedicated lock
                            # Make copies to avoid issues if known_face_data_fr is updated by another thread
                            current_known_embeddings = list(known_face_data_fr.get("embeddings", []))
                            current_known_labels = list(known_face_data_fr.get("labels", []))

                        if current_known_embeddings and current_known_labels:
                            try:
                                # Ensure embeddings are numpy arrays for linalg.norm
                                known_embeddings_np = np.array(current_known_embeddings, dtype=np.float32)
                                current_embedding_np = np.array(embedding, dtype=np.float32).reshape(1, -1) # Reshape for broadcasting

                                if known_embeddings_np.ndim == 2 and known_embeddings_np.shape[1] == 128 and \
                                   current_embedding_np.shape[1] == 128 :
                                    
                                    distances = np.linalg.norm(known_embeddings_np - current_embedding_np, axis=1)
                                    if distances.size > 0:
                                        min_distance_idx = np.argmin(distances)
                                        min_distance = distances[min_distance_idx]

                                        if min_distance < FR_SIMILARITY_THRESHOLD:
                                            recognized_username_val = current_known_labels[min_distance_idx]
                                            recognition_status_val = "recognized"
                                            cv2.putText(display_frame, f"User: {recognized_username_val}", (best_box_coords[0], best_box_coords[1] - 25),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                        else:
                                            recognition_status_val = "unknown_face"
                                            cv2.putText(display_frame, "Unknown User", (best_box_coords[0], best_box_coords[1] - 25),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                    else: # No known embeddings loaded
                                        recognition_status_val = "no_known_faces"

                                else: # Shape mismatch
                                     logging.warning(f"FR: Embedding shape mismatch. Known: {known_embeddings_np.shape}, Current: {current_embedding_np.shape}")
                                     recognition_status_val = "error_embedding_shape"

                            except Exception as e_dist:
                                logging.error(f"FR: Error during distance calculation: {e_dist}", exc_info=True)
                                recognition_status_val = "error_recognition"
                        else: # No known embeddings
                            recognition_status_val = "no_known_faces"
                    else: # Embedding computation failed
                        recognition_status_val = "error_embedding_compute"
                    
                    # Update global state for login status
                    with face_rec_state_lock: # Use the new dedicated lock
                        login_recognition_state_fr.update({
                            "status": recognition_status_val,
                            "username": recognized_username_val,
                            "timestamp": current_time
                        })
                        # logging.debug(f"FR: Login state updated: {login_recognition_state_fr}")

                # Draw bounding box on display_frame if a face was detected
                if best_box_coords:
                    (startX, startY, endX, endY) = best_box_coords
                    cv2.rectangle(display_frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    # Add confidence text if needed:
                    # cv2.putText(display_frame, f"{best_conf:.2f}", (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            else: # DNN detector not loaded
                cv2.putText(display_frame, "DNN Model Not Loaded", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                with face_rec_state_lock:
                    login_recognition_state_fr['status'] = "error_model_load"


            # Encode and yield the frame
            ret_enc, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret_enc:
                logging.warning(f"FR: JPEG encoding failed for {user_id_cam}")
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Small delay to control FPS and reduce CPU, adjust as needed
            # The camera's own FPS and processing time will largely dictate this.
            # time.sleep(1 / 30) # Example: Aim for ~30 FPS in the loop if camera is faster

        # The log "FR: {user_id_cam} stream loop ended..." (previously on line 2745) is intentionally removed here as per user snippet.
    except Exception as e:
        logging.error(f"FR: Exception in login video generator ({user_id_cam}): {e}", exc_info=True)
        with face_rec_state_lock: # Use the new dedicated lock
            login_recognition_state_fr.update({"status": "error_stream", "username": None})
    finally:
        logging.info(f"FR: Releasing camera for {user_id_cam} from login generator.") # Make sure this log appears
        camera_service.release(user_id_cam) # THIS IS KEY


def generate_face_registration_feed_fr():
    """Generates video frames for face registration, guiding user through poses."""
    global registration_process_state_fr
    user_id_cam = "face_reg_guidance_stream"
    desired_width, desired_height = 640, 480

    logging.info(f"FR: Attempting to acquire camera for {user_id_cam}")
    cap = camera_service.acquire(user_id_cam, desired_width, desired_height)

    if not cap:
        logging.error(f"FR: Could not acquire camera for {user_id_cam}. Registration feed will not start.")
        with face_rec_state_lock:
            registration_process_state_fr['status'] = "error_camera_unavailable"
            registration_process_state_fr['instruction'] = "Error: Camera unavailable."
        camera_service.release(user_id_cam)
        return

    logging.info(f"FR: Camera acquired for {user_id_cam}. Starting registration video stream.")
    
    try:
        while camera_service.is_active_by(user_id_cam):
            cap_instance = camera_service.get_capture_for_user(user_id_cam)
            if not cap_instance:
                time.sleep(0.1)
                continue

            ret, frame = cap_instance.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1) # Flip horizontally
            display_frame = frame.copy()
            (h, w) = display_frame.shape[:2]
            
            current_pose_idx = 0
            current_instruction = "Initializing..."
            face_detected_this_frame = False
            current_auto_capture_countdown = None

            with face_rec_state_lock: # Access shared state
                current_pose_idx = registration_process_state_fr.get("current_pose_index", 0)
                current_instruction = registration_process_state_fr.get("instruction", "Look at the camera.")
                # Check if auto-capture is active and get countdown
                if registration_process_state_fr.get("status", "").startswith("capturing_pose_") and \
                   registration_process_state_fr.get("auto_capture_start_time") is not None:
                    elapsed_time = time.time() - registration_process_state_fr["auto_capture_start_time"]
                    countdown_val = max(0, FR_CAPTURE_DELAY_SECONDS - int(elapsed_time))
                    current_auto_capture_countdown = countdown_val
                    registration_process_state_fr["auto_capture_countdown"] = countdown_val # Update state for SSE
                else:
                    registration_process_state_fr["auto_capture_countdown"] = None


            # --- Face Detection for guidance & auto-capture logic ---
            best_rect_dlib = None
            if dnn_detector_fr:
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                dnn_detector_fr.setInput(blob)
                detections = dnn_detector_fr.forward()
                best_conf = 0.0
                
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > FR_DNN_CONFIDENCE_THRESHOLD and confidence > best_conf:
                        best_conf = confidence
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        startX, startY = max(0, startX), max(0, startY)
                        endX, endY = min(w - 1, endX), min(h - 1, endY)
                        if startX < endX and startY < endY:
                            best_rect_dlib = dlib.rectangle(startX, startY, endX, endY)
                            cv2.rectangle(display_frame, (startX, startY), (endX, endY), (0, 255, 0), 2) # Green box for detected face
                            face_detected_this_frame = True
                            break # Use the first good detection for registration guidance

            with face_rec_state_lock:
                registration_process_state_fr["face_detected_for_auto_capture"] = face_detected_this_frame
                # This state is checked by the /upload_face_fr route for auto-capture logic

            # Display instructions and countdown on the frame
            cv2.putText(display_frame, current_instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            if current_auto_capture_countdown is not None:
                 cv2.putText(display_frame, f"Capturing in: {current_auto_capture_countdown}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            
            with face_rec_state_lock: # Check for mouth open pose specifically
                if registration_process_state_fr.get("status") == f"capturing_pose_{FR_POSES.index('mouth_open')}" and best_rect_dlib:
                    if is_mouth_open_fr(frame, best_rect_dlib):
                         cv2.putText(display_frame, "Mouth Open: OK!", (best_rect_dlib.left(), best_rect_dlib.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)
                    else:
                         cv2.putText(display_frame, "Open Your Mouth", (best_rect_dlib.left(), best_rect_dlib.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)


            ret_enc, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret_enc:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        logging.info(f"FR: {user_id_cam} stream loop ended.")

    except Exception as e:
        logging.error(f"FR: Exception in registration video generator ({user_id_cam}): {e}", exc_info=True)
        with face_rec_state_lock:
            registration_process_state_fr['status'] = "error_stream"
            registration_process_state_fr['instruction'] = "Video stream error."
    finally:
        logging.info(f"FR: Releasing camera for {user_id_cam} from registration generator.")
        camera_service.release(user_id_cam)
        # Reset parts of registration state if camera stops unexpectedly
        with face_rec_state_lock:
            if registration_process_state_fr['status'] not in ["completed", "error_final", "idle"]:
                 # registration_process_state_fr['status'] = "error_camera_stopped" # Or similar
                 registration_process_state_fr['instruction'] = "Camera stopped. Please restart registration."
                 # Consider full reset of registration_process_state_fr here or in a dedicated reset function


def generate_private_data_feed_fr():
    """Generates a simple video feed for the private data page, could include face detection."""
    user_id_cam = "profile_viewer_stream"
    desired_width, desired_height = 640, 480

    logging.info(f"FR: Attempting to acquire camera for {user_id_cam}")
    cap = camera_service.acquire(user_id_cam, desired_width, desired_height)

    if not cap:
        logging.error(f"FR: Could not acquire camera for {user_id_cam}. Private data feed will not start.")
        camera_service.release(user_id_cam)
        return

    logging.info(f"FR: Camera acquired for {user_id_cam}. Starting private data video stream.")
    try:
        while camera_service.is_active_by(user_id_cam):
            cap_instance = camera_service.get_capture_for_user(user_id_cam)
            if not cap_instance:
                time.sleep(0.1)
                continue
            
            ret, frame = cap_instance.read()
            if not ret:
                time.sleep(0.05)
                continue
            
            frame = cv2.flip(frame, 1) # Flip horizontally
            display_frame = frame.copy()
            (h, w) = display_frame.shape[:2]

            # Optional: Add face detection here just to draw a box, not for recognition
            if dnn_detector_fr:
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                dnn_detector_fr.setInput(blob)
                detections = dnn_detector_fr.forward()
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > FR_DNN_CONFIDENCE_THRESHOLD:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        cv2.rectangle(display_frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                        break # Show first detected face

            ret_enc, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret_enc:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        logging.info(f"FR: {user_id_cam} stream loop ended.")
    except Exception as e:
        logging.error(f"FR: Exception in private data video generator ({user_id_cam}): {e}", exc_info=True)
    finally:
        logging.info(f"FR: Releasing camera for {user_id_cam} from private data generator.")
        camera_service.release(user_id_cam)


# --- Existing Video Generators (Fall, Clap, ASL) ---
# ... existing code ...
# --- Flask Routes ---

# --- Face Recognition (FR) Routes --- # ADDED_FR

@app.route('/fr_home') # Renamed from '/' in face_recong_app
def home_fr():
    # This page might just redirect to login_fr or show some info
    # For now, it assumes a template 'fr_home.html' or similar might exist
    # Or, more practically, it could be the main entry point to the FR features.
    # If not used, can be removed. For now, let's assume it might be a menu page for FR.
    # return render_template('fr_landing.html') # Requires a new template
    return redirect(url_for('login_fr_page'))

@app.route('/', methods=['GET', 'POST'])
@app.route('/login_fr', methods=['GET', 'POST'])
def login_fr_page():
    """Login page for face recognition system."""
    if 'fr_username' in session: # Check if user is already logged in via FR
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username')
        pin = request.form.get('pin')
        
        # Face recognition status is checked via SSE, here we validate PIN after face is "recognized"
        # Or, if face rec is optional/backup, validate username/PIN directly
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.hashed_pin, pin):
            # Additional check: was face recently recognized for this user?
            # This logic might be more complex depending on UX flow.
            # For now, basic username/PIN check after potential face rec.
            with face_rec_state_lock:
                is_face_recognized_for_user = (login_recognition_state_fr.get("status") == "recognized" and
                                               login_recognition_state_fr.get("username") == username)
            
            # If requiring face recognition as part of login:
            # if not is_face_recognized_for_user:
            #     flash("Face not recognized for this user, or recognition timed out. Please try again.", "danger")
            #     return redirect(url_for('login_fr_page'))

            session['fr_username'] = user.username # Set Flask session
            # session['fr_user_id'] = user.id # Optionally store user ID
            flash(f'Welcome back, {user.username}!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or PIN. Please try again.', 'danger')
            # Reset face recognition state on failed PIN attempt to force re-recognition for security
            with face_rec_state_lock:
                login_recognition_state_fr.update({"status": "pin_failed", "username": None, "timestamp": time.time()})

    return render_template('login.html') # Assumes login.html is correctly pathed


@app.route('/check_user_exists_fr', methods=['POST'])
def check_user_exists_fr():
    data = request.get_json()
    username = data.get('username')
    if not username:
        return jsonify({'exists': False, 'error': 'Username not provided'}), 400
    
    user = User.query.filter_by(username=username).first()
    if user:
        return jsonify({'exists': True})
    else:
        return jsonify({'exists': False})
    

@app.route('/release_login_camera_fr', methods=['POST'])
def release_login_camera_fr_endpoint():
    global camera_service # Ensure camera_service is accessible
    user_id_to_release = "face_login_stream" # The user_id used by generate_face_login_feed_fr

    logging.info(f"FR: Received request to release camera for '{user_id_to_release}'.")
    if camera_service.is_active_by(user_id_to_release):
        if camera_service.release(user_id_to_release):
            logging.info(f"FR: Camera for '{user_id_to_release}' released successfully via endpoint.")
            return jsonify({"status": "success", "message": f"Camera for {user_id_to_release} released."}), 200
        else:
            logging.warning(f"FR: Failed to release camera for '{user_id_to_release}' via endpoint, though it was reported active.")
            return jsonify({"status": "error", "message": f"Failed to release camera for {user_id_to_release}."}), 500
    elif camera_service.is_active():
        active_user = camera_service.get_active_user()
        logging.info(f"FR: Camera for '{user_id_to_release}' was not active. Current active user: '{active_user}'.")
        return jsonify({"status": "info", "message": f"Camera for {user_id_to_release} was not active. Camera held by '{active_user}'."}), 200
    else:
        logging.info(f"FR: Camera for '{user_id_to_release}' was not active, and no other camera user active.")
        return jsonify({"status": "info", "message": f"Camera for {user_id_to_release} was not active."}), 200

@app.route('/register_fr', methods=['GET', 'POST'])
def register_fr_page():
    """User registration page (username and PIN)."""
    global temp_registration_data
    if request.method == 'POST':
        print("DEBUG: === ENTERED /register_fr POST handler ===")
        username = request.form.get('username')
        pin = request.form.get('pin')
        confirm_pin = request.form.get('confirmPin')
        role = request.form.get('role') # ADDED
        phone = request.form.get('phone') # ADDED
        birthday = request.form.get('birthday')
        print(f"DEBUG: Form data received: username='{username}', pin='{pin}', confirm_pin='{confirm_pin}', role='{role}', phone='{phone}', birthday='{birthday}'")

        # Modify CHECK A to include these if they are truly required for this step
        # For now, the original check only looks at username, pin, confirm_pin
        if not username or not pin or not confirm_pin or not role or not phone or not birthday: # UPDATED CHECK A
            print("DEBUG: Failing at CHECK A - missing fields")
            flash('All fields are required (username, role, pin, confirm_pin, phone, birthday).', 'danger') # Updated message
            return render_template('register.html')

        if not re.match("^[a-zA-Z0-9_]{3,20}$", username): # <<< CHECK B
            print("DEBUG: Failing at CHECK B - invalid username format")
            flash("Username must be 3-20 characters, alphanumeric and underscores only.", "danger")
            return render_template('register.html', username=username)

        if len(pin) != 6: # <<< CHECK C (We fixed this from < 4 to != 6)
            print("DEBUG: Failing at CHECK C - invalid PIN length")
            flash('PIN must be exactly 6 digits.', 'danger')
            return render_template('register.html', username=username)

        if pin != confirm_pin: # <<< CHECK D
            print("DEBUG: Failing at CHECK D - PINs do not match")
            flash('PINs do not match.', 'danger')
            return render_template('register.html', username=username)
        
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            print(f"DEBUG: Failing at CHECK E - Username '{username}' already exists.") # <<< PRINT BEFORE RETURN
            flash('Username already exists. Please choose another.', 'warning')
            return render_template('register.html', username=username)

        #if User.query.filter_by(username=username).first(): # <<< CHECK E
        #    print("DEBUG: Failing at CHECK E - username already exists")
        #    flash('Username already exists. Please choose another.', 'warning')
        #    return render_template('register.html', username=username)
        
        # If all checks pass, it should proceed here:
        with face_rec_state_lock: 
             temp_registration_data['username'] = username
             temp_registration_data['hashed_pin'] = generate_password_hash(pin)
             # STORE THE ADDITIONAL FIELDS
             temp_registration_data['role'] = role
             temp_registration_data['phone_number'] = phone
             temp_registration_data['birthday'] = birthday
             # IMPORTANT: Reset the state for a new registration attempt
             registration_process_state_fr.update({
                "status": "awaiting_face_registration", # This is the key starting status
                "current_pose_index": 0,
                "total_images_captured": 0, # This is a UI helper, server relies on len(embeddings)
                "instruction": "Please click 'Start Camera' to begin face registration.", # <<-- CORRECTED INSTRUCTION
                "error_message": None,
                "face_detected_for_auto_capture": False,
                "auto_capture_countdown": None,
                "auto_capture_start_time": None,
                "captured_embeddings_for_user": [] # Clear any previous embeddings for this session
            })

        # logging.debug(f"DEBUG: In /register_fr POST, temp_registration_data BEFORE redirect: {temp_registration_data}")
        # logging.debug(f"DEBUG: In /register_fr POST, registration_process_state_fr: {registration_process_state_fr}")
        flash('User details set. Now proceed to face registration by starting the camera.', 'info')
        return redirect(url_for('face_registration_fr_page')) # <<< THIS IS THE GOAL
    print("DEBUG: Processing GET request for /register_fr")
    return render_template('register.html')

@app.route('/face_registration_fr', methods=['GET'])
def face_registration_fr_page():
    # logging.debug(f"DEBUG: GET /face_registration_fr. Temp data: {temp_registration_data}")
    # logging.debug(f"DEBUG: GET /face_registration_fr. Reg state: {registration_process_state_fr}")
    with face_rec_state_lock: 
        if 'username' not in temp_registration_data or 'hashed_pin' not in temp_registration_data:
            # logging.debug("DEBUG: temp_registration_data missing, redirecting to register_fr.")
            flash("Please complete username and PIN registration first.", "warning")
            return redirect(url_for('register_fr_page'))
        
        try:
            poses_json_str = json.dumps(FR_POSES)
        except Exception as e:
            logging.error(f"Error converting FR_POSES to JSON: {e}")
            poses_json_str = "[]"

        max_images_for_template = len(FR_POSES)

        # If refreshing and state was 'awaiting_face_registration', ensure instruction is accurate.
        # The SSE should quickly update this anyway based on current server state.
        if registration_process_state_fr.get("status") == "awaiting_face_registration" and \
           registration_process_state_fr.get("instruction") != "Please click 'Start Camera' to begin face registration.":
            registration_process_state_fr["instruction"] = "Please click 'Start Camera' to begin face registration."

    return render_template(
        'face_registration.html', 
        username=temp_registration_data.get('username', 'User'),
        poses_json=poses_json_str,
        max_images=max_images_for_template,
        capture_delay_ms=FR_CAPTURE_DELAY_SECONDS * 1000
    )


@app.route('/upload_face_fr', methods=['POST'])
def upload_face_fr():
    global registration_process_state_fr, temp_registration_data
    logging.info(f"FR_DEBUG: ENTERED /upload_face_fr. Content-Type: {request.content_type}")
    logging.info(f"FR_DEBUG: Current registration_process_state_fr['status'] BEFORE processing: {registration_process_state_fr.get('status')}")

    with face_rec_state_lock: # Ensure exclusive access to shared state
        if 'username' not in temp_registration_data:
            return jsonify({'status': 'error', 'message': 'Registration session not found.'}), 400

        current_server_pose_idx_from_state = registration_process_state_fr.get("current_pose_index", 0)
        # This check should be based on number of embeddings captured vs. total poses
        if len(registration_process_state_fr.get("captured_embeddings_for_user", [])) >= len(FR_POSES):
             # If all poses' images are captured based on embeddings list length
             if registration_process_state_fr.get("status") != "all_poses_captured_pending_save":
                registration_process_state_fr["status"] = "all_poses_captured_pending_save"
                registration_process_state_fr["instruction"] = "All face poses captured! Click 'Submit Registration'."
             # Do not return error if client is just sending detection status for a "completed" state
             # return jsonify({'status': 'info', 'message': 'All poses already captured server-side.'}), 200


        image_bgr = None # For actual image capture
        is_status_update = False # Flag to differentiate payload types

        if request.content_type == 'application/json':
            is_status_update = True
            data = request.get_json()
            if 'detected' not in data: # Invalid JSON payload for status update
                return jsonify({'status': 'error', 'message': 'Unsupported JSON payload for /upload_face_fr status update.'}), 400
            
            client_detected = data.get('detected', False)
            client_sent_pose_idx = data.get('pose_index', 0) # Client's idea of current pose

            # ADD DETAILED LOGGING HERE
            logging.info(f"FR (upload_face_fr JSON): Received detected={client_detected}, client_pose_idx={client_sent_pose_idx}. "
                         f"Current server status: '{registration_process_state_fr.get('status')}', "
                         f"server_pose_idx: {registration_process_state_fr.get('current_pose_index')}, "
                         f"embeddings: {len(registration_process_state_fr.get('captured_embeddings_for_user',[]))}")

            # --- State Transition and Auto-Capture Logic for Status Updates ---

            # 1. Transition from 'awaiting_face_registration' to the first pose
            if registration_process_state_fr.get("status") == "awaiting_face_registration":
                logging.info("FR (upload_face_fr JSON): Server status is 'awaiting_face_registration'.")
                if client_detected: # This will be true once camera starts & client posts
                    server_pose_to_start = registration_process_state_fr.get('current_pose_index', 0) # Should be 0
                    if server_pose_to_start >= len(FR_POSES): # Should not happen if logic is correct
                        logging.error("FR Error: Attempting to start pose beyond available poses.")
                        registration_process_state_fr["status"] = "error"
                        registration_process_state_fr["instruction"] = "System error. Please restart."
                        return jsonify({'status': 'error', 'message': 'System error.'}), 500

                    logging.info(f"FR (upload_face_fr JSON): Client detected face for 'awaiting_face_registration'. Transitioning to pose {server_pose_to_start}.")
                    registration_process_state_fr["status"] = f"capturing_pose_{server_pose_to_start}"
                    registration_process_state_fr["instruction"] = f"Please look {FR_POSES[server_pose_to_start % len(FR_POSES)].upper()} and hold."
                    registration_process_state_fr["error_message"] = None
                    registration_process_state_fr["face_detected_for_auto_capture"] = True # Client says face is there
                    registration_process_state_fr["auto_capture_start_time"] = time.time() # Start countdown
                    registration_process_state_fr["auto_capture_countdown"] = FR_CAPTURE_DELAY_SECONDS
                else:
                    # This case is less likely if client's isClientFaceDetected is basic,
                    # but good for robustness if client-side detection becomes real.
                    logging.info("FR (upload_face_fr JSON): Client reports no face for 'awaiting_face_registration'. Waiting.")
                    registration_process_state_fr["instruction"] = "Position your face in the camera view to start."
                    registration_process_state_fr["face_detected_for_auto_capture"] = False
                    registration_process_state_fr["auto_capture_start_time"] = None
                
                return jsonify({'status': 'success', 'message': 'Initial face detection status processed by server.'}), 200
            
            # 2. Handle ongoing auto-capture logic if already in a capturing state
            server_current_pose_idx = registration_process_state_fr.get("current_pose_index", 0)
            
            # Ensure we are in a valid capturing state and client is on the same pose
            if registration_process_state_fr.get("status", "").startswith("capturing_pose_") and \
               client_sent_pose_idx == server_current_pose_idx:

                registration_process_state_fr["face_detected_for_auto_capture"] = client_detected
                
                if client_detected:
                    if registration_process_state_fr.get("auto_capture_start_time") is None:
                        registration_process_state_fr["auto_capture_start_time"] = time.time()
                    registration_process_state_fr["error_message"] = None # Clear "face lost" type errors
                else: # Face lost during countdown
                    registration_process_state_fr["auto_capture_start_time"] = None
                    registration_process_state_fr["auto_capture_countdown"] = None
                    registration_process_state_fr["instruction"] = f"Face lost. Re-align for {FR_POSES[server_current_pose_idx % len(FR_POSES)].upper()}."
                    # Error message might also be set here by SSE or client side

                # Check if auto-capture should be triggered NOW
                should_trigger_capture_action = False
                if registration_process_state_fr.get("face_detected_for_auto_capture") and \
                   registration_process_state_fr.get("auto_capture_start_time") is not None:
                    elapsed = time.time() - registration_process_state_fr["auto_capture_start_time"]
                    if elapsed >= FR_CAPTURE_DELAY_SECONDS:
                        should_trigger_capture_action = True
                        # Timer will be fully reset after successful performCapture or if face is lost
                        # For now, we are instructing to capture, client will take picture and send it.
                        # The performCapture on client will reset its own timers.
                        # Server timers will be reset upon receiving the image.
                
                if should_trigger_capture_action:
                    logging.info(f"FR: Server instructing client to trigger capture for pose index {server_current_pose_idx}.")
                    # Update instruction for the brief moment client is taking picture
                    registration_process_state_fr["instruction"] = f"Capturing for {FR_POSES[server_current_pose_idx % len(FR_POSES)].upper()}..."
                    return jsonify({
                        'status': 'success',
                        'message': 'Conditions met, server instructing client to trigger capture.',
                        'action': 'trigger_capture',
                        'pose_name': FR_POSES[server_current_pose_idx % len(FR_POSES)]
                    }), 200
                else:
                    # Just acknowledge status, SSE will provide countdown or other instructions
                    return jsonify({'status': 'success', 'message': 'Face detection status acknowledged for current pose.'}), 200
            else: # Client pose index mismatch, or server not in an active capturing state for this JSON update
                logging.warning(f"FR: Status update for mismatched pose or wrong server state. Client: {client_sent_pose_idx}, Server: {server_current_pose_idx}, Status: {registration_process_state_fr.get('status')}")
                return jsonify({'status': 'info', 'message': 'State mismatch or server not ready for this pose update.'}), 200

        elif request.content_type.startswith('image/') or \
             request.content_type == 'application/octet-stream' or \
             ('data:image' in request.data.decode('utf-8', errors='ignore')[:30]):
            try:
                image_data_url = request.data.decode('utf-8')
                image_bgr = decode_image_fr(image_data_url)
                if image_bgr is None: raise ValueError("Image decoding failed")
                logging.info(f"FR: Received image data for pose capture. Pose from header: {request.headers.get('X-Current-Pose-Name')}")
            except Exception as e:
                logging.error(f"FR: Error decoding direct image data: {e}")
                registration_process_state_fr['error_message'] = 'Could not decode image data.'
                registration_process_state_fr['instruction'] = 'Error: Image data corrupt. Please try again.'
                return jsonify({'status': 'error', 'message': registration_process_state_fr['error_message']}), 400 # Bad request
        else:
            return jsonify({'status': 'error', 'message': 'Unsupported content type or no valid data.'}), 415

        # --- Actual Image Processing for Capture (if image_bgr is populated) ---
        if image_bgr is None: # Should have been handled by status update logic if no image was sent
            return jsonify({'status': 'error', 'message': 'Internal error: Expected image data but none found after status checks.'}), 500

        logging.info(f"FR: Processing image for embedding. Server current pose index: {current_server_pose_idx_from_state}")
        (h, w) = image_bgr.shape[:2]
        best_rect_dlib = None
        if dnn_detector_fr:
            blob = cv2.dnn.blobFromImage(cv2.resize(image_bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            dnn_detector_fr.setInput(blob)
            detections = dnn_detector_fr.forward()
            best_conf_img = 0.0
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > FR_DNN_CONFIDENCE_THRESHOLD and confidence > best_conf_img:
                    best_conf_img = confidence
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    best_rect_dlib = dlib.rectangle(max(0, startX), max(0, startY), min(w - 1, endX), min(h - 1, endY))
                    # No break here, find the best confidence detection
        
        if not best_rect_dlib:
            registration_process_state_fr['error_message'] = "No face detected in the submitted image. Please try again."
            registration_process_state_fr['instruction'] = registration_process_state_fr['error_message']
             # Reset auto-capture timers as this capture attempt failed.
            registration_process_state_fr["face_detected_for_auto_capture"] = False
            registration_process_state_fr["auto_capture_start_time"] = None
            registration_process_state_fr["auto_capture_countdown"] = None
            return jsonify({'status': 'error', 'message': registration_process_state_fr['error_message']}), 200 # 200 OK for client to update UI

        current_pose_name_server = FR_POSES[current_server_pose_idx_from_state % len(FR_POSES)]
        if current_pose_name_server == "mouth_open" and not is_mouth_open_fr(image_bgr, best_rect_dlib):
            registration_process_state_fr['error_message'] = "Mouth not detected as open. Please ensure your mouth is clearly open."
            registration_process_state_fr['instruction'] = registration_process_state_fr['error_message']
            registration_process_state_fr["face_detected_for_auto_capture"] = False # Reset timers
            registration_process_state_fr["auto_capture_start_time"] = None
            registration_process_state_fr["auto_capture_countdown"] = None
            return jsonify({'status': 'error', 'message': registration_process_state_fr['error_message'], 'pose_check_failed': 'mouth_open'}), 200

        embedding = compute_face_embedding_fr(image_bgr, best_rect_dlib)
        if embedding is None:
            registration_process_state_fr['error_message'] = "Could not compute face embedding. Try a clearer image."
            registration_process_state_fr['instruction'] = registration_process_state_fr['error_message']
            registration_process_state_fr["face_detected_for_auto_capture"] = False # Reset timers
            registration_process_state_fr["auto_capture_start_time"] = None
            registration_process_state_fr["auto_capture_countdown"] = None
            return jsonify({'status': 'error', 'message': registration_process_state_fr['error_message']}), 200

        registration_process_state_fr["captured_embeddings_for_user"].append(embedding.tolist())
        
        logging.info(f"FR: Captured embedding for pose {current_pose_name_server} for user {temp_registration_data['username']}. Total embeddings: {len(registration_process_state_fr['captured_embeddings_for_user'])}")
        
        # Advance to next pose IF more poses are left
        if len(registration_process_state_fr["captured_embeddings_for_user"]) < len(FR_POSES):
            registration_process_state_fr['current_pose_index'] += 1
            new_server_pose_idx = registration_process_state_fr['current_pose_index']
            registration_process_state_fr['status'] = f"capturing_pose_{new_server_pose_idx}"
            registration_process_state_fr['instruction'] = f"Great! Now, please look {FR_POSES[new_server_pose_idx % len(FR_POSES)].upper()} and hold."
        else: # All poses captured
            registration_process_state_fr['status'] = "all_poses_captured_pending_save"
            registration_process_state_fr['instruction'] = "All face poses captured! Click 'Submit Registration'."
            # current_pose_index remains at the last valid index or len(FR_POSES)

        # Reset flags for the next auto-capture cycle (or for completion)
        registration_process_state_fr['error_message'] = None 
        registration_process_state_fr["face_detected_for_auto_capture"] = False
        registration_process_state_fr["auto_capture_start_time"] = None
        registration_process_state_fr["auto_capture_countdown"] = None

        return jsonify({
            'status': 'success',
            'message': f"Pose {current_pose_name_server} captured and processed.",
            'images_captured': len(registration_process_state_fr["captured_embeddings_for_user"]),
            'server_status': registration_process_state_fr['status']
            # 'next_pose_index_server' is implicitly handled by server_status and instruction
        })

@app.route('/complete_registration_fr', methods=['POST'])
def complete_registration_fr():
    """Finalizes registration: calculates average embedding, saves user to DB."""
    global temp_registration_data, registration_process_state_fr, known_face_data_fr

    with face_rec_state_lock: # Ensure exclusive access
        if 'username' not in temp_registration_data or not registration_process_state_fr["captured_embeddings_for_user"]:
            flash("No registration data or face embeddings found. Please start over.", "error")
            # Fully reset registration state
            registration_process_state_fr = { "status": "idle", "current_pose_index": 0, "total_images_captured": 0,
                                            "instruction": "Please start camera.", "error_message": None, 
                                            "face_detected_for_auto_capture": False, "auto_capture_countdown": None, 
                                            "auto_capture_start_time": None, "captured_embeddings_for_user": []}
            temp_registration_data.clear()
            return redirect(url_for('register_fr_page'))

        reg_data = temp_registration_data # Use the whole dictionary for cleaner code
        
        # Calculate average embedding
        embeddings_list = registration_process_state_fr["captured_embeddings_for_user"]
        if not embeddings_list:
            flash("No embeddings were captured during registration.", "danger")
            registration_process_state_fr['status'] = "error_final"
            registration_process_state_fr['error_message'] = "No embeddings captured."
            return redirect(url_for('face_registration_fr_page'))

        try:
            avg_embedding_np = np.mean(np.array(embeddings_list), axis=0)
            avg_embedding_list = avg_embedding_np.tolist()
        except Exception as e:
            logging.error(f"FR: Error averaging embeddings for {reg_data['username']}: {e}", exc_info=True)
            flash("Error processing face data. Please try registration again.", "danger")
            registration_process_state_fr['status'] = "error_final"
            registration_process_state_fr['error_message'] = "Could not process embeddings."
            return redirect(url_for('face_registration_fr_page'))

        # Save user to database
        new_user = User(
            username=reg_data['username'], 
            hashed_pin=reg_data['hashed_pin'], 
            role=reg_data.get('role'),
            phone_number=reg_data.get('phone_number'),
            birthday=reg_data.get('birthday'),
            embedding=json.dumps(avg_embedding_list)
        )
        try:
            db.session.add(new_user)
            db.session.commit()
            logging.info(f"FR: User {reg_data['username']} registered successfully with averaged embedding.")

            # Reload embeddings into memory cache
            load_embeddings_from_db_fr(force_reload=True)
            
            flash(f"User {reg_data['username']} registered successfully! You can now log in.", "success")
            
            # Clear temporary registration data and reset state
            temp_registration_data.clear()
            registration_process_state_fr = { "status": "idle", "current_pose_index": 0, "total_images_captured": 0,
                                            "instruction": "Please start camera.", "error_message": None, 
                                            "face_detected_for_auto_capture": False, "auto_capture_countdown": None, 
                                            "auto_capture_start_time": None, "captured_embeddings_for_user": []}
            
            return redirect(url_for('login_fr_page'))

        except IntegrityError: # Should have been caught by check_user_exists, but as a safeguard
            db.session.rollback()
            flash("Username already exists. This should not happen at this stage.", "danger")
            logging.warning(f"FR: IntegrityError during final registration for {reg_data['username']}, already exists.")
        except Exception as e:
            db.session.rollback()
            logging.error(f"FR: Error saving user {reg_data['username']} to database: {e}", exc_info=True)
            flash("An error occurred while saving your registration. Please try again.", "danger")
        
        registration_process_state_fr['status'] = "error_final"
        registration_process_state_fr['error_message'] = "Database error during registration."
        return redirect(url_for('face_registration_fr_page'))


@app.route('/private_data_fr')
def private_data_fr_page():
    """Protected page, requires FR login."""
    if 'fr_username' not in session:
        flash('Please log in to access this page.', 'warning')
        return redirect(url_for('login_fr_page'))

    username = session['fr_username']
    user = User.query.filter_by(username=username).first() # Query the database

    if not user:
        flash('User data not found. Please log in again.', 'danger')
        session.pop('fr_username', None)
        return redirect(url_for('login_fr_page'))

    # Create the dictionary to pass to the template
    user_data_to_pass = {
        'phone_number': user.phone_number,
        'birthday': user.birthday,
        'role': user.role
        # Add any other fields from the User model you want to display
    }

    return render_template('private_data.html', username=username, user_data=user_data_to_pass)


@app.route('/logout_fr')
def logout_fr():
    """Logs out the face recognition user."""
    session.pop('fr_username', None)
    # session.pop('fr_user_id', None) # If user_id was stored
    with face_rec_state_lock: # Reset login state on logout
         login_recognition_state_fr.update({"status": "logged_out", "username": None, "timestamp": time.time()})
    flash('You have been logged out from the FR system.', 'info')
    # Release camera if it was used by profile viewer, though it should auto-release
    camera_service.release("profile_viewer_stream") 
    return redirect(url_for('login_fr_page'))


# --- FR Video Feed Routes ---
@app.route('/login_video_feed_fr')
def login_video_feed_fr_endpoint():
    """Video feed for the FR login page."""
    return Response(stream_with_context(generate_face_login_feed_fr()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/face_registration_video_feed_fr')
def face_registration_video_feed_fr_endpoint():
    """Video feed for the FR registration page."""
    return Response(stream_with_context(generate_face_registration_feed_fr()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/private_data_video_feed_fr')
def private_data_video_feed_fr_endpoint():
    """Video feed for the private data page (e.g., showing user's camera)."""
    if 'fr_username' not in session: # Protect this feed
        return "Not authorized", 401
    return Response(stream_with_context(generate_private_data_feed_fr()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# --- FR Status Feed Routes (Server-Sent Events) ---
@app.route('/login_status_feed_fr')
def login_status_feed_fr_endpoint():
    """SSE feed for real-time login recognition status."""
    def event_stream():
        last_sent_status = None
        last_sent_username = None
        while True:
            with face_rec_state_lock: # Use the new dedicated lock
                current_status = login_recognition_state_fr.get("status")
                current_username = login_recognition_state_fr.get("username")
                timestamp = login_recognition_state_fr.get("timestamp")
            
            # Send update only if status or username changed
            if current_status != last_sent_status or current_username != last_sent_username:
                data_to_send = {
                    "status": current_status,
                    "username": current_username,
                    "timestamp": timestamp
                }
                # logging.debug(f"SSE Login Status: {data_to_send}")
                yield f"data: {json.dumps(data_to_send)}\n\n"
                last_sent_status = current_status
                last_sent_username = current_username
            
            time.sleep(0.2) # Polling interval for updates to send via SSE
    return Response(stream_with_context(event_stream()), mimetype='text/event-stream')


@app.route('/face_registration_status_feed_fr')
def registration_status_feed_fr_endpoint():
    """SSE feed for real-time registration process status."""
    def event_stream():
        last_state_snapshot = {}
        initial_sent = False 

        while True:
            current_state_for_sse = {} # Renamed to avoid confusion with other current_state vars
            with face_rec_state_lock: # Use the new dedicated lock
                # Create a snapshot of the parts of registration_process_state_fr you want to send
                # Ensure all keys have default values if they might be missing from registration_process_state_fr
                current_pose_idx_val = registration_process_state_fr.get("current_pose_index", 0)
                captured_embeddings = registration_process_state_fr.get("captured_embeddings_for_user", [])

                current_state_for_sse = {
                    "status": registration_process_state_fr.get("status", "idle"), # Default to 'idle'
                    "current_pose_index": current_pose_idx_val,
                    "current_pose_name": FR_POSES[current_pose_idx_val] if current_pose_idx_val < len(FR_POSES) else "Done",
                    "instruction": registration_process_state_fr.get("instruction", "Please start the camera."), # Initial instruction
                    "error_message": registration_process_state_fr.get("error_message", None),
                    "total_images_captured_for_user": len(captured_embeddings),
                    "max_poses": len(FR_POSES), 
                    "face_detected_for_auto_capture": registration_process_state_fr.get("face_detected_for_auto_capture", False),
                    "auto_capture_countdown": registration_process_state_fr.get("auto_capture_countdown", None)
                }

                current_status_val = registration_process_state_fr.get("status", "idle")
                auto_capture_start_time_val = registration_process_state_fr.get("auto_capture_start_time")
                is_face_detected_val = registration_process_state_fr.get("face_detected_for_auto_capture", False)

                # Update total_images_captured_for_user based on the length of the embeddings list
                current_state_for_sse["total_images_captured_for_user"] = len(registration_process_state_fr.get("captured_embeddings_for_user", []))


                if current_status_val.startswith("capturing_pose_") and \
                   is_face_detected_val and \
                   auto_capture_start_time_val is not None:
                    elapsed_time = time.time() - auto_capture_start_time_val
                    countdown_val = max(0, FR_CAPTURE_DELAY_SECONDS - int(elapsed_time))
                    current_state_for_sse["auto_capture_countdown"] = countdown_val
                else:
                    current_state_for_sse["auto_capture_countdown"] = None # Explicitly null if not counting down

                # Ensure instruction is current
                current_state_for_sse["instruction"] = registration_process_state_fr.get("instruction", "Follow instructions.")
                current_state_for_sse["error_message"] = registration_process_state_fr.get("error_message", None)
            
            if not initial_sent or current_state_for_sse != last_state_snapshot:
                # print(f"SSE Reg Status Sending: {current_state_for_sse}") # Server debug
                try:
                    json_payload = json.dumps(current_state_for_sse)
                    yield f"data: {json_payload}\n\n"
                    last_state_snapshot = current_state_for_sse.copy() # Use .copy() for dicts
                    initial_sent = True 
                except TypeError as te:
                    print(f"SSE TypeError during json.dumps: {te}. State was: {current_state_for_sse}")
                    # yield f"event: error\ndata: {json.dumps({'error': 'Server serialization error')}\n\n" # Optional: send error event
            
            time.sleep(0.2) # Check for state changes every 0.2 seconds
    return Response(stream_with_context(event_stream()), mimetype='text/event-stream')


# --- Existing Routes (Index, Fall, Clap, ASL, Upload etc.) ---
# Keep all existing routes from app.py as they were.
# Example:
# @app.route('/')
# def index():
# ... and so on for all other routes ...

# --- Make sure initialize_database includes the new User model ---
# db.create_all() will handle all models registered with the SQLAlchemy instance 'db'
# So no explicit change needed there, but ensure it's called.

# --- Existing App Initialization (Main block) ---
# Modify the main block to include FR model loading.

# ... (ensure all your other routes from the original app.py are here) ...
# Example of where they would be:
# @app.route('/')
# def index():
#     # ... existing index logic ...
#     # You might want to add links to the FR system from the main index page
#     return render_template('index.html', fr_active=True) # Pass a flag if FR features are enabled

# ... (other existing routes like /start_webcam, /stop_webcam, /video_feed for fall detection etc.)

# --- Ensure initialize_database and load_preferences are called correctly ---
def initialize_app(current_app_instance):
    with current_app_instance.app_context():
        logging.info("Initializing application...")
        
        # Initialize database and create tables if they don't exist
        try:
            # Ensure folders exist
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            os.makedirs(PROCESSED_FOLDER, exist_ok=True)
            
            initialize_database() # This should create User, ClapSession, FallEvent tables
            logging.info("Database initialized (tables checked/created).")
        except Exception as e:
            logging.error(f"Error during database initialization: {e}", exc_info=True)
            # Depending on severity, might want to exit or alert.

        # Load preferences
        global preferences
        preferences = load_preferences() # Assuming load_preferences uses current_app
        logging.info(f"Preferences loaded: {preferences}")

        # Load Face Recognition Models
        if not load_all_models_fr():
            logging.error("CRITICAL: Failed to load one or more Face Recognition models. FR features may not work.")
            # You might want to disable FR routes or show a warning if loading fails
        else:
            logging.info("Face Recognition models loaded successfully.")
            # Load known face embeddings from DB after models are ready
            if not load_embeddings_from_db_fr():
                logging.warning("FR: Could not load embeddings from database on startup.")
            else:
                logging.info(f"FR: Loaded {len(known_face_data_fr.get('embeddings',[]))} known face embeddings.")

        # Initialize ASL detector (if still part of the app)
        global asl_detector_instance, asl_labels # If these are global
        # asl_detector_instance, asl_labels = initialize_asl_detector() # Assuming this function exists and is correct
        # if asl_detector_instance:
        #     logging.info("ASL Detector initialized successfully.")
        # else:
        #     logging.warning("ASL Detector failed to initialize.")
        
        # Any other initializations
        logging.info("Application initialization complete.")


if __name__ == '__main__':
    # The initialize_app function should be called to set up everything
    # For development, Flask's dev server typically handles app context for the first request.
    # However, for model loading and DB setup before first request, explicit app_context is good.
    
    # To ensure initialization runs before the server starts handling requests:
    initialize_app(app)

    # Determine host IP dynamically for LAN access (optional, from original app.py)
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
    except socket.gaierror:
        local_ip = '127.0.0.1' # Fallback if hostname resolution fails
    logging.info(f"Flask app running on http://{local_ip}:5000 and http://127.0.0.1:5000")
    
    # Use socketio.run for SocketIO compatibility, debug=False for production/controlled logging
    # For development with auto-reload, use debug=True but be mindful of model reloading.
    # Set use_reloader=False if model loading is slow and causes issues with reloader.
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    # Alternatively, for non-SocketIO parts or simpler setup during merge:
    # app.run(host='0.0.0.0', port=5000, debug=True)

# --- END OF FILE app.py ---