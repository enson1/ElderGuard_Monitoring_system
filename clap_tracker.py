import cv2
import mediapipe as mp
import math
import time
import threading
import logging
import numpy as np

# Assuming CameraService is available in the context where this class is used.
# If not, it needs to be imported or passed differently.
# from app import CameraService # This would create a circular import if app.py imports ClapTracker

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClapTracker:
    """
    Handles the Buddha Clap tracking functionality, including camera capture (via CameraService),
    pose detection, clap counting, and frame generation in a separate thread.
    """
    def __init__(self, camera_service, target_claps=7, distance_threshold=0.12, calories_per_clap=0.2, desired_width=640, desired_height=480):
        self.camera_service = camera_service # Store CameraService instance
        self.target_claps = target_claps
        self.distance_threshold = distance_threshold
        self.calories_per_clap = calories_per_clap
        self.desired_width = desired_width
        self.desired_height = desired_height

        # _active flag will now reflect if WE successfully acquired the camera and are running.
        # The definitive source of camera state is camera_service.is_active_by('clap_tracker')
        self._active = False 
        self._thread = None
        self._lock = threading.Lock() # For internal state like _output_frame, _clap_count
        self._output_frame = None
        self._cap = None # This will hold the VideoCapture object from CameraService
        self._pose = None
        self._fps = 15.0 # Default, will be updated from CameraService

        # Tracking state
        self._clap_count = 0
        self._calories_burned = 0.0
        self._hands_state = "APART"
        self._target_reached = False

    def is_active(self):
        """Check if the tracker has acquired the camera and its thread is running."""
        # This checks if our component believes it's active AND the camera service agrees.
        # The primary check for routes should be camera_service.is_active_by('clap_tracker')
        # This method can be used internally or for detailed status.
        with self._lock: # Protect self._active
            return self._active and self.camera_service.is_active_by('clap_tracker')

    def get_frame(self):
        """Get the latest processed frame for MJPEG streaming."""
        with self._lock:
            return self._output_frame

    def get_status(self):
        """Get the current tracking status (clap count, calories, etc.)."""
        is_actually_active_via_service = self.camera_service.is_active_by('clap_tracker')
        with self._lock: # Ensure consistent read of internal state
            return {
                "clap_count": self._clap_count,
                "calories_burned": self._calories_burned,
                "target_claps": self.target_claps,
                "target_reached": self._target_reached,
                 # reflects if thread is running and we *think* we have camera
                "active": self._active and is_actually_active_via_service
            }

    def set_target(self, new_target):
        """Safely update the target number of claps."""
        try:
            target_int = int(new_target)
            if target_int < 0:
                 logger.warning(f"Attempted to set invalid clap target: {target_int}. Using 0 instead.")
                 target_int = 0
            with self._lock:
                 self.target_claps = target_int
                 self._target_reached = (self.target_claps > 0 and self._clap_count >= self.target_claps)
                 logger.info(f"Clap target updated to: {self.target_claps}")
            return True, "Target updated"
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid value provided for clap target: {new_target}. Error: {e}")
            return False, "Invalid target value"

    def start(self) -> tuple[bool, str]:
        """
        Attempts to acquire the camera via CameraService and start the clap tracker background thread.
        Returns (success_boolean, message_string).
        """
        if self.is_active(): # Check our active state first
            logger.warning("Clap tracker start called, but it's already considered active.")
            return False, "Clap Tracker is already active."

        logger.info("Attempting to start Clap Tracker...")
        
        # Acquire camera using CameraService
        self._cap = self.camera_service.acquire('clap_tracker', self.desired_width, self.desired_height)
        
        if not self._cap:
            logger.error("Clap Tracker failed to acquire camera via CameraService.")
            return False, "Failed to acquire camera. It might be in use or unavailable."

        logger.info("Clap Tracker successfully acquired camera.")
        self._fps = self.camera_service.get_fps() # Get FPS after acquiring

        with self._lock:
            self._active = True # Set our internal active flag
            # Reset state when starting
            self._clap_count = 0
            self._calories_burned = 0.0
            self._hands_state = "APART"
            self._target_reached = False
            self._output_frame = None

        self._thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._thread.start()
        logger.info("Clap Tracker processing thread started.")
        return True, "Clap Tracker started successfully."

    def stop(self) -> tuple[bool, str]:
        """Stop the clap tracker background thread and release resources via CameraService."""
        if not self._active and not self.camera_service.is_active_by('clap_tracker'): # Check both
            logger.warning("Clap tracker stop called, but it's not considered active.")
            return False, "Clap Tracker is not active."

        logger.info("Stopping Clap Tracker...")
        
        release_message = "Camera released."
        release_success = True

        with self._lock:
            self._active = False # Signal thread to stop

        if self._thread is not None:
            logger.info("Joining Clap Tracker processing thread...")
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                 logger.warning("Clap tracker thread did not stop gracefully.")
            self._thread = None
        logger.info("Clap Tracker processing thread stopped.")

        # Release camera using CameraService
        if self.camera_service.is_active_by('clap_tracker'): # Check if we still hold it
            if not self.camera_service.release('clap_tracker'):
                logger.error("Clap Tracker failed to release camera via CameraService.")
                release_message = "Failed to release camera properly."
                release_success = False
            else:
                logger.info("Clap Tracker successfully released camera via CameraService.")
        else:
            logger.info("Clap Tracker: Camera was not held by 'clap_tracker' at stop time, or already released.")
        
        self._cap = None # Clear our reference to the capture object
        return release_success, f"Clap Tracker stopped. {release_message}"

    def _processing_loop(self):
        """The main loop running in the background thread. Uses self._cap from CameraService."""
        # FRAME_WIDTH, FRAME_HEIGHT are now self.desired_width, self.desired_height or from self.camera_service.get_dimensions()
        
        if not self._cap: # Should have been set by start()
            logger.error("Clap Tracker: _processing_loop started without a valid camera capture object.")
            with self._lock: self._active = False
            return

        try:
            logger.info(f"Initializing MediaPipe Pose for Clap Tracker (using CameraService camera).")
            mp_pose = mp.solutions.pose
            self._pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles

            logger.info(f"Clap Tracker processing loop started with FPS: {self._fps:.2f}")

            # The main loop condition relies on self._active, which is controlled by start/stop
            # and camera_service.is_active_by('clap_tracker') to ensure we should be running.
            while self._active and self.camera_service.is_active_by('clap_tracker'):
                start_frame_time = time.monotonic()

                if not self._cap or not self._cap.isOpened():
                     logger.error("Clap tracker camera (from CameraService) became unavailable or closed.")
                     with self._lock: self._active = False 
                     break

                ret, frame = self._cap.read()
                if not ret or frame is None:
                    # If camera service still says we are active, but read fails, could be a temp issue
                    if self.camera_service.is_active_by('clap_tracker'):
                        logger.warning("Clap Tracker: Failed to read frame, but still active. Retrying.")
                        time.sleep(0.05)
                        continue
                    else: # Camera service says we are no longer active
                        logger.info("Clap Tracker: Read failed and no longer active by CameraService. Exiting loop.")
                        with self._lock: self._active = False
                        break
                
                # --- Preprocessing ---
                frame = cv2.flip(frame, 1)
                img_h, img_w, _ = frame.shape # Get dimensions from actual frame
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False

                results = self._pose.process(image_rgb)
                frame.flags.writeable = True
                current_distance = float('inf')

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

                    if left_wrist.visibility > 0.5 and right_wrist.visibility > 0.5:
                        lx, ly = left_wrist.x, left_wrist.y
                        rx, ry = right_wrist.x, right_wrist.y
                        current_distance = math.sqrt((rx - lx)**2 + (ry - ly)**2)
                        
                        with self._lock: # Lock for updating clap count and related state
                            if current_distance < self.distance_threshold:
                                if self._hands_state == "APART":
                                    self._clap_count += 1
                                    self._calories_burned = self._clap_count * self.calories_per_clap
                                    self._hands_state = "TOGETHER"
                                    logger.debug(f"CLAP! Count: {self._clap_count}")
                                    if self.target_claps > 0 and self._clap_count >= self.target_claps:
                                        self._target_reached = True
                            else:
                                if self._hands_state == "TOGETHER":
                                    self._hands_state = "APART"
                    
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                with self._lock: # Safely read shared state for drawing
                    clap_c = self._clap_count
                    cal_b = self._calories_burned
                    tgt_r = self._target_reached
                    tgt_claps_for_drawing = self.target_claps
                
                # Draw info
                cv2.putText(frame, f"Claps: {clap_c}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
                target_text = f"Target: {tgt_claps_for_drawing}" if tgt_claps_for_drawing > 0 else "Target: N/A"
                cv2.putText(frame, target_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 150, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Calories: {cal_b:.1f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
                if tgt_r:
                    cv2.putText(frame, "TARGET REACHED!", (img_w // 2 - 200, img_h // 2), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, "Clap Tracker: ACTIVE", (10, img_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                ret_encode, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ret_encode:
                    with self._lock:
                        self._output_frame = buffer.tobytes()
                else:
                    logger.warning("Failed to encode clap tracker frame to JPEG.")

                elapsed_time = time.monotonic() - start_frame_time
                current_service_fps = self.camera_service.get_fps() # Get live FPS for rate control
                target_frame_duration = 1.0 / current_service_fps if current_service_fps > 0 else 1.0 / 15.0
                sleep_time = max(0, target_frame_duration - elapsed_time)
                time.sleep(sleep_time)

        except Exception as e:
            logger.error(f"Exception in clap tracker processing loop: {e}", exc_info=True)
        finally:
            logger.info("Cleaning up Clap Tracker resources from processing loop...")
            with self._lock:
                self._output_frame = None 
                self._active = False # Ensure active is false on exit

            if self._pose:
                self._pose.close()
                self._pose = None
                logger.info("Closed MediaPipe Pose resources for ClapTracker.")
            
            # Camera release is primarily handled by stop() method using CameraService.
            # self._cap is just our reference to the one from CameraService.
            # If CameraService still thinks we own it, stop() will release.
            # If it was taken by another user, release by us would fail anyway.
            logger.info("Clap Tracker processing loop finished.")

# Example Usage (if run standalone for testing, CameraService would need to be mocked or a dummy provided)
if __name__ == '__main__':
    # This standalone example won't work directly without a CameraService mock.
    # For testing, you'd typically mock the CameraService.
    
    class MockCameraService:
        def __init__(self, camera_index):
            self.camera_index = camera_index
            self.active_user = None
            self.mock_cap = None
            self.fps = 30.0
            self.width = 640
            self.height = 480
            logger.info("MockCameraService initialized.")

        def acquire(self, user_id, desired_width, desired_height):
            if self.active_user is None:
                self.active_user = user_id
                # Try to open a real camera if available for the mock
                self.mock_cap = cv2.VideoCapture(self.camera_index)
                if not self.mock_cap.isOpened():
                    logger.error(f"MockCameraService: Failed to open real camera index {self.camera_index}")
                    self.active_user = None # Failed acquisition
                    return None
                self.mock_cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
                self.mock_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
                self.width = int(self.mock_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(self.mock_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.fps = self.mock_cap.get(cv2.CAP_PROP_FPS) or 30.0
                logger.info(f"MockCameraService: Acquired by {user_id}. Dimensions: {self.width}x{self.height}, FPS: {self.fps}")
                return self.mock_cap
            logger.warning(f"MockCameraService: Acquisition failed for {user_id}, already held by {self.active_user}")
            return None

        def release(self, user_id):
            if self.active_user == user_id:
                logger.info(f"MockCameraService: Released by {user_id}")
                if self.mock_cap:
                    self.mock_cap.release()
                    self.mock_cap = None
                self.active_user = None
                return True
            logger.warning(f"MockCameraService: Release failed for {user_id}, not active user or not held.")
            return False

        def is_active_by(self, user_id):
            is_active = self.active_user == user_id
            logger.debug(f"MockCameraService: is_active_by({user_id}) -> {is_active} (current: {self.active_user})")
            return is_active

        def get_fps(self):
            return self.fps
        
        def get_dimensions(self):
            return self.width, self.height

    logging.basicConfig(level=logging.DEBUG) # Enable debug for testing
    logger.info("Testing ClapTracker with MockCameraService...")
    
    mock_cam_service = MockCameraService(camera_index=0) # Use a camera index that might work
    
    # Provide necessary arguments including camera_service
    tracker = ClapTracker(
        camera_service=mock_cam_service, 
        target_claps=5,
        desired_width=640,  # Example dimensions
        desired_height=480
    )

    logger.info("Starting ClapTracker...")
    success, message = tracker.start()
    logger.info(f"Start attempt: Success={success}, Message='{message}'")

    if success:
        try:
            last_status_log_time = time.time()
            while tracker.is_active(): # is_active now uses camera_service
                frame_bytes = tracker.get_frame()
                if frame_bytes:
                    frame_np = np.frombuffer(frame_bytes, dtype=np.uint8)
                    img = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
                    if img is not None:
                        cv2.imshow("Clap Tracker Test", img)
                    else:
                        logger.warning("Failed to decode frame from ClapTracker.")
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("\'q\' pressed, stopping tracker.")
                    break
                
                # Log status periodically
                if time.time() - last_status_log_time > 2:
                    status = tracker.get_status()
                    logger.info(f"ClapTracker Status: {status}")
                    last_status_log_time = time.time()
                    if status.get('target_reached'):
                        logger.info("Target reached! Stopping test.")
                        break
                
                # Auto-stop after some time for testing if no target
                # if time.time() - start_time_test > 20 and tracker.target_claps == 0:
                #     logger.info("Test time limit reached.")
                #     break

                time.sleep(0.01) # Small sleep to prevent busy-waiting on imshow
        
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received.")
        finally:
            logger.info("Stopping ClapTracker from test...")
            stop_success, stop_message = tracker.stop()
            logger.info(f"Stop attempt: Success={stop_success}, Message='{stop_message}'")
            cv2.destroyAllWindows()
            logger.info("Test finished.")
    else:
        logger.error("Could not start ClapTracker for testing (e.g., camera acquisition failed).") 