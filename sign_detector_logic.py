import cv2
import numpy as np
import tensorflow as tf
import os
import time

class SignDetector:
    def __init__(self, model_path, train_path_for_labels, img_height=64, img_width=64, conf_threshold=0.85, stability_threshold_time=0.7):
        self.IMG_HEIGHT = img_height
        self.IMG_WIDTH = img_width
        self.CONF_THRESHOLD = conf_threshold
        self.STABILITY_THRESHOLD_TIME = stability_threshold_time
        self.MODEL_PATH = model_path
        self.TRAIN_PATH_FOR_LABELS = train_path_for_labels

        self._configure_gpu()
        self.model = self._load_model()
        self.labels = self._load_labels()

        # State variables
        self.translated_text = ""
        self.current_stable_prediction = None
        self.last_consistent_time = time.time()
        self.last_added_char = None
        self.last_frame_prediction = None # Store prediction of the very last frame processed
        self.last_frame_confidence = 0.0  # Store confidence of the very last frame processed


    def _configure_gpu(self):
        """Configures GPU memory growth."""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs configured.")
            except RuntimeError as e:
                print("GPU Memory growth error:", e)

    def _load_model(self):
        """Loads the Keras model."""
        print(f"Loading model from: {self.MODEL_PATH}")
        try:
            model = tf.keras.models.load_model(self.MODEL_PATH)
            print("Model loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            # Depending on how critical this is, you might exit or return None
            # For a web app, returning None and handling it might be better than exiting
            return None

    def _load_labels(self):
        """Loads labels from the training directory or uses defaults."""
        if os.path.exists(self.TRAIN_PATH_FOR_LABELS) and os.path.isdir(self.TRAIN_PATH_FOR_LABELS):
            labels = sorted([d for d in os.listdir(self.TRAIN_PATH_FOR_LABELS) if os.path.isdir(os.path.join(self.TRAIN_PATH_FOR_LABELS, d))])
            print(f"Loaded {len(labels)} labels from {self.TRAIN_PATH_FOR_LABELS}: {labels}")

        else:
            print(f"Warning: TRAIN_PATH_FOR_LABELS '{self.TRAIN_PATH_FOR_LABELS}' not found or not a directory, using default labels. ENSURE THIS ORDER IS CORRECT.")
            labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                      'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                      'del', 'nothing', 'space']
            print(f"Using default labels: {labels}")
        return labels

    def preprocess_frame(self, frame_roi_gray):
        """Preprocesses a single grayscale frame for the model."""
        if frame_roi_gray is None or frame_roi_gray.size == 0:
             print("Warning: Received empty frame for preprocessing.")
             return None
        try:
            img_resized = cv2.resize(frame_roi_gray, (self.IMG_WIDTH, self.IMG_HEIGHT))
            img_array = np.asarray(img_resized, dtype=np.float32)
            mean = np.mean(img_array)
            std = np.std(img_array)
            if std < 1e-6: std = 1e-6
            img_normalized = (img_array - mean) / std
            img_final = np.expand_dims(img_normalized, axis=-1) # Add channel dim
            img_final = np.expand_dims(img_final, axis=0)      # Add batch dim
            return img_final
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            return None


    def process_frame(self, roi_gray):
        """Processes a single frame ROI, updates state, and returns detection info."""
        if self.model is None:
            print("Model not loaded, cannot process frame.")
            return None, 0.0 # Return None prediction and 0 confidence

        processed_roi = self.preprocess_frame(roi_gray)
        if processed_roi is None:
             return self.last_frame_prediction, self.last_frame_confidence # Return previous if preprocess failed

        try:
            predictions = self.model.predict(processed_roi, verbose=0)
            predicted_index = np.argmax(predictions[0])
            prediction_probability = np.max(predictions[0])
        except Exception as e:
            print(f"Error during model prediction: {e}")
            return self.last_frame_prediction, self.last_frame_confidence # Return previous on error

        current_prediction = None # Reset prediction for this frame's logic

        # Check confidence threshold
        if prediction_probability >= self.CONF_THRESHOLD and predicted_index < len(self.labels):
            current_prediction = self.labels[predicted_index]

        # Store raw prediction for immediate display feedback
        self.last_frame_prediction = current_prediction
        self.last_frame_confidence = float(prediction_probability) # Ensure it's a standard float

        # --- Stability Check & Text Assembly ---
        current_time = time.time()

        # Prepare potential prediction for stability logic (ignore 'nothing')
        potential_stable_pred = current_prediction
        if potential_stable_pred is not None and potential_stable_pred == 'nothing':
            potential_stable_pred = None # Treat 'nothing' like an uncertain prediction for stability

        if potential_stable_pred is not None and potential_stable_pred == self.current_stable_prediction:
            # Prediction is stable (same as the last one that was above threshold and not 'nothing')
            if current_time - self.last_consistent_time >= self.STABILITY_THRESHOLD_TIME:
                # Sign has been held stable long enough
                if potential_stable_pred != self.last_added_char: # Add only if it's different from last added
                    action_taken = False
                    if potential_stable_pred == 'space':
                        if not self.translated_text.endswith(" "): # Avoid double spaces
                           self.translated_text += " "
                           action_taken = True
                           print("Action: Added Space")
                        self.last_added_char = None # Reset last added after space attempt
                    elif potential_stable_pred == 'del':
                        if len(self.translated_text) > 0:
                             self.translated_text = self.translated_text[:-1] # Remove last character
                             action_taken = True
                             print("Action: Deleted Character")
                        self.last_added_char = None # Reset last added after delete attempt
                    else: # It's a letter
                        self.translated_text += potential_stable_pred
                        self.last_added_char = potential_stable_pred # Track the added letter
                        action_taken = True
                        print(f"Action: Added '{potential_stable_pred}'")

                    # Reset timer ONLY if an action was taken to prevent immediate re-adding
                    if action_taken:
                        self.last_consistent_time = time.time()

        elif potential_stable_pred != self.current_stable_prediction:
             # Prediction changed OR dropped below threshold OR became 'nothing'
             self.current_stable_prediction = potential_stable_pred # Update the candidate for stability
             self.last_consistent_time = time.time() # Reset stability timer
             # If the prediction is now unstable/None/'nothing', allow the next stable sign to be added immediately
             if self.current_stable_prediction is None:
                 self.last_added_char = None

        # Return the raw prediction and confidence for immediate display this frame
        return self.last_frame_prediction, self.last_frame_confidence

    def get_translated_text(self):
        """Returns the current translated text."""
        return self.translated_text

    def get_stable_prediction(self):
        """Returns the current stable prediction candidate."""
        # Return the candidate for stability, which might be None if confidence low or 'nothing'
        return self.current_stable_prediction

    def reset_translated_text(self):
        """Resets the accumulated translated text and related stability state."""
        print("Resetting translated text and stability state.") # Keep log
        self.translated_text = ""
        # Reset the actual state variables used in process_frame
        self.current_stable_prediction = None
        self.last_consistent_time = time.time() # Reset timer to now
        self.last_added_char = None
        # last_frame_prediction/confidence are overwritten each frame, no need to reset explicitly
        print("Detector's translated text and stability state reset.")

# --- END OF FILE sign_detector_logic.py ---