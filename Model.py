import pandas as pd
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import csv
import os
import time
import math
import datetime
from PIL import Image
import joblib

#-----------------------------
# Load the trained model
#-----------------------------

bundle = joblib.load("best_posture_model.pkl")

model = bundle["model"]
features = bundle["features"]

# -----------------------------
# Setup MediaPipe
# -----------------------------
model_path = 'C:\\Users\\ChuTs\\OneDrive - Government of Ontario\\Desktop\\Current Projects\\The-Posture-Project-\\pose_landmarker_full.task'

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a pose landmarker instance with the video mode:
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO)

# -----------------------------
# Setup Webcam
# -----------------------------
cap = cv2.VideoCapture(0)


# -----------------------------
# Helper functions
# -----------------------------

def rotate_y(v, theta):
    R = np.array([
        [ np.cos(theta), 0, np.sin(theta)],
        [ 0,             1, 0            ],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    return R @ v

def lm_to_vec(lm):
    return np.array([lm.x, lm.y, lm.z], dtype=float)

def angle_between(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))



def distance_normalization_factor(left_shoulder, right_shoulder, nose, is_nose = False):
	if not is_nose:
		inter_shoulder_distance = math.sqrt(
			(left_shoulder.x - right_shoulder.x)**2 + 
			(left_shoulder.y - right_shoulder.y)**2 + 
			(left_shoulder.z - right_shoulder.z)**2
		)
		
		# Prevent division by zero just in case MediaPipe glitches
		if inter_shoulder_distance == 0:
			inter_shoulder_distance = 1e-6
		
		return inter_shoulder_distance
	
	else:
		center_x = (left_shoulder.x + right_shoulder.x) / 2
		center_y = (left_shoulder.y + right_shoulder.y) / 2
		# center_z = (left_shoulder.z + right_shoulder.z) / 2

		nose_distance = math.sqrt(
			(nose.x - center_x)**2 + 
			(nose.y - center_y)**2 
			# + (nose.z - center_z)**2
		)

		return nose_distance

def rotation_normalization_factor(left_shoulder, right_shoulder):
	angle_of_rotation = math.atan2(
		(left_shoulder.z - right_shoulder.z), 
		(left_shoulder.x - right_shoulder.x) 
	) 
	return angle_of_rotation


	

# -----------------------------
# Define variables
# -----------------------------
count = 0
max_shoulder_width = 0
max_neck_length = 0
min_nose_depth = 10000000000000000000000
output_dir = f"Frames"


# -----------------------------
# Main loop
# -----------------------------

os.makedirs(output_dir, exist_ok=True)
print(f"Created new session folder: {output_dir}")



with PoseLandmarker.create_from_options(options) as landmarker:
	while cap.isOpened():
		
		ret, frame = cap.read()
		timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
		if not ret:
			break

		h, w, _ = frame.shape

		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		result = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
		
		landmarksorg = landmarker.detect_for_video(result, timestamp)

		
		# Check if pose_landmarks exists and is not empty
		if landmarksorg.pose_landmarks and len(landmarksorg.pose_landmarks) > 0:
			landmarks = landmarksorg.pose_world_landmarks[0]
			landmarks_non_world = landmarksorg.pose_landmarks[0]
			
			# Put the rest of your posture processing code here
			# e.g., neck_y_component_length = ratio * neck_vector[1]

		else:
			# Handle frames where no person is detected
			print(f"No person detected at timestamp {timestamp}")
			# You might want to 'continue' the loop or skip this frame
		
	

		# Key landmarks
		nose = landmarks[0]
		left_shoulder = landmarks[11]
		right_shoulder = landmarks[12]
		left_ear = landmarks[7]
		right_ear = landmarks[8]

		#Key ladmarks for non-world coordinates
		nose_non_world = landmarks_non_world[0]
		left_shoulder_non_world = landmarks_non_world[11]
		right_shoulder_non_world = landmarks_non_world[12]

		normalization_factor_distance = distance_normalization_factor(left_shoulder_non_world, right_shoulder_non_world, nose_non_world, is_nose=False)
		normalization_factor_nose = distance_normalization_factor(left_shoulder_non_world, right_shoulder_non_world, nose_non_world, is_nose=True)
		angle_of_rotation = rotation_normalization_factor(left_shoulder_non_world, right_shoulder_non_world)
		

		# -----------------------------
		# Feature 1: Neck vector 
		# -----------------------------
		shoulder_mid = (lm_to_vec(left_shoulder) + lm_to_vec(right_shoulder)) / 2
		neck_vector = lm_to_vec(nose) - shoulder_mid

		# -----------------------------
		# Feature 2: Neck length
		# -----------------------------
		neck_length_y = neck_vector[1]


		# -----------------------------
		# Feature 2: Neck Angle
		# -----------------------------
		vertical = np.array([0, -1, 0])
		slouch_angle = angle_between(vertical, neck_vector)

		# -----------------------------
		# Feature 3: Shoulder width
		# -----------------------------
		inter_shoulder_distance = math.sqrt(
        (left_shoulder.x - right_shoulder.x)**2 + 
        (left_shoulder.y - right_shoulder.y)**2)
		
        #-----------------------------
        # Feature row for prediction
        #-----------------------------
		feature_row = pd.DataFrame([{
            "intershoulder_distance": inter_shoulder_distance,
            "necklength_y": neck_length_y,
            "slouchangle": slouch_angle,
            "angle_of_rotation": angle_of_rotation,
            "normalization_factor_distance": normalization_factor_distance,
            "nose_normalization_factor": normalization_factor_nose
        }])
		X_new = feature_row[features]
		
		predictions = model.predict(X_new)
		probabilities = model.predict_proba(X_new)
		# -----------------------------
		# Display text
		# -----------------------------
		
		label = "Slouched" if predictions[0] == 1 else "Upright"
		cv2.putText(frame, f"Prediction: {label}", (30, 170),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2),


		cv2.imshow("Posture Detection", frame) 


		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

cap.release()
cv2.destroyAllWindows()

