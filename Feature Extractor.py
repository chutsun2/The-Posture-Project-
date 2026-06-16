
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import csv
import os
import time

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
# CSV file setup
# -----------------------------
csv_file = "posture_data.csv"


with open(csv_file, "w", newline="") as f:
	writer = csv.writer(f)
	writer.writerow([
		"neck_angle",
		"normalized_neck_length",
		"shoulder_width",
		"head_tilt",
		"nose_depth",
		"left_shoulder_depth",
		"right_shoulder_depth"
	])

# -----------------------------
# Helper functions
# -----------------------------
def to_pixel(lm, w, h):
	return np.array([int(lm.x * w), int(lm.y * h)])

def angle_between(v1, v2):
	v1 = v1 / (np.linalg.norm(v1) + 1e-6)
	v2 = v2 / (np.linalg.norm(v2) + 1e-6)
	return np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))

count = 0

# -----------------------------
# Main loop
# -----------------------------
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
		landmarks = landmarksorg.pose_landmarks[0]
		realworld_landmarks = landmarksorg.pose_world_landmarks[0]

		# Key landmarks
		nose = landmarks[0]
		left_shoulder = landmarks[11]
		right_shoulder = landmarks[12]
		left_ear = landmarks[7]
		right_ear = landmarks[8]

		# Key world landmarks
		nose_world = realworld_landmarks[0]
		left_shoulder_world = realworld_landmarks[11]
		right_shoulder_world = realworld_landmarks[12]

		# Convert to pixels
		nose_p = to_pixel(nose, w, h)
		ls_p = to_pixel(left_shoulder, w, h)
		rs_p = to_pixel(right_shoulder, w, h)
		le_p = to_pixel(left_ear, w, h)
		re_p = to_pixel(right_ear, w, h)

		# -----------------------------
		# Feature 1: Neck vector & angle
		# -----------------------------
		shoulder_mid = (ls_p + rs_p) / 2
		neck_vector = nose_p - shoulder_mid
		vertical = np.array([0, -1])
		neck_angle = angle_between(neck_vector, vertical)

		# -----------------------------
		# Feature 2: Neck length
		# -----------------------------
		neck_length = np.linalg.norm(neck_vector)

		# -----------------------------
		# Feature 3: Shoulder width
		# -----------------------------
		shoulder_width = np.linalg.norm(ls_p - rs_p)

		# Normalize neck length
		normalized_neck_length = neck_length / (shoulder_width + 1e-6)

		# -----------------------------
		# Feature 4: Head tilt
		# -----------------------------
		ear_vector = re_p - le_p
		horizontal = np.array([1, 0])
		head_tilt = angle_between(ear_vector, horizontal)

		# -----------------------------
		# Feature 5: Depth
		# -----------------------------
		nose_depth = nose.z
		left_shoulder_depth = left_shoulder.z
		right_shoulder_depth = right_shoulder.z

		# -----------------------------
		# Classification (simple rule)
		# -----------------------------
		if neck_angle > 25:
			posture = "Slouching"
			color = (0, 0, 255)
		else:
			posture = "Good Posture"
			color = (0, 255, 0)

		# -----------------------------
		# Draw lines (visual debugging)
		# -----------------------------
		cv2.line(frame, tuple(shoulder_mid.astype(int)), tuple(nose_p), color, 2)
		cv2.line(frame, tuple(ls_p), tuple(rs_p), (255, 0, 0), 2)


		# -----------------------------
		# Display text
		# -----------------------------
		cv2.putText(frame, posture, (30, 50),
					cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
		cv2.putText(frame, f"Angle: {int(neck_angle)}", (30, 90),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
		cv2.putText(frame, f"Nose depth: {nose_depth:.2f}", (30, 130),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
		cv2.putText(frame, f"Shoulder 1 depth: {left_shoulder_depth:.2f}", (30, 170),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
		cv2.putText(frame, f"Shoulder 2 depth: {right_shoulder_depth:.2f}", (30, 210),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
		
		#World
		# cv2.putText(frame, f"Nose depth: {nose_world.z:.2f}", (int(w*0.4), 130),
		# 			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
		# cv2.putText(frame, f"Shoulder 1 depth: {left_shoulder_world.z:.2f}", (int(w*0.4), 170),
		# 			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
		# cv2.putText(frame, f"Shoulder 2 depth: {right_shoulder_world.z:.2f}", (int(w*0.4), 210),
		# 			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

		# -----------------------------
		# Save features
		# -----------------------------
		with open(csv_file, "a", newline="") as f:
			writer = csv.writer(f)
			writer.writerow([
				neck_angle,
				normalized_neck_length,
				shoulder_width,
				head_tilt,
				nose_depth,
				left_shoulder.z,
				right_shoulder.z
			])
		
		count +=1

		cv2.imshow("Posture Detection", frame)

		if count == 1000:
			print("Now slouch")
			time.sleep(5)  # Give user time to change posture
		
		if count >= 2000:
			break

		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

cap.release()
cv2.destroyAllWindows()

