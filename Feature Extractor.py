
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
height = input("Enter height option, 1 for high 2 for low, 3 for test: ")
csv_file = f"posture_data_{height}.csv"


with open(csv_file, "w", newline="") as f:
	writer = csv.writer(f)
	writer.writerow([
		"normalized_shoulderwidth_1",
		"normalized_shoulderwidth_2",
		"shoulderwidth_nontransformed",
		"normalized_neck_length",
		"nose_depth",
		"neck_angle",
		"shoulder_width",
		"head_tilt",
		"nose_depth",
		"left_shoulder_depth",
		"right_shoulder_depth"
	])

# -----------------------------
# Helper functions
# -----------------------------


def signed_angle(v1, v2):
    # normalize (optional but keeps things stable)
    v1 = v1 / (np.linalg.norm(v1) + 1e-6)
    v2 = v2 / (np.linalg.norm(v2) + 1e-6)

    dot = np.dot(v1, v2)
    cross = v1[0]*v2[1] - v1[1]*v2[0]

    angle = np.arctan2(cross, dot)  # radians
    return np.degrees(angle)



def rotate_y(v, theta):
    R = np.array([
        [ np.cos(theta), 0, np.sin(theta)],
        [ 0,             1, 0            ],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    return R @ v

def to_pixel(lm, w, h):
	return np.array([int(lm[0]*w), int(lm[1]*h)])

def to_pixel_nontransformed(lm, w, h):
	return np.array([int(lm.x * w), int(lm.y * h)])



def transform(landmark, left_shoulder, right_shoulder, is_nose = False):
    # 1. Calculate the center point between shoulders to use as our local origin (0,0,0)
    center_x = (left_shoulder.x + right_shoulder.x) / 2
    center_y = (left_shoulder.y + right_shoulder.y) / 2
    center_z = (left_shoulder.z + right_shoulder.z) / 2
    
    # 2. Calculate rotation angle based on shoulders
    # Added math.atan2 for stability to avoid division-by-zero errors if shoulders align vertically
	
    if is_nose:
        angle_of_rotation = math.atan2(
			(landmark.z - center_z), 
			(landmark.x - center_x)
		) + math.pi/2  # +90 degrees to align with vertical axis
    else:
        angle_of_rotation = math.atan2(
        (left_shoulder.z - right_shoulder.z), 
        (left_shoulder.x - right_shoulder.x) 
    ) 
    
    # 3. Shift the current landmark so the shoulder center is (0,0,0)
    v = np.array([
        landmark.x - center_x, 
        landmark.y - center_y, 
        landmark.z - center_z
    ])
    
    # 4. Rotate around our new local origin
    # We use negative angle to 'undo' your body's rotation and face the camera straight
    rotated_array = rotate_y(v, angle_of_rotation)
    rotated_array[0] = rotated_array[0] + center_x  # Shift back to original coordinate space
    rotated_array[1] = rotated_array[1] + center_y
    rotated_array[2] = rotated_array[2] + center_z

    
    # 5. Calculate 3D inter-shoulder distance for scale normalization
    inter_shoulder_distance = math.sqrt(
        (left_shoulder.x - right_shoulder.x)**2 + 
        (left_shoulder.y - right_shoulder.y)**2 + 
        (left_shoulder.z - right_shoulder.z)**2
    )
    
    # Prevent division by zero just in case MediaPipe glitches
    if inter_shoulder_distance == 0:
        inter_shoulder_distance = 1e-6
        
    # 6. Normalize by the shoulder width
    normalized_x = rotated_array[0] / inter_shoulder_distance
    normalized_y = rotated_array[1] / inter_shoulder_distance
    normalized_z = rotated_array[2] / inter_shoulder_distance
    # return normalized_x, normalized_y, normalized_z
    return rotated_array[0], rotated_array[1], rotated_array[2], angle_of_rotation


	

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
			landmarks = landmarksorg.pose_landmarks[0]
			
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

		list_of_landmarks = [left_ear, right_ear, left_shoulder, right_shoulder] 
		transformed_features = [transform(lm, left_shoulder, right_shoulder) for lm in list_of_landmarks]
		nose_feature = transform(nose, left_shoulder, right_shoulder, is_nose=True)



		# Convert to pixels
		nose_p = to_pixel(nose_feature, w, h)
		le_p = to_pixel(transformed_features[0], w, h)
		re_p = to_pixel(transformed_features[1], w, h)
		ls_p = to_pixel(transformed_features[2], w, h)
		rs_p = to_pixel(transformed_features[3], w, h)

		#Non-transformed pixel coordinates for depth features
		nose_p_nontransformed = to_pixel_nontransformed(nose, w, h)
		ls_p_nontransformed = to_pixel_nontransformed(left_shoulder, w, h)	
		rs_p_nontransformed = to_pixel_nontransformed(right_shoulder, w, h)
		le_p_nontransformed = to_pixel_nontransformed(left_ear, w, h)
		re_p_nontransformed = to_pixel_nontransformed(right_ear, w, h)

		

		# -----------------------------
		# Feature 1: Neck vector & angle
		# -----------------------------
		shoulder_mid = (ls_p + rs_p) / 2
		neck_vector = nose_p - shoulder_mid
		vertical = np.array([0, 1])
		neck_angle = signed_angle(neck_vector, vertical)

		#Non-transformed neck vector and angle for depth features
		shoulder_mid_nontransformed = (ls_p_nontransformed + rs_p_nontransformed) / 2
		neck_vector_nontransformed = nose_p_nontransformed - shoulder_mid_nontransformed
		vertical_nontransformed = np.array([0, 1])
		neck_angle_nontransformed = signed_angle(neck_vector_nontransformed, vertical_nontransformed)

		# -----------------------------
		# Feature 2: Neck length
		# -----------------------------
		neck_length = np.linalg.norm(neck_vector)

		# -----------------------------
		# Feature 3: Shoulder width
		# -----------------------------
		shoulder_width = math.sqrt((ls_p[0] - rs_p[0])**2 + (ls_p[1] - rs_p[1])**2)
		shoulder_width_nontransformed = ls_p_nontransformed[0] - rs_p_nontransformed[0]


		# -----------------------------
		# Feature 4: Head tilt
		# -----------------------------
		ear_vector = re_p - le_p
		horizontal = np.array([1, 0])
		head_tilt = signed_angle(ear_vector, horizontal)

		# -----------------------------
		# Feature 5: Depth
		# -----------------------------
		nose_depth = nose.z
		left_shoulder_depth = left_shoulder.z
		right_shoulder_depth = right_shoulder.z

		#-----------------------------
		# Transformed Features
		#-----------------------------
		# list_of_landmarks = [nose, left_shoulder, right_shoulder, left_ear, right_ear] 
		# transformed_features = [transform(lm) for lm in list_of_landmarks]

		

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
		cv2.line(frame, tuple(shoulder_mid_nontransformed.astype(int)), tuple(nose_p_nontransformed), (255, 255, 0), 2)
		cv2.line(frame, tuple(ls_p), tuple(rs_p), (255, 0, 0), 2)
		cv2.line(frame, tuple(ls_p_nontransformed), tuple(rs_p_nontransformed), (255, 255, 0), 2)


		# -----------------------------
		# Display text
		# -----------------------------
		cv2.putText(frame, f"Angle {180 / math.pi * transformed_features[3][3]:.2f}", (30, 50),
					cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
		cv2.putText(frame, f"Shoulder width transformed: {shoulder_width:.2f}", (30, 170),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
		cv2.putText(frame, f"Shoulder width: {shoulder_width_nontransformed:.2f}", (30, 200),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
	

		# -----------------------------
		# Save the features
		# -----------------------------
		with open(csv_file, "a", newline="") as f:
			writer = csv.writer(f)
			writer.writerow([
				shoulder_width_nontransformed/math.sin(90 - abs(transformed_features[3][3]))**2,
				shoulder_width_nontransformed/math.sin(90 - abs(transformed_features[3][3])),
				shoulder_width_nontransformed,
				transformed_features[0][1],  # Normalized neck length (y component)
				nose.z,
				neck_angle,
				shoulder_width,
				head_tilt,
				nose_depth,
				left_shoulder.z,
				right_shoulder.z
			])
		
		cv2.imshow("Posture Detection", frame) 

		filename = os.path.join(output_dir, f"frame_{count:05d}.jpg")
		cv2.imwrite(filename, frame)


		count +=1


		if count == 500:
			break
			print("Now slouch")
			time.sleep(5)  # Give user time to change posture
		
		if count >= 2000:
			break

		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

cap.release()
cv2.destroyAllWindows()

