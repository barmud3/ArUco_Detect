import cv2
import cv2.aruco as aruco
import numpy as np
import os

def detect_aruco_markers(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
    parameters = aruco.DetectorParameters()

    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        return ids, corners
    else:
        return [], []

def calculate_error(target, current):
    return np.linalg.norm(target - current) / np.linalg.norm(target) * 100

def calculate_angle(corners):
    # Calculate the angle of the line connecting the top-left and top-right corners
    vector = corners[1] - corners[0]
    angle = np.arctan2(vector[1], vector[0])
    return np.degrees(angle)

def provide_feedback(target_center, current_center, target_size, current_size, target_angle, current_angle, tolerance=7):
    position_error_x = abs(target_center[0] - current_center[0])
    position_error_y = abs(target_center[1] - current_center[1])
    size_error = abs(target_size - current_size) / target_size * 100
    angle_error = abs(target_angle - current_angle)
    
    errors = {
        'Move Right': position_error_x if current_center[0] < target_center[0] else 0,
        'Move Left': position_error_x if current_center[0] > target_center[0] else 0,
        'Move Down': position_error_y if current_center[1] < target_center[1] else 0,
        'Move Up': position_error_y if current_center[1] > target_center[1] else 0,
        'Move Forward': size_error if current_size < target_size * (1 - tolerance / 100) else 0,
        'Move Backward': size_error if current_size > target_size * (1 + tolerance / 100) else 0,
        'Turn Right': angle_error if current_angle < target_angle else 0,
        'Turn Left': angle_error if current_angle > target_angle else 0
    }
    
    # Prioritize errors
    priority_order = ['Move Right', 'Move Left', 'Move Down', 'Move Up', 'Move Forward', 'Move Backward', 'Turn Right', 'Turn Left']
    for action in priority_order:
        if errors[action] > tolerance:
            return action, False
    
    return "Matching", True

# Load the target frame and detect the ArUco marker
target_frame_path = 'C:\\New folder\\frame.png'
print(f"Trying to load image from: {target_frame_path}")

if not os.path.isfile(target_frame_path):
    print(f"File not found: {target_frame_path}")
    exit(1)

target_frame = cv2.imread(target_frame_path)

if target_frame is None:
    print(f"Error: Unable to open image file {target_frame_path}")
    exit(1)

target_ids, target_corners = detect_aruco_markers(target_frame)

if len(target_ids) == 0:
    print("No ArUco marker found in the target image.")
    exit(1)

# Assuming there's only one ArUco marker in the target image
target_id = target_ids[0][0]
target_corner = target_corners[0][0]
target_center = np.mean(target_corner, axis=0)
target_size = np.linalg.norm(target_corner[0] - target_corner[2])
target_angle = calculate_angle(target_corner)

# Get target frame dimensions
ref_height, ref_width = target_frame.shape[:2]

# Open video stream with camera set to match target frame dimensions
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, ref_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ref_height)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_ids, current_corners = detect_aruco_markers(frame)

    if len(current_ids) > 0 and target_id in current_ids:
        index = np.where(current_ids == target_id)[0][0]
        current_id = current_ids[index][0]
        current_corner = current_corners[index][0]
        current_center = np.mean(current_corner, axis=0)
        current_size = np.linalg.norm(current_corner[0] - current_corner[2])
        current_angle = calculate_angle(current_corner)

        feedback, matching = provide_feedback(target_center, current_center, target_size, current_size, target_angle, current_angle, tolerance=20)
        cv2.putText(frame, f"ID: {current_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, feedback, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if matching else (0, 0, 255), 2)

        if matching:
            cv2.putText(frame, "Matching", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "ArUco ID not found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()