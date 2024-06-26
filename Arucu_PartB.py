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

# Load the reference image and detect the ArUco marker
reference_image_path = 'C:\\New folder\\frame.png'
print(f"Trying to load image from: {reference_image_path}")

if not os.path.isfile(reference_image_path):
    print(f"File not found: {reference_image_path}")
    exit(1)

reference_image = cv2.imread(reference_image_path)

if reference_image is None:
    print(f"Error: Unable to open image file {reference_image_path}")
    exit(1)

reference_ids, reference_corners = detect_aruco_markers(reference_image)

if len(reference_ids) == 0:
    print("No ArUco marker found in the reference image.")
    exit(1)

# Assuming there's only one ArUco marker in the reference image
reference_id = reference_ids[0][0]
reference_corner = reference_corners[0][0]
reference_center = np.mean(reference_corner, axis=0)
reference_size = np.linalg.norm(reference_corner[0] - reference_corner[2])
reference_angle = calculate_angle(reference_corner)

# Get reference image dimensions
ref_height, ref_width = reference_image.shape[:2]

# Open video stream with camera set to match reference image dimensions
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, ref_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ref_height)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_ids, current_corners = detect_aruco_markers(frame)

    if len(current_ids) > 0 and reference_id in current_ids:
        index = np.where(current_ids == reference_id)[0][0]
        current_id = current_ids[index][0]
        current_corner = current_corners[index][0]
        current_center = np.mean(current_corner, axis=0)
        current_size = np.linalg.norm(current_corner[0] - current_corner[2])
        current_angle = calculate_angle(current_corner)

        feedback, matching = provide_feedback(reference_center, current_center, reference_size, current_size, reference_angle, current_angle, tolerance=20)
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