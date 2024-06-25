import cv2
import numpy as np

# Load the ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
parameters = cv2.aruco.DetectorParameters()

# Function to calculate the yaw, pitch, and roll angles
def calculate_orientation_angles(rvec):
    R, _ = cv2.Rodrigues(rvec)
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.degrees(pitch), np.degrees(roll), np.degrees(yaw)

# Normalize angle differences to be within -180 to 180 degrees
def normalize_angle_difference(angle1, angle2):
    diff = angle1 - angle2
    return (diff + 180) % 360 - 180

class DroneMovement:
    def __init__(self, target_positions, target_orientations):
        self.target_positions = target_positions
        self.target_orientations = target_orientations
        self.position_tolerance = 0.9999  # Adjust as needed
        self.orientation_priority_threshold = 10.0  # Threshold in degrees for prioritizing orientation commands

    def get_movement_command(self, current_positions, current_orientations, current_ids):
        best_command = None
        best_delta = float('inf')
        orientation_priority = False

        for i, current_id in enumerate(current_ids):
            if current_id in self.target_positions:
                t_tx, t_ty, t_tz = self.target_positions[current_id]
                t_yaw, t_pitch, t_roll = self.target_orientations[current_id]

                tx, ty, tz = current_positions[i]
                yaw, pitch, roll = current_orientations[i]

                position_deltas = {
                    'up': t_ty - ty,
                    'down': ty - t_ty,
                    'left': tx - t_tx,
                    'right': t_tx - tx,
                    'forward': t_tz - tz,
                    'backward': tz - t_tz
                }

                yaw_delta = normalize_angle_difference(t_yaw, yaw)

                orientation_deltas = {
                    'turn-left': -yaw_delta,
                    'turn-right': yaw_delta
                }

                # Check if orientation delta exceeds the priority threshold
                if abs(yaw_delta) > self.orientation_priority_threshold:
                    orientation_priority = True

                # Combine position and orientation deltas
                combined_deltas = {**position_deltas, **orientation_deltas}

                # Print deltas for debugging
                print(f"Current ID: {current_id}")
                print(f"Position Deltas: {position_deltas}")
                print(f"Orientation Deltas: {orientation_deltas}")

                # Find the command with the smallest delta
                for command, delta in combined_deltas.items():
                    # Skip position commands if orientation priority is set
                    if orientation_priority and command not in orientation_deltas:
                        continue
                    print(f"Evaluating Command: {command}, Delta: {delta}")
                    if abs(delta) < abs(best_delta):
                        best_command = command
                        best_delta = delta

                print(f"Best Command So Far: {best_command}, Best Delta So Far: {best_delta}")

        return best_command





    def is_match(self, current_positions, current_orientations, current_ids, orientation_tolerance=0.3):
        for i, current_id in enumerate(current_ids):
            if current_id in self.target_positions:
                t_tx, t_ty, t_tz = self.target_positions[current_id]
                t_yaw, t_pitch, t_roll = self.target_orientations[current_id]

                tx, ty, tz = current_positions[i]
                yaw, pitch, roll = current_orientations[i]

                # Calculate tolerance for angles (10% tolerance)
                yaw_tolerance = orientation_tolerance * abs(t_yaw)
                pitch_tolerance = orientation_tolerance * abs(t_pitch)
                roll_tolerance = orientation_tolerance * abs(t_roll)

                # Check if within orientation tolerance
                if (abs(normalize_angle_difference(t_yaw, yaw)) > yaw_tolerance or
                    abs(normalize_angle_difference(t_pitch, pitch)) > pitch_tolerance or
                    abs(normalize_angle_difference(t_roll, roll)) > roll_tolerance):
                    continue

                # Calculate position tolerance
                pos_tolerance_x = self.position_tolerance * abs(t_tx)
                pos_tolerance_y = self.position_tolerance * abs(t_ty)
                pos_tolerance_z = self.position_tolerance * abs(t_tz)

                # Check if within position tolerance
                if (abs(t_tx - tx) > pos_tolerance_x or
                    abs(t_ty - ty) > pos_tolerance_y or
                    abs(t_tz - tz) > pos_tolerance_z):
                    continue

                # If both orientation and position are within tolerance, return True
                return True

        return False

    def execute_command(self, command):
        # Define actions based on commands
        if command == 'forward':
            print("Moving forward")
            # Implement forward movement logic
        elif command == 'backward':
            print("Moving backward")
            # Implement backward movement logic
        elif command == 'right':
            print("Moving right")
            # Implement right movement logic
        elif command == 'left':
            print("Moving left")
            # Implement left movement logic
        elif command == 'up':
            print("Moving up")
            # Implement upward movement logic
        elif command == 'down':
            print("Moving down")
            # Implement downward movement logic
        elif command == 'turn-right':
            print("Turning right")
            # Implement turn-right logic
        elif command == 'turn-left':
            print("Turning left")
            # Implement turn-left logic
        elif command == 'tilt-forward':
            print("Tilting forward")
            # Implement tilt forward logic
        elif command == 'tilt-backward':
            print("Tilting backward")
            # Implement tilt backward logic
        elif command == 'tilt-right':
            print("Tilting right")
            # Implement tilt right logic
        elif command == 'tilt-left':
            print("Tilting left")
            # Implement tilt left logic
        else:
            print("Unknown command")

# Define camera matrix and distortion coefficients
frame_width = 640  # Adjust if necessary
frame_height = 480  # Adjust if necessary
camera_matrix = np.array([[1000, 0, frame_width / 2],
                          [0, 1000, frame_height / 2],
                          [0, 0, 1]], dtype=float)
dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

# Function to capture and process the target frame
def process_target_frame(target_frame):
    gray_target = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)
    corners_target, ids_target, _ = cv2.aruco.detectMarkers(gray_target, aruco_dict, parameters=parameters)
    if ids_target is not None:
        rvecs_target, tvecs_target, _ = cv2.aruco.estimatePoseSingleMarkers(corners_target, 0.05, camera_matrix, dist_coeffs)
        target_positions = {id[0]: tuple(tvec.flatten()) for id, tvec in zip(ids_target, tvecs_target)}
        target_orientations = {id[0]: calculate_orientation_angles(rvec) for id, rvec in zip(ids_target, rvecs_target)}
        return target_positions, target_orientations, set(id[0] for id in ids_target)
    else:
        print("No ArUco markers found in the target frame.")
        exit()

# Load and process the target frame from a photo file
target_frame_path = 'C:\\New folder\\frame.png'

target_frame = cv2.imread(target_frame_path)

if target_frame is None:
    print(f"Could not load image from {target_frame_path}")
    exit()

target_positions, target_orientations, target_ids = process_target_frame(target_frame)
drone_movement = DroneMovement(target_positions, target_orientations)

# Initialize live video capture (using PC camera)
live_cap = cv2.VideoCapture(0)  # Change 0 to the appropriate camera index if needed

while live_cap.isOpened():
    ret, frame = live_cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)
        current_positions = [tuple(tvec.flatten()) for tvec in tvecs]
        current_orientations = [calculate_orientation_angles(rvec) for rvec in rvecs]
        current_ids = [id[0] for id in ids]

        # Check if the detected markers match the target markers
        if target_ids.issubset(current_ids):
            command = drone_movement.get_movement_command(current_positions, current_orientations, current_ids)

            for i in range(len(ids)):
                cv2.putText(frame, f'ID: {ids[i][0]}', (int(corners[i][0][0][0]), int(corners[i][0][0][1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if command:
                cv2.putText(frame, f'Command: {command}', (10, frame_height - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Execute the command
                drone_movement.execute_command(command)

            if drone_movement.is_match(current_positions, current_orientations, current_ids):
                cv2.putText(frame, 'Matching!', (10, frame_height - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow('Live Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

live_cap.release()
cv2.destroyAllWindows()

print("Processing complete.")
