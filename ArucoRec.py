import cv2
import numpy as np
import csv

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

# Video input and output setup
video_path = 'C:\\Users\\בר\\OneDrive\\שולחן העבודה\\מדמח\\שנה ג\\רובוטים אוטונומים\\Matala2\\Aruco_Code_Rec\\ArucoVideo.mp4'
output_video_path = 'C:\\Users\\בר\\OneDrive\\שולחן העבודה\\מדמח\\שנה ג\\רובוטים אוטונומים\\Matala2\\Aruco_Code_Rec\\annotated_video.mp4'
csv_file = 'aruco_data.csv'

cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

camera_matrix = np.array([[1000, 0, frame_width / 2],
                          [0, 1000, frame_height / 2],
                          [0, 0, 1]], dtype=float)
dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

csv_columns = ['Frame', 'ID', 'Corner1_X', 'Corner1_Y', 'Corner2_X', 'Corner2_Y', 'Corner3_X', 'Corner3_Y', 'Corner4_X', 'Corner4_Y', 'Distance', 'Yaw', 'Pitch', 'Roll']
csv_data = []

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)

        for i in range(len(ids)):
            axis_length = 0.1
            rvec, tvec = rvecs[i], tvecs[i]
            points, _ = cv2.projectPoints(axis_length * np.float32([[0,0,0], [1,0,0], [0,1,0], [0,0,1]]), rvec, tvec, camera_matrix, dist_coeffs)
            origin = tuple(corners[i][0][0].astype(int))
            cv2.line(frame, origin, tuple(points[0].ravel().astype(int)), (0,0,255), 3) # x-axis
            cv2.line(frame, origin, tuple(points[1].ravel().astype(int)), (0,255,0), 3) # y-axis
            cv2.line(frame, origin, tuple(points[2].ravel().astype(int)), (255,0,0), 3) # z-axis
            
            distance = np.linalg.norm(tvecs[i])
            yaw, pitch, roll = calculate_orientation_angles(rvecs[i])

            corner_points = corners[i].reshape((4, 2))
            corner_list = corner_points.flatten().tolist()

            row_data = [frame_count, ids[i][0]] + corner_list + [distance, yaw, pitch, roll]
            csv_data.append(row_data)

            cv2.putText(frame, f'ID: {ids[i][0]}', (int(corners[i][0][0][0]), int(corners[i][0][0][1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out.write(frame)
    frame_count += 1

cap.release()
out.release()

# Write to CSV file
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(csv_columns)
    writer.writerows(csv_data)

print("Processing complete. Check the output video and CSV file.")