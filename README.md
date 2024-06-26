# ArUco Marker Detection and Pose Estimation

This project involves detecting ArUco markers in a video, estimating their pose, and annotating the video with the detected markers and their orientations. The information is also logged into a CSV file. Additionally, it provides feedback commands for drone movements based on positional and orientational errors relative to a reference ArUco marker.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Part A: ArUco Marker Detection and Pose Estimation](#part-a-aruco-marker-detection-and-pose-estimation)
  - [Part B: Drone Control Using ArUco Markers](#part-b-drone-control-using-aruco-markers)
- [Files](#files)
- [Output](#output)
  - [Part A: Annotated Video](#part-a-annotated-video)
  - [Part A: CSV File](#part-a-csv-file)
  - [Part B: Feedback Commands](#part-b-feedback-commands)
- [Contact](#contact)

## Installation

### Prerequisites

- Python 3.x
- OpenCV
- NumPy

### Installation Steps

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/your-repository-name.git
    cd your-repository-name
    ```

2. **Install the required Python packages:**
    ```sh
    pip install opencv-python opencv-contrib-python numpy
    ```

## Usage

### Part A: ArUco Marker Detection and Pose Estimation

1. **Place your input video in the project directory:**
    Update the `video_path` variable in the script to point to your input video file.

2. **Run the script:**
    ```sh
    python aruco_detection.py
    ```

    This will process the video, detect ArUco markers, annotate the video, and generate a CSV file with the markers' data.

### Part B: Drone Control Using ArUco Markers

1. **Set Up Reference Image:**
   - Place the reference image containing the ArUco marker in the specified path.
   - Update the `reference_image_path` variable in the script to point to your reference image file.

2. **Run the script:**
   - Execute the script to start processing the live video feed and provide feedback commands.

   ```sh
   python drone_control.py
## Files

- `aruco_detection.py`: The main script that performs ArUco marker detection and pose estimation.
- `drone_control.py`: The script that provides feedback commands for drone movements based on positional and orientational errors.
- `aruco_data.csv`: CSV file containing information about detected markers (frame, ID, corner coordinates, distance, yaw, pitch, roll).
- `ArucoVideo.mp4`: Sample input video (you need to provide your own video or use a sample video named `ArucoVideo.mp4`).
- `annotated_video.mp4`: Output video with annotated ArUco markers.
- `C:\\New folder\\frame.png`: Path to the reference image used for drone control.

## Output

### Part A: Annotated Video

The script outputs an annotated video (`annotated_video.mp4`) where each detected ArUco marker is annotated with its ID and orientation axes (X, Y, Z).

### Part A: CSV File

The script also generates a CSV file (`aruco_data.csv`) containing the following columns:
- `Frame`: Frame number.
- `ID`: ArUco marker ID.
- `Corner1_X`, `Corner1_Y`, `Corner2_X`, `Corner2_Y`, `Corner3_X`, `Corner3_Y`, `Corner4_X`, `Corner4_Y`: Coordinates of the four corners of the marker.
- `Distance`: Distance from the camera to the marker.
- `Yaw`: Yaw angle of the marker.
- `Pitch`: Pitch angle of the marker.
- `Roll`: Roll angle of the marker.

### Part B: Feedback Commands

The `drone_control.py` script provides real-time feedback commands for drone movements displayed on the live video feed:
- **Move Right/Left/Up/Down**: Commands to adjust the position of the drone.
- **Move Forward/Backward**: Commands to adjust the distance of the drone.
- **Turn Right/Left**: Commands to adjust the orientation of the drone.
