# ArUco Marker Detection and Pose Estimation

This project involves detecting ArUco markers in a video, estimating their pose, and annotating the video with the detected markers and their orientations. The information is also logged into a CSV file.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Files](#files)
- [Output](#output)
- [License](#license)

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
    pip install opencv-python numpy
    ```

## Usage

1. **Place your input video in the project directory:**
    Update the `video_path` variable in the script to point to your input video file.

2. **Run the script:**
    ```sh
    python aruco_detection.py
    ```

    This will process the video, detect ArUco markers, annotate the video, and generate a CSV file with the markers' data.

## Files

- `aruco_detection.py`: The main script that performs ArUco marker detection and pose estimation.
- `aruco_data.csv`: CSV file containing information about detected markers (frame, ID, corner coordinates, distance, yaw, pitch, roll).
- `ArucoVideo.mp4`: Sample input video (you need to provide your own video or use a sample video named `ArucoVideo.mp4`).
- `annotated_video.mp4`: Output video with annotated ArUco markers.

## Output

### Annotated Video
The script outputs an annotated video (`annotated_video.mp4`) where each detected ArUco marker is annotated with its ID and orientation axes (X, Y, Z).

### CSV File
The script also generates a CSV file (`aruco_data.csv`) containing the following columns:
- Frame: Frame number.
- ID: ArUco marker ID.
- Corner1_X, Corner1_Y, Corner2_X, Corner2_Y, Corner3_X, Corner3_Y, Corner4_X, Corner4_Y: Coordinates of the four corners of the marker.
- Distance: Distance from the camera to the marker.
- Yaw: Yaw angle of the marker.
- Pitch: Pitch angle of the marker.
- Roll: Roll angle of the marker.
