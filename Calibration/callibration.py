import cv2
import numpy as np
import glob

# Define the dimensions of the chessboard
chessboard_size = (8, 6)  # This should match the pattern you printed
square_size = 20.0  # Size of a square in mm

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ..., (8,6,0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store object points and image points from all the images
objpoints = []  # 3d points in real-world space
imgpoints = []  # 2d points in image plane

# Load calibration images
images = glob.glob('images/*.png')  # Adjust the path to your calibration images

if not images:
    print('hello not working')
else:
    print('It is working')

if not images:
    print("No images found. Check the path to your images folder.")
    exit(1)

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Failed to load image {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(100)
    else:
        print(f"Chessboard corners not found in {fname}")

cv2.destroyAllWindows()

# Check if we have enough points for calibration
if not objpoints or not imgpoints:
    print("Not enough points were found for calibration.")
    exit(1)

# Calibrate the camera
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

#dist_coeffs = dist_coeffs.flatten().tolist()
# Save the calibration results for later use
calibration_data = {'camera_matrix': camera_matrix, 'dist_coeffs': dist_coeffs, 'rvecs': rvecs, 'tvecs': tvecs}
np.save('calibration_data.npy', calibration_data)

print("Camera matrix:\n", camera_matrix)
print("Distortion coefficients:\n", dist_coeffs)
