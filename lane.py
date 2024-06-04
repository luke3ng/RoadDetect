import cv2
import numpy as np

def detectLanes(edges, frame):
    # Use Hough Line Transform to detect lines in the image
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=50)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else np.inf
            if abs(slope) > 0.4:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame

cap = cv2.VideoCapture('istockphoto-583789290-640_adpp_is.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Crop the top half of the frame
    height, width = frame.shape[:2]
    cropped_frame = frame[height//2:, :]

    # Convert the cropped frame to HSV color space
    hsv = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)

    # Define range of yellow color in HSV
    lower_yellow = np.array([15, 50, 100])
    upper_yellow = np.array([40, 255, 255])

    # Define range of white color in HSV
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 25, 255])

    # Create a mask. Threshold the HSV image to get only yellow and white colors
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask = cv2.bitwise_or(mask_yellow, mask_white)

    # Apply Gaussian Blur to the mask
    blurred_mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Use Canny edge detection
    edges = cv2.Canny(blurred_mask, 50, 150)

    # Detect lanes and draw on the cropped frame
    lineImage = detectLanes(edges, cropped_frame)

    # Place the processed cropped frame back into the original frame
    frame[height//2:, :] = lineImage

    cv2.imshow('Frame', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
