import cv2
import numpy as np





cap = cv2.VideoCapture('istockphoto-583789290-640_adpp_is.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


    #blur image
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_yellow = np.array([15, 50, 180])
    upper_yellow = np.array([40, 255, 255])

    lower_white = np.array([0, 0, 186])
    upper_white = np.array([172, 111, 255])

    # Create a mask. Threshold the HSV image to get only yellow colors
    mask1 = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask2 = cv2.inRange(hsv, lower_white, upper_white)
    mask = mask2 | mask1
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('Frame', result)


    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()