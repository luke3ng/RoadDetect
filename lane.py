import cv2
import numpy as np




def detectLanes(result, frame):
    # Convert the image to grayscale



    # Use Hough Line Transform to detect lines in the image
    lines = cv2.HoughLinesP(result, 1, np.pi / 180, threshold=1, minLineLength=10, maxLineGap=0)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            slope = (y2-y1)/(x2-x1)
            if(slope > .4 or slope <-.4):

              cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame

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
    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)


    lineImage = detectLanes(result,frame)


    cv2.imshow('Frame', frame)


    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()