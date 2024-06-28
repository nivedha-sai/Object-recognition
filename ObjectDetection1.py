import cv2
import numpy as np
from tracker import *

# Create tracker object
tracker = EuclideanDistTracker()

# Open video file
cap = cv2.VideoCapture("conveyor_belt.mp4")

# Initialize a counter for objects
object_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Define the region of interest (ROI) for the conveyor belt
    belt = frame[100:360, 280:]
    gray_belt = cv2.cvtColor(belt, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, threshold = cv2.threshold(gray_belt, 80, 255, cv2.THRESH_BINARY)

    # Detect contours of the cement packages
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)

        # Calculate the area of the contour
        area = cv2.contourArea(cnt)

        # Filter out small contours that may be noise and detect only larger objects
        if area > 19000:  # Adjust this threshold based on your needs
            detections.append([x, y, w, h])
            cv2.rectangle(belt, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(belt, str(object_counter), (x, y), 1, 1, (0, 255, 0))
            object_counter += 1

    # Update the tracker with the detections
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(threshold, str(id), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
        cv2.rectangle(belt, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the results
    cv2.imshow('Frame', frame)
    cv2.imshow("Belt", belt)
    cv2.imshow("Threshold", threshold)

    key = cv2.waitKey(1)
    if key == 30:  # Press 'Esc' to exit
        break