# Importing Libraries
import cv2
import numpy as np
import torch

# Loading the model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")

# Initialize the Kalman filter
kf = cv2.KalmanFilter(4, 2)
kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kf.processNoiseCov = 1e-5 * np.eye(4, dtype=np.float32)
kf.measurementNoiseCov = 1e-1 * np.eye(2, dtype=np.float32)

# Initialize the velocity
velocity = np.zeros((2, 1), dtype=np.float32)

# Reading the video
vid = cv2.VideoCapture("zebra.mp4")
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)
fps = vid.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('velocity.mp4', fourcc, fps,
                      (int(vid.get(3)), int(vid.get(4))))

# Looping through the video
while vid:
    ret, frame = vid.read()

    if ret:
        # Detecting the animal
        results = model(frame)

        # Looping through the detections
        for box, name in zip(results.xyxy[0], results.pandas().xyxy[0]['name']):
            xB = int(box[2])
            xA = int(box[0])
            yB = int(box[3])
            yA = int(box[1])
            midX = (xB - xA) + xA
            midY = (yB - yA) + yA

            # Taking the current position
            measurement = np.array([[np.float32(midX)], [np.float32(midY)]])
            # Making a prediction for the next position
            prediction = kf.predict()
            # Passing current position
            kf.correct(measurement)

            # Accessing the velocity from the prediction
            velocity = np.array([kf.statePost[2], kf.statePost[3]], dtype=np.float32)
            velocity = [int(v) for v in velocity]

            # Drawing rectangles around the animal
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
            cv2.putText(frame, str(abs(velocity[0])), (xA, yA - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("img", frame)
            out.write(frame)

    else:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
