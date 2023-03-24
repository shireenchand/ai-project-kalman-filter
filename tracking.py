# Importing Libraries
from kalmanfilter import KalmanFilter
import cv2
import torch

# Loading Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")

# Defining Kalman Filter
kf = KalmanFilter()

# Reading the video
vid = cv2.VideoCapture("giraffe.mp4")
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)
fps = vid.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('outpy.mp4', fourcc, fps,
                      (int(vid.get(3)), int(vid.get(4))))

# Looping through the video
while vid:
    # Reading the frame
    ret, frame = vid.read()

    if ret:
        # Detecting the animal in the video
        results = model(frame)
        # Looping through the detections
        for box, name in zip(results.xyxy[0], results.pandas().xyxy[0]['name']):
            xB = int(box[2])
            xA = int(box[0])
            yB = int(box[3])
            yA = int(box[1])
            midX = (xB - xA) + xA
            midY = (yB - yA) + yA

            # Drawing the rectangle around the animal
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

            # Predicting the next coordinates
            predX, predY = kf.predict(midX, midY)

            # Drawing the line to show predicted direction
            cv2.arrowedLine(frame, (midX, midY), (predX, predY), (0, 0, 255), 2)
            cv2.putText(frame, name, (xA, yA - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("img", frame)
            out.write(frame)

    else:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
