# Importing libraries
import torch
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

# Loading the model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")

# Reading the video
vid = cv2.VideoCapture("video2.mp4")
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)
fps = vid.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('counting.mp4', fourcc, fps,
                      (int(vid.get(3)), int(vid.get(4))))

# Using DeepSort as the tracker
object_tracker = DeepSort(max_age=30, n_init=2, nms_max_overlap=0.5, max_cosine_distance=0.3, nn_budget=100,
                          override_track_class=None, embedder="mobilenet", half=True, bgr=True, embedder_gpu=True,
                          embedder_model_name=None, embedder_wts=None, polygon=False, today=None)

ids = []


# Function to get detections
def detections(frame):
    result = model(frame)
    labels, cord = result.xyxyn[0][:, -1].to('cpu').numpy(), result.xyxyn[0][:, :-1].to('cpu').numpy()
    return labels, cord


# Function to plot boxes
def plot_boxes(results, frame, width, height, confidence=0.3):
    detections = []
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i, l in enumerate(labels):
        row = cord[i]
        if row[4].item() > confidence:
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                row[3] * y_shape)
            detections.append(([x1, y1, int(x2 - x1), int(y2 - y1)], row[4].item(), 'person'))
    return frame, detections


# Looping through the video
while vid:
    ret, frame = vid.read()

    if ret:
        # Detecting animals
        result = detections(frame)
        img, det = plot_boxes(result, frame, frame.shape[0], frame.shape[1], confidence=0.3)
        # Getting the tracks
        tracks = object_tracker.update_tracks(det, frame=img)

        # Looping through the tracks
        for track in tracks:
            track_id = track.track_id
            bbox = track.to_tlbr().astype(np.int32)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, str(track_id), (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2)
            if track_id not in ids:
                ids.append(track_id)

        cv2.putText(frame, f"No of sheeps: {str(len(ids))}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.imshow("img", frame)
        out.write(frame)
    else:
        break

vid.release()
