# app.py
from flask import Flask, Response, render_template
import cv2
import threading
import queue
import time
import cv2
import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

app = Flask(__name__)

frame_queue = queue.Queue(maxsize=30) # Global variables for frame handling
model = YOLO("best.pt")
video_path = 'armed_robbery_raw1.mp4'

def predict(im0):
    """Run prediction using a YOLO model for the input image im0."""
    results = model(im0)
    return results

def plot_bboxes(results, im0):
    """Plots bounding boxes on an image given detection results; returns annotated image, class IDs, and confidence scores."""
    class_ids = []
    confidences = []  # List to store confidence scores
    annotator = Annotator(im0, 3, results[0].names)

    boxes = results[0].boxes.xyxy.cpu()  # Bounding box coordinates
    clss = results[0].boxes.cls.cpu().tolist()  # Class labels (as indices)
    confs = results[0].boxes.conf.cpu().tolist()  # Confidence scores
    names = results[0].names  # Class names

    for box, cls, conf in zip(boxes, clss, confs):
        class_ids.append(cls)
        confidences.append(f"{conf:.2f}")  # Append confidence score formatted to 2 decimal places
        label = f"{names[int(cls)]} {conf:.2f}"  # Include confidence in the label
        annotator.box_label(box, label=label, color=colors(int(cls), True))

    return im0, class_ids, confidences

def process_video():
    """Background thread function to process video frames."""
    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
            continue

        # Process frame with YOLO detection
        results = predict(frame)
        frame, class_ids, confidences = plot_bboxes(results, frame)

        # Put frame in queue, remove oldest if full
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass

        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            pass

        time.sleep(0.005)  # Limit processing rate

def generate_frames():
    """Generate frames for the video feed."""
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.01)

@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start video processing in background thread
    video_thread = threading.Thread(target=process_video, daemon=True)
    video_thread.start()
    
    app.run(host='0.0.0.0', port=5000, threaded=True)