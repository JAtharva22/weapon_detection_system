# app.py
from flask import Flask, Response, render_template
import cv2
import threading
import queue
import time
from gun_det_app import ObjectDetection

app = Flask(__name__)

# Global variables for frame handling
frame_queue = queue.Queue(maxsize=30)

# Initialize object detection
sender_email = "manas.divekar76@gmail.com"
receiver_email = "me.atharvajadhav@gmail.com"
email_password = "jemu imks maqm kaow"
video_path = 'armed_robbery_raw1.mp4'

object_detection = ObjectDetection(
    video_path=video_path,
    sender_email=sender_email,
    receiver_email=receiver_email,
    email_password=email_password
)

def process_video():
    """Background thread function to process video frames."""
    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
            continue

        # Process frame with YOLO detection
        results = object_detection.predict(frame)
        frame, class_ids, confidences = object_detection.plot_bboxes(results, frame)

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

        time.sleep(0.003)  # Limit processing rate

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