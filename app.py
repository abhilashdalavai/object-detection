from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
import os
import uuid
import time
import cv2
from ultralytics import YOLO

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = YOLO('yolov8l.pt')  # Change model if needed

latest_labels = []
video_source = None
webcam_on = False

# Control flags for video playback
is_paused = False
is_stopped = False


def generate_frames(source):
    global latest_labels, is_paused, is_stopped
    cap = cv2.VideoCapture(source)

    while cap.isOpened():
        if is_stopped:
            break

        if is_paused:
            time.sleep(0.1)
            continue

        success, frame = cap.read()
        if not success:
            break

        results = model(frame)
        boxes = results[0].boxes
        latest_labels = [model.names[int(cls)] for cls in boxes.cls]

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open("detection_log.txt", "a") as f:
            for label in set(latest_labels):
                f.write(f"[{timestamp}] Detected: {label}\n")

        annotated_frame = results[0].plot()
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    # Reset control flags when video ends or stopped
    is_stopped = False
    is_paused = False


@app.route('/', methods=['GET', 'POST'])
def index():
    global video_source, webcam_on, is_paused, is_stopped

    if request.method == 'POST':
        # Video upload
        if 'video_file' in request.files:
            file = request.files['video_file']
            if file.filename != '':
                filename = str(uuid.uuid4()) + ".mp4"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                video_source = filepath
                webcam_on = False
                is_paused = False
                is_stopped = False

        # Toggle webcam on/off
        elif 'toggle_webcam' in request.form:
            if webcam_on:
                video_source = None
                webcam_on = False
                is_paused = False
                is_stopped = False
            else:
                video_source = 0
                webcam_on = True
                is_paused = False
                is_stopped = False

        return redirect(url_for('index'))

    return render_template(
        'index.html',
        webcam_on=webcam_on,
        video_source=video_source,
    )


@app.route('/video')
def video():
    if video_source is None:
        return Response(status=204)
    return Response(generate_frames(video_source),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/labels')
def labels():
    return jsonify(list(set(latest_labels)))


@app.route('/control', methods=['POST'])
def control():
    global is_paused, is_stopped, video_source

    data = request.get_json()
    action = data.get('action') if data else None

    if action == 'pause':
        is_paused = True
    elif action == 'play':
        is_paused = False
    elif action == 'stop':
        is_stopped = True
        video_source = None  # clear video source to stop stream
        # Reset pause as well
        is_paused = False
        return jsonify({'redirect': url_for('index')})

    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(debug=True)