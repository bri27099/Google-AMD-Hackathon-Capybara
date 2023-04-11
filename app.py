import cv2
from flask import Flask, render_template, Response

app = Flask(__name__)

camera = cv2.VideoCapture(0)

def grayscale(frame):
    # convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray

def generate_frames():
    while True:
        # read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
            # convert the frame to grayscale
            gray = grayscale(frame)
            # encode the grayscale frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', gray)
            # convert the buffer to bytes
            frame_bytes = buffer.tobytes()
            # yield the grayscale frame bytes to be displayed on the webpage
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
