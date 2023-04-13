import cv2
from flask import Flask, render_template, Response
import Movenet_w_feedback
import time
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import helper
import cv2 as cv
import time
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
        plt.imsave(r"D:/Google-AMD-Hackathon-Capybara/Model/actual.jpg",frame)
        if not success:
            break
        else:
            # convert the frame to grayscale
            
            # yield the grayscale frame bytes to be displayed on the webpage
            gray = Movenet_w_feedback.compare_two()
            #print(type(output))
            # encode the grayscale frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', gray)
            # convert the buffer to bytes
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        #time.sleep(20)

@app.route('/')
def index():
    image_url = "images/bridge.jpg"
    return render_template('index.html', image_url=image_url)
    

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
