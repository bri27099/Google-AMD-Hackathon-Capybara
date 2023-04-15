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
import io
from PIL import Image
from flask import Flask, render_template, Response, redirect, url_for, session, request, send_file
app = Flask(__name__)
app.secret_key = 'my_secret_key'

ref_imgs = [ 'static/images/bridge.jpg','static/images/tree.jpg', 'static/images/cobra.png']
ref_imgs1 = ['static/images/tree.jpg', 'static/images/cobra.png',  'static/images/bridge.jpg']

i = 0

camera = cv2.VideoCapture(0)

def grayscale(frame):
    # convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray

def generate_frames():
    global i
    while True:
        # read the camera frame
        success, frame = camera.read()
        plt.imsave(r"D:/Google-AMD-Hackathon-Capybara/Model/actual.jpg",frame)
        if not success:
            break
        else:
            # convert the frame to grayscale
            
            # yield the grayscale frame bytes to be displayed on the webpage
            gray = Movenet_w_feedback.compare_two(ref_imgs[i%3])
            print('i inside gen_frame: ', i)

            #print(type(output))
            # encode the grayscale frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', gray)
            # convert the buffer to bytes
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        #time.sleep(20)

def give_bro():
    while True:
        img=cv.imread("D:\Google-AMD-Hackathon-Capybara\Model\static\images\bridge.png")
        frame_bytes = img.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/',methods=['POST','GET'])
def index():
    return render_template('index.html')

@app.route('/butt',methods=['POST','GET'])
def home():
    return render_template('home.html')

@app.route('/reference_img',methods=['POST','GET'])
def reference_img():
    image_url = "images/bridge.jpg"
    Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/change_image', methods=['POST'])
def change_image():
    # Create a new image here, e.g. by opening a different file, processing the original image, etc.
    # For demonstration purposes, we'll just flip the original image horizontally.
    with open('static/images/tree.jpg', 'rb') as f:
        img = Image.open(io.BytesIO(f.read()))
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        output = io.BytesIO()
        img.save(output, format='JPEG')
        output.seek(0)
        
        # Store the new image in the session
        session['new_image'] = 'static/images/tree.jpg'
        
        # Redirect back to the index page
        return redirect(url_for('index'))

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/get_new_image')
def get_new_image():
    # Create a new image here, e.g. by opening a different file, processing the original image, etc.
    # For demonstration purposes, we'll just flip the original image horizontally.
    global i
    with open(ref_imgs1[i%3], 'rb') as f:
        img = Image.open(io.BytesIO(f.read()))
        # img = img.transpose(Image.FLIP_LEFT_RIGHT)
        output = io.BytesIO()
        img.save(output, format='JPEG')
        output.seek(0)
        print('i inside get_new_img: ', i)
    
        i+=1
        if i == 15:
            return 0
        return send_file(output, mimetype='image/jpeg')
    
@app.route('/intermediate',methods=['POST','GET'])
def intermediate():
    return render_template('index1.html')

@app.route('/advanced',methods=['POST','GET'])
def advanced():
    return render_template('index1.html')

@app.route('/beginner',methods=['POST','GET'])
def beginner():
    return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True)
