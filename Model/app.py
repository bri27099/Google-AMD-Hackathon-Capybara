import cv2
from flask import Flask, render_template, Response, jsonify
import Movenet_w_feedback
import time
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import helper
import cv2 as cv
import time
import threading
import io
import os
import base64
from PIL import Image
from flask import Flask, render_template, Response, redirect, url_for, session, request, send_file
app = Flask(__name__)
app.secret_key = 'my_secret_key'

ref_imgs = [ 'static/images/bridge.jpg','static/images/tree.jpg', 'static/images/cobra.png']
ref_imgs_int = [ 'static/images/boat.jpg','static/images/ddog.jpg', 'static/images/triangle.jpg']
ref_imgs_adv = [ 'static/images/dancer.jpg','static/images/crow.jpg', 'static/images/forward_bendC.jpg']

ref_imgs1 = ['static/images/tree.jpg', 'static/images/cobra.png',  'static/images/bridge.jpg']
ref_imgs2 = [ 'static/images/ddog.jpg', 'static/images/triangle.jpg','static/images/boat.jpg']
ref_imgs3 = [ 'static/images/crow.jpg', 'static/images/forward_bendC.jpg', 'static/images/dancer.jpg']

pose_text=["Setu Bandha Sarvangasana (Bridge)","Vrikshasana (Tree)","Bhujangasana (Cobra)"]
pose_text_int=["Paripurna Navasana (Boat)","Adho Mukha Shvanasana (Downward Dog)","Trikonasana (Triangle)"]
pose_text_adv=["Bakasana (Crow)","Prasarita Padottanasana C (Forward Bend C)","Natarajasana (Dancer)"]

page='beginner'
score = 0
i = 0
count = 0
my_int=0
text_disp=''
text=''
camera = cv2.VideoCapture(0)



def grayscale(frame):
    # convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray

def update_int():
    global my_int
    global text_disp
    global text
    while True:
        new_int = int(score)
        my_int = new_int
        text_disp = text
        print(text_disp)
        time.sleep(5)

def generate_frames():
    global i
    global page
    global score
    global text
    while True:
        # read the camera frame
        # convert the frame to grayscale
        success, frame = camera.read()
        plt.imsave(r"./actual.jpg",frame)
        # yield the grayscale frame bytes to be displayed on the webpage
        if (page=='beginner'):
            gray, score, text = Movenet_w_feedback.compare_two(ref_imgs[i%3])
        if (page=='intermediate'):
            gray, score, text = Movenet_w_feedback.compare_two(ref_imgs_int[i%3])
        if (page=='advanced'):
            gray, score, text = Movenet_w_feedback.compare_two(ref_imgs_adv[i%3])
        #print(score)
        #print('i inside gen_frame: ', i)

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
        img=cv.imread(".\static\images\bridge.png")
        frame_bytes = img.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/',methods=['POST','GET'])
def index():
    return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload():
#     # Get the image data from the POST request
#     img = request.files['image'].read()

#     # Decode the image data
#     nparr = np.fromstring(img, np.uint8)
#     frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#     # Save the image as 'actual.jpg'
#     cv2.imwrite('actual.jpg', frame)

#     # Return a response to the client
#     return 'Image uploaded successfully!'
@app.route('/capture', methods=['POST'])
def capture():
    image_data = request.form['image']
    encoded_data = image_data.split(',')[1]
    decoded_data = base64.b64decode(encoded_data)
    with open('actual1.jpg', 'wb') as f:
        f.write(decoded_data)
    #print(image_data)
    return 'Image saved successfully.'


@app.route('/home',methods=['POST','GET'])
def home():
    global i
    i=0
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
    global score
    global count
    with open(ref_imgs1[i%3], 'rb') as f:
        img = Image.open(io.BytesIO(f.read()))
        # img = img.transpose(Image.FLIP_LEFT_RIGHT)
        output = io.BytesIO()
        img.save(output, format='JPEG')
        output.seek(0)
        #print('i inside get_new_img: ', i)
    
        i+=1
        if i == 15:
            return 0
        count+=1
        if (count==150) or (count==0):
            return send_file(output, mimetype='image/jpeg')
        return send_file(output, mimetype='image/jpeg')
    

@app.route('/get_new_image1')
def get_new_image1():
    # Create a new image here, e.g. by opening a different file, processing the original image, etc.
    # For demonstration purposes, we'll just flip the original image horizontally.
    global i
    global score
    with open(ref_imgs2[i%3], 'rb') as f:
        img = Image.open(io.BytesIO(f.read()))
        # img = img.transpose(Image.FLIP_LEFT_RIGHT)
        output = io.BytesIO()
        img.save(output, format='JPEG')
        output.seek(0)
        #print('i inside get_new_img: ', i)
    
        i+=1
        if i == 15:
            return 0
        return send_file(output, mimetype='image/jpeg')
    
@app.route('/get_new_image2')
def get_new_image2():
    # Create a new image here, e.g. by opening a different file, processing the original image, etc.
    # For demonstration purposes, we'll just flip the original image horizontally.
    global i
    global score
    with open(ref_imgs3[i%3], 'rb') as f:
        img = Image.open(io.BytesIO(f.read()))
        # img = img.transpose(Image.FLIP_LEFT_RIGHT)
        output = io.BytesIO()
        img.save(output, format='JPEG')
        output.seek(0)
        #print('i inside get_new_img: ', i)
    
        i+=1
        if i == 15:
            return 0
        return send_file(output, mimetype='image/jpeg')
    

    
@app.route('/intermediate',methods=['POST','GET'])
def intermediate():
    global page
    global score
    page='intermediate'
    return render_template('index1.html')

@app.route('/advanced',methods=['POST','GET'])

def advanced():
    global page
    global score
    page='advanced'
    return render_template('index2.html')

@app.route('/beginner',methods=['POST','GET'])
def beginner():
    global page
    global score
    page='beginner'
    return render_template('index.html')

@app.route('/get_int')
def return_int():
    global my_int
    global text_disp
    return jsonify(my_int=my_int, my_text=get_text1(my_int),get_text=text_disp)

def get_text1(my_int):
    global i
    global page
    if (page=="beginner"):
        return (pose_text[i%3])
    if (page=="intermediate"):
        return (pose_text_int[i%3])
    if (page=="advanced"):
        return (pose_text_adv[i%3])


if __name__ == '__main__':
    t = threading.Thread(target=update_int)
    t.start()
    app.run(debug=True)
