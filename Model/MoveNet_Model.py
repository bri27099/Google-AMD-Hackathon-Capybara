# Import TF and TF Hub libraries.
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
from helper import *
import cv2 as cv

# Load the input image.
image_path = r'C:\Users\brind\OneDrive\Pictures\Camera Roll\WIN_20230330_20_31_33_Pro.jpg'
image = tf.io.read_file(image_path)

# print(type(image))
# print(image.shape())
image = tf.compat.v1.image.decode_jpeg(image)

image = tf.expand_dims(image, axis=0)
# Resize and pad the image to keep the aspect ratio and fit the expected size.
image = tf.cast(tf.image.resize_with_pad(image, 192, 192), dtype=tf.int32)

# Download the model from TF Hub.
model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = model.signatures['serving_default']
keypoints_labels = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']

# # Run model inference.
# outputs = movenet(image)
# # Output is a [1, 1, 17, 3] tensor.
# keypoints = outputs['output_0']

# # Define the connections between keypoints
# print(outputs)
# print(keypoints)

# display_image = tf.expand_dims(image, axis=0)
# display_image = tf.cast(tf.image.resize_with_pad(
#     image, 1280, 1280), dtype=tf.int32)
# output_overlay = draw_prediction_on_image(
#     np.squeeze(display_image.numpy(), axis=0), keypoints)

# plt.figure(figsize=(5, 5))
# plt.imshow(output_overlay)
# _ = plt.axis('off')
# plt.show()


  

def predict_me(image_path):
    # image_path = r'C:\Users\brind\OneDrive\Pictures\Camera Roll\WIN_20230330_20_31_33_Pro.jpg'
    print(image_path)
    image = tf.io.read_file(image_path)
    image =tf.convert_to_tensor(image)
    # # print()
    image = tf.compat.v1.image.decode_jpeg(image)

    image = tf.expand_dims(image, axis=0)
    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    image = tf.cast(tf.image.resize_with_pad(image, 192, 192), dtype=tf.int32)

    # # Download the model from TF Hub.
    # # model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
    # # movenet = model.signatures['serving_default']
    keypoints_labels = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']

    # Run model inference.
    outputs = movenet(image)
    # Output is a [1, 1, 17, 3] tensor.
    keypoints = outputs['output_0']

    # Define the connections between keypoints
    print(outputs)
    print(keypoints)

    # display_image = tf.expand_dims(image, axis=0)
    display_image = tf.cast(tf.image.resize_with_pad(
        image, 1280, 1280), dtype=tf.int32)
    output_overlay = draw_prediction_on_image(
        np.squeeze(display_image.numpy(), axis=0), keypoints)
    print(type(output_overlay))
    return output_overlay

# exit()
# define a video capture object
vid = cv.VideoCapture(0)
i = 0  
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    file_path = r'C:\Users\brind\OneDrive\Desktop\hacku\our_yoga_fotos'+'\img_'+str(i)+'.jpg'
    cv.imwrite(file_path, frame)
    i+=1
    pred_img = predict_me(file_path)
    
    # Display the resulting frame
    cv.imshow('frame', frame)

    cv.imshow('pred_frame', pred_img)
    # output_path = r'C:\Users\brind\OneDrive\Desktop\hacku\our_yoga_outputs' +'\img_'+str(i)+'.jpg'
    output_path = r'C:\Users\brind\OneDrive\Desktop\hacku\points_on_blank' +'\img_'+str(i)+'.jpg'

    cv.imwrite(output_path, pred_img)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    if i == 100:
        break

    
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()