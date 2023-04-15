# Import TF and TF Hub libraries.
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import helper
import cv2 as cv

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

IMG_SIZE = 0
# Load the input image.
# image_path = r'C:\Users\brind\OneDrive\Pictures\Camera Roll\WIN_20230330_20_31_33_Pro.jpg'
# image_path = r'/home/ayush/Desktop/Manipal/Google-AMD-Hackathon-Capybara/Pictures/Screenshot from 2023-03-07 00-20-05.jpg'

# image = tf.io.read_file(image_path)

# # print(type(image))
# # print(image.shape())
# image = tf.compat.v1.image.decode_jpeg(image)

# image = tf.expand_dims(image, axis=0)
# # Resize and pad the image to keep the aspect ratio and fit the expected size.
# image = tf.cast(tf.image.resize_with_pad(image, 192, 192), dtype=tf.int32)

# Download the model from TF Hub.
model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = model.signatures['serving_default']
keypoints_labels = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']

# Run model inference.
# print(image.shape)


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
path1 = r"D:/Google-AMD-Hackathon-Capybara/Model/actual.jpg"
# path2 = r"D:/Google-AMD-Hackathon-Capybara/Model/sample.jpg"


'''
def get_feedback(shape, output_overlay1, kp1, output_overlay2, kp2):


    # kp1 = get_keypoints(shape, keypoints1)
    # kp2 = get_keypoints(shape, keypoints2)

    feedback_img = output_overlay1.copy()
    
    keypoints_to_label = ['left_shoulder', 'right_shoulder', 'left_wrist', 'right_wrist', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
    i = 0
    for k1, k2, in zip(kp1, kp2):
        if keypoints_labels[i] not in keypoints_to_label:
            i+=1
            continue
        start_point, end_point = (int(k1[0]) - 80, int(k1[1]) - 60), (int(k2[0]) - 80, int(k2[1]) - 60)
        feedback_img = cv.arrowedLine(feedback_img, start_point, end_point, 
                    (255, 0, 0) , 13, tipLength = 0.2) 
        feedback_img = cv.putText(feedback_img, '100', (600, 600), cv.FONT_HERSHEY_SIMPLEX, 50, (255, 0, 0), 2, cv.LINE_AA)
        i+=1
    # plt.figure(figsize=(5, 5))
    # plt.imshow(output_overlay1)
    # _ = plt.axis('off')
    # plt.show()

    # print(kp1)
    # print(kp2)
    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(8, 8)
    fig.set_dpi(200)

    # Display the images on each subplot
    off = 200
    # axs[0].imshow(output_overlay1[off:-off, :], cmap='viridis')
    # axs[1].imshow(output_overlay2[off:-off, :], cmap='viridis')
    # axs[2].imshow(feedback_img[off:-off, :], cmap='viridis')

    # Add titles to each subplot
    axs[0].set_title('You')
    axs[1].set_title('Them')
    axs[2].set_title('Feedback')
    
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()'''

def get_roi(image_roi, keypoints):
    min_x, min_y = np.min(keypoints, axis=0)
    max_x, max_y = np.max(keypoints, axis=0)
    #print('In get_roi, img_shape = ', image_roi.shape)
    roi = image_roi[int(min_y):int(max_y), int(min_x):int(max_x)]
    #plt.imshow(cv.rectangle(image_roi, (int(min_x),int(min_y)), (int(max_x),int(max_y)), (255, 0, 0), 3))
    #plt.show()

    #print(int(min_y),':',int(max_y),'|', int(min_x),':',int(max_x))
    return roi   

def eucl_dist(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5


def score_frame(kp1, kp2):
    global IMG_SIZE
    IMG_SIZE = 1280
    #print('IMG_SIZE = ', IMG_SIZE)
    i = 0

    scores = []
    max_val = 0
    for k1, k2, in zip(kp1, kp2):
        dist = eucl_dist(k1, k2)
        norm_dist = (1/dist) / (1/IMG_SIZE)
        # avg_score += norm_dist
        scores.append(norm_dist)

       # print('Score: ', norm_dist)
    
    avg_score = 0

    norm_scores = []
    for score in scores:
        norm_score = max(1, score/45)
        norm_scores.append(norm_score)
        avg_score += norm_score


    # avg_score = round(avg_score/len(keypoints_labels) * 100, 2)
   # print('Average score for frame: ', avg_score/len(keypoints_labels))
    return (avg_score/len(keypoints_labels) * 100)

    
def compare_two(path2):
    global IMG_SIZE
    image1 = tf.io.read_file(path1)
    image2 = tf.io.read_file(path2)
    print(path2)
    image1 = tf.compat.v1.image.decode_jpeg(image1)
    image2 = tf.compat.v1.image.decode_jpeg(image2)

    img1, img2 = np.array(image1), np.array(image2)

    IMG_SIZE = img2.shape[0]

    image1 = tf.expand_dims(image1, axis=0)
    image1 = tf.cast(tf.image.resize_with_pad(image1, 192, 192), dtype=tf.int32)


    outputs1= movenet(image1)
    keypoints1 = outputs1['output_0']

    # print(outputs1)
    # print(keypoints1)

    display_image1 = tf.expand_dims(image1, axis=0)
    display_image1 = tf.cast(tf.image.resize_with_pad(
        image1, 1280, 1280), dtype=tf.int32)

    shape = np.squeeze(display_image1.numpy()).shape

    # print('here', keypoints1)
    # kp1 = get_keypoints(shape, keypoints1)



    output_overlay1 = helper.draw_prediction_on_image(
        np.squeeze(display_image1.numpy(), axis=0), keypoints1)

    # plt.figure(figsize=(5, 5))    
    # plt.imshow(output_overlay1)
    # _ = plt.axis('off')
    # plt.show()

    image2 = tf.expand_dims(image2, axis=0)
    image2 = tf.cast(tf.image.resize_with_pad(image2, 192, 192), dtype=tf.int32)

    outputs2= movenet(image2)
    keypoints2 = outputs2['output_0']

    # print(outputs2)
    # print(keypoints2)

    display_image2 = tf.expand_dims(image2, axis=0)
    display_image2 = tf.cast(tf.image.resize_with_pad(
        image2, 1280, 1280), dtype=tf.int32)
    output_overlay2 = helper.draw_prediction_on_image(
        np.squeeze(display_image2.numpy(), axis=0), keypoints2)

    # plt.figure(figsize=(5, 5))
    # plt.imshow(output_overlay2)
    # _ = plt.axis('off')
    # plt.show()


    shape = np.squeeze(display_image2.numpy()).shape
    kp1 = get_keypoints(shape, keypoints1)
    kp2 = get_keypoints(shape, keypoints2)
    score = score_frame(kp1,kp2)
    feedback_img = output_overlay1.copy()
    
    keypoints_to_label = ['left_shoulder', 'right_shoulder', 'left_wrist', 'right_wrist', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
    i = 0
    for k1, k2, in zip(kp1, kp2):
        if keypoints_labels[i] not in keypoints_to_label:
            i+=1
            continue
        start_point, end_point = (int(k1[0]) - 80, int(k1[1]) - 60), (int(k2[0]) - 80, int(k2[1]) - 60)
        feedback_img = cv.arrowedLine(feedback_img, start_point, end_point, 
                    (255, 0, 0) , 13, tipLength = 0.2) 
        #feedback_img = cv.putText(feedback_img, '100', (600, 600), cv.FONT_HERSHEY_SIMPLEX, 50, (255, 0, 0), 2, cv.LINE_AA)
        i+=1
    feedback_img = cv.putText(feedback_img, path2[path2.index('/')+1: -3], (300, 300), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
    # plt.figure(figsize=(5, 5))
    # plt.imshow(output_overlay1)
    # _ = plt.axis('off')
    # plt.show()

    # print(kp1)
    # print(kp2)
    # fig, axs = plt.subplots(1, 3)
    # fig.set_size_inches(8, 8)
    # fig.set_dpi(200)

    # Display the images on each subplot
    # axs[0].imshow(output_overlay1[off:-off, :], cmap='viridis')
    # axs[1].imshow(output_overlay2[off:-off, :], cmap='viridis')
    # axs[2].imshow(feedback_img[off:-off, :], cmap='viridis')

    # Add titles to each subplot
    # axs[0].set_title('You')
    # axs[1].set_title('Them')
    # axs[2].set_title('Feedback')
    
    # for ax in axs.flat:
    #     ax.set_xticks([])
    #     ax.set_yticks([])

    return feedback_img, score
    
    #exit()
def get_keypoints(
    shape, keypoints_with_scores):

    height, width, channel = shape
    #print('height, width', (height, width),'\ngetting keypoints')
    aspect_ratio = float(width) / height


    (keypoint_locs, keypoint_edges,
    edge_colors) = helper._keypoints_and_edges_for_display(
       keypoints_with_scores, height, width)
    # print(keypoint_locs)
    return keypoint_locs



#compare_two(image1, image2)

  
# def predict_me(image_path):
#     # image_path = r'C:\Users\brind\OneDrive\Pictures\Camera Roll\WIN_20230330_20_31_33_Pro.jpg'
#     print(image_path)
#     image = tf.io.read_file(image_path)
#     # image = cv.imread(image_path)

#     image =tf.convert_to_tensor(image)
#     # # print()
#     image = tf.compat.v1.image.decode_jpeg(image)

#     image = tf.expand_dims(image, axis=0)
#     # Resize and pad the image to keep the aspect ratio and fit the expected size.
#     image = tf.cast(tf.image.resize_with_pad(image, 192, 192), dtype=tf.int32)

#     # # Download the model from TF Hub.
#     # # model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
#     # # movenet = model.signatures['serving_default']

#     # Run model inference.
#     outputs = movenet(image)
#     print('predicted successfully')
#     # Output is a [1, 1, 17, 3] tensor.
#     keypoints = outputs['output_0']

#     # Define the connections between keypoints
#     # print(outputs)
#     # print(keypoints)

#     # display_image = tf.expand_dims(image, axis=0)
#     display_image = tf.cast(tf.image.resize_with_pad(
#         image, 1280, 1280), dtype=tf.int32)
#     output_overlay = helper.draw_prediction_on_image(
#         np.squeeze(display_image.numpy(), axis=0), keypoints)
#     print('sucfully got output_overlay2')


#     # plt.imshow(output_overlay)
#     # plt.show()
#     # output_overlay = cv.resize(output_overlay, (192, 192))
#     # print(output_overlay.shape)
#     # plt.imshow(output_overlay)
#     # plt.show()

#     # cv.imshow('sds',np.zeros_like(output_overlay))
#     # cv.waitKey(0)
#     return output_overlay

# # define a video capture object
# vid = cv.VideoCapture(0)
# i = 0  
# while(True):
      
#     # Capture the video frame
#     # by frame
#     ret, frame = vid.read()
#     file_path = r'/home/ayush/Desktop/Manipal/Google-AMD-Hackathon-Capybara/Pictures/our_yoga_fotos/'+'img_'+str(i)+'.jpg'
#     # cv.imwrite(file_path, cv.resize(frame, (192, 192)))
#     print('To predict file at: ', file_path)
#     # pred_img = predict_me(file_path)
#     print('still here.')
#     # Display the resulting frame
#     # cv.imshow('frame', frame)

#     # print(pred_img)
#     # print(pred_img.shape)

#     # cv.imshow('pred_frame', pred_img)
#     # # output_path = r'C:\Users\brind\OneDrive\Desktop\hacku\our_yoga_outputs' +'\img_'+str(i)+'.jpg'
#     output_path = r'/home/ayush/Desktop/Manipal/Google-AMD-Hackathon-Capybara/Pictures/points_on_blank/' +'img_'+str(i)+'.jpg'

#     # cv.imwrite(output_path, pred_img)
#     # plt.imsave(output_path, pred_img)
#     i+=1

#     # the 'q' button is set as the
#     # quitting button you may use any
#     # desired button of your choice
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break

#     if i == 100:
#         break

    
  
# # After the loop release the cap object
# vid.release()
# # Destroy all the windows
# cv2.destroyAllWindows()