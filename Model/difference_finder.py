# ##################SSIM score###############################

# import cv2
# import skimage
# # from skimage.measure import compare_ssim
# from skimage.metrics import structural_similarity as compare_ssim


# # Load the detected downward dog image
# detected_img = cv2.imread(r'C:\Users\brind\OneDrive\Desktop\hacku\DATASET\TRAIN\tree\00000098.jpg')

# # Load the correct downward dog image
# correct_img = cv2.imread(r'C:\Users\brind\OneDrive\Desktop\hacku\DATASET\TRAIN\goddess\00000100.jpg')

# # Resize both images to the same size
# detected_img = cv2.resize(detected_img, (224, 224))
# correct_img = cv2.resize(correct_img, (224, 224))

# # Convert both images to grayscale
# detected_gray = cv2.cvtColor(detected_img, cv2.COLOR_BGR2GRAY)
# correct_gray = cv2.cvtColor(correct_img, cv2.COLOR_BGR2GRAY)

# # Calculate the SSIM between the detected and correct images
# (score, diff) = compare_ssim(detected_gray, correct_gray, full=True)
# diff = (diff * 255).astype("uint8")

# # Print the SSIM score and display the difference image
# print("SSIM score: {}".format(score))
# if score<0.7:
#     print("diff poses biyatchhhhhhhhhhhhhhhhh")
# else:
#     print("same pose loser")
# cv2.imshow("Detected vs Correct", diff)
# cv2.waitKey(0)
# cv2.destroyAllWindows()








#####################cosine similarity#####################################

# import cv2
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

# # Load the two images to be compared
# detected_img = cv2.imread(r'C:\Users\brind\OneDrive\Desktop\hacku\DATASET\TRAIN\tree\00000098.jpg')

# # # Load the correct downward dog image
# correct_img = cv2.imread(r'C:\Users\brind\OneDrive\Desktop\hacku\DATASET\TRAIN\goddess\00000115.jpg')

# # Resize the images to a common size
# img_detected = cv2.resize(detected_img, (224, 224))
# img_correct = cv2.resize(correct_img, (224, 224))

# # Convert the images to grayscale
# gray_detected = cv2.cvtColor(img_detected, cv2.COLOR_BGR2GRAY)
# gray_correct = cv2.cvtColor(img_correct, cv2.COLOR_BGR2GRAY)

# # Flatten the grayscale images into a 1D array
# flat_detected = gray_detected.flatten()
# flat_correct = gray_correct.flatten()

# # Compute the cosine similarity between the two images
# similarity = cosine_similarity(flat_detected.reshape(1,-1), flat_correct.reshape(1,-1))

# # Print the similarity score
# print("Cosine similarity score: ", similarity)
# if similarity<0.9:
#     print("diff poses biyatchhhhhhhhhhhhhhhhh")
# else:
#     print("same pose loser")

# # Display the two images side by side
# img_display = np.hstack((img_detected, img_correct))
# cv2.imshow('Detected pose vs Correct pose', img_display)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



################sift######################




import cv2

# Load the two images to be compared
img_detected = cv2.imread(r'C:\Users\brind\OneDrive\Desktop\hacku\points_on_blank\img_60.jpg')
img_correct = cv2.imread(r'C:\Users\brind\OneDrive\Desktop\hacku\tree_pts.png')[60:-60, 60:-60]



img_correct = cv2.resize(img_correct, (500, 500))
img_detected = cv2.resize(img_detected, (500, 500))


# Convert the images to grayscale
gray_detected = cv2.cvtColor(img_detected, cv2.COLOR_BGR2GRAY)
gray_correct = cv2.cvtColor(img_correct, cv2.COLOR_BGR2GRAY)

# Create a SIFT object
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors for both images
kp_detected, des_detected = sift.detectAndCompute(gray_detected, None)
kp_correct, des_correct = sift.detectAndCompute(gray_correct, None)

# Match keypoints between the two images using a FLANN-based matcher
flann = cv2.FlannBasedMatcher()
matches = flann.knnMatch(des_detected, des_correct, k=2)

# Apply ratio test to filter out poor matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Draw the matched keypoints between the two images
img_matches = cv2.drawMatches(img_detected, kp_detected, img_correct, kp_correct, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the matched keypoints image
cv2.imshow('Matched Keypoints', img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()




######################surf

# import cv2
# import numpy as np

# # Load the two images
# img1 = cv2.imread(r'C:\Users\brind\OneDrive\Desktop\hacku\DATASET\TRAIN\tree\00000098.jpg', cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread(r'C:\Users\brind\OneDrive\Desktop\hacku\DATASET\TRAIN\tree\00000099.jpg', cv2.IMREAD_GRAYSCALE)

# # Initialize the SURF feature detector
# surf = cv2.xfeatures2d.SURF_create()

# # Detect keypoints and descriptors in both images
# kp1, des1 = surf.detectAndCompute(img1, None)
# kp2, des2 = surf.detectAndCompute(img2, None)

# # Initialize the BFMatcher
# bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# # Match the keypoints in both images
# matches = bf.match(des1, des2)

# # Sort the matches by distance
# matches = sorted(matches, key=lambda x: x.distance)

# # Calculate the Euclidean distance between the matched keypoints
# distances = []
# for match in matches:
#     img1_idx = match.queryIdx
#     img2_idx = match.trainIdx
#     (x1, y1) = kp1[img1_idx].pt
#     (x2, y2) = kp2[img2_idx].pt
#     distances.append(np.sqrt((x1 - x2)**2 + (y1 - y2)**2))

# # Print the distances for the first 10 matches
# print(distances[:10])






