import cv2, os
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import numpy as np
import math
from time import sleep
extractor = cv2.ORB_create()

def build_histogram(descriptor_list, cluster_alg):
    histogram = np.zeros(len(cluster_alg.cluster_centers_))
    cluster_result =  cluster_alg.predict(descriptor_list)
    for i in cluster_result:
        histogram[i] += 1.0
    return histogram

def gray(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_img	
	
def features(image, extractor):
    keypoints, descriptors = extractor.detectAndCompute(image, None)
    return keypoints, descriptors

	
kmeans = KMeans(n_clusters = 800)	
preprocessed_image = []
files = [x for x in os.listdir() if "jpg" in x]+[x for x in os.listdir() if "png" in x]
print(files)
images = [cv2.imread(img) for img in files]
descriptor_list = np.array([])
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
for image in images:
	image = gray(image)
	keypoint, descriptor = features(image, extractor)
	if len(descriptor_list) == 0:
		descriptor_list = np.array(descriptor)
	else:
		descriptor_list = np.vstack((descriptor_list, descriptor))
kmeans.fit(descriptor_list)	  
for image in images:
      image = gray(image)
      keypoint, descriptor = features(image, extractor)
      if (descriptor is not None):
          histogram = build_histogram(descriptor, kmeans)
          preprocessed_image.append(histogram)	

camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])

def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)
 
im_dst = cv2.imread("pred/Virat-Kohli.jpg")
video_av = cv2.VideoCapture('video_av.webm') 
video_sam = cv2.VideoCapture('video_sam.mp4') 
video_kohli = cv2.VideoCapture('video_kohli.mp4') 	
def get_match(frame):	
	keypoint, descriptor = features(frame, extractor)
	histogram = build_histogram(descriptor, kmeans)
	neighbor = NearestNeighbors(n_neighbors = 3)
	neighbor.fit(preprocessed_image)
	dist, result = neighbor.kneighbors([histogram])
	print(result, dist)
	print([files[i] for i in result[0]])
	predicted_file = files[result[0][0]]
	print(predicted_file)
	if predicted_file == "background.png":
		cv2.imshow('frame',frame)
		return
	predicted = cv2.imread(predicted_file)
	keypoint2, descriptor2 = features(predicted, extractor)	
	matches = bf.match(descriptor, descriptor2)
	matches = sorted(matches, key=lambda x: x.distance)
	print(len(matches), len(keypoint), len(keypoint2), len(descriptor), len(descriptor2))
	src_pts = np.float32([keypoint[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
	dst_pts = np.float32([keypoint2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
	#cv2.imshow('input', cv2.drawKeypoints(frame,keypoint,None,color=(0,255,0), flags=0))
	#cv2.imshow('predicted', cv2.drawKeypoints(predicted,keypoint2,None,color=(0,255,0), flags=0))	
	imMatches = cv2.drawMatches(frame, keypoint, predicted, keypoint2, matches, None)
	#cv2.imshow("matches", imMatches)
	homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
	if predicted_file == "Avengers.jpg":
	    ret, frame = video_av.read()
	elif predicted_file == "Samantha.jpg":
	    ret, frame = video_sam.read()
	elif predicted_file == "Virat-Kohli.jpg":
	    ret, frame = video_kohli.read()		
	im_out = cv2.warpPerspective(frame, homography, (predicted.shape[1],predicted.shape[0]))
	cv2.imshow('frame', im_out)
	print(homography)
#get_match(im_dst)	
cap = cv2.VideoCapture(0)
frame_count = 0
while(True):
	frame_count = frame_count + 1
	ret, frame = cap.read()
	gray_img = gray(frame)
	get_match(frame)	
	sleep(0.1)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()