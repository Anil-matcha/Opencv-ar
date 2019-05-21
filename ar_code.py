import cv2, os
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import numpy as np
import math
from time import sleep

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

def get_match(frame, bbox, box):	
	#print(box)
	keypoint, descriptor = features(frame, extractor)
	histogram = build_histogram(descriptor, kmeans)
	neighbor = NearestNeighbors(n_neighbors = 3)
	neighbor.fit(preprocessed_image)
	dist, result = neighbor.kneighbors([histogram])
	predicted_file = files[result[0][0]]
	if predicted_file == "background.png":
		cv2.imshow('frame',frame)
		return
	predicted = cv2.imread(predicted_file)
	if predicted_file == "Avengers.jpg":
	    ret, vid_frame = video_av.read()
	elif predicted_file == "Samantha.jpg":
	    ret, vid_frame = video_sam.read()
	elif predicted_file == "Virat-Kohli.jpg":
	    ret, vid_frame = video_kohli.read()	
	src_pts = np.array([[vid_frame.shape[0], vid_frame.shape[1]], [0, vid_frame.shape[1]], [0, 0], [vid_frame.shape[0], 0]])
	homography, mask = cv2.findHomography(src_pts, bbox)		
	im_out = cv2.warpPerspective(vid_frame, homography, (frame.shape[1],frame.shape[0]))
	cv2.fillConvexPoly(frame, bbox.astype(int), 0, 16);
	frame = frame + im_out 
	cv2.imshow('frame', frame)
	
if __name__ == "__main__":
	extractor = cv2.ORB_create()
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	kmeans = KMeans(n_clusters = 800)	
	preprocessed_image = []
	files = [x for x in os.listdir() if "jpg" in x]+[x for x in os.listdir() if "png" in x]
	print(files)
	images = [cv2.imread(img) for img in files]
	descriptor_list = np.array([])
	video_av = cv2.VideoCapture('video_av.webm') 
	video_sam = cv2.VideoCapture('video_sam.mp4') 
	video_kohli = cv2.VideoCapture('video_kohli.mp4') 			
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
	
	cap = cv2.VideoCapture(0)
	frame_count = 0
	while(True):
		frame_count = frame_count + 1
		ret, frame = cap.read()
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		mask = cv2.inRange(hsv, (0, 0, 0), (180, 255,40))
		contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
		if(len(contours)==0):
			continue
		contours_sizes= [(cv2.contourArea(cnt), cnt, i) for i, cnt in enumerate(contours)]
		maxc = max(contours_sizes, key=lambda x: x[0])
		biggest_contour = maxc[1]
		rect = cv2.minAreaRect(biggest_contour)
		(x, y), (w, h), a = rect
		bbox = cv2.boxPoints(rect)
		box = np.int0(bbox)
		cv2.drawContours(frame,[box],0,(0,255,255),2)
		cv2.imshow("frame", frame)
		get_match(frame, bbox, (int(w), int(h)))	
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()