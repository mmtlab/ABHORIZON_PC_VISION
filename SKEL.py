#!/usr/bin/env python

import sender
import mediapipe as mp
import cv2
import os
import time
import configparser
import numpy as np
import socket
import imutils
import csv





def ex_string_to_ID(ex_string):
    # read from ex_info
    config = configparser.ConfigParser()
    config.read('exercise_info.ini')
    sections = config.sections()
    # print("sections are : {}".format(sections))
    ex_string=ex_string.rstrip("\n")
    for exercise in sections:

        if exercise == ex_string:
            # config.get("test", "foo")

            ID = int(config.get(exercise, 'ID'))
            return ID

    print("no exercise found")
    ID = 0
    return ID





def warpImages(img1, img2, H):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    list_of_points_1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    temp_points = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)

    # When we have established a homography we need to warp perspective
    # Change field of view
    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)

    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

    translation_dist = [-x_min, -y_min]

    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max - x_min, y_max - y_min))
    output_img[translation_dist[1]:rows1 + translation_dist[1], translation_dist[0]:cols1 + translation_dist[0]] = img1

    return output_img


def draw_matches(img1, keypoints1, img2, keypoints2, matches):
    r, c = img1.shape[:2]
    r1, c1 = img2.shape[:2]

    # Create a blank image with the size of the first image + second image
    output_img = np.zeros((max([r, r1]), c + c1, 3), dtype='uint8')
    output_img[:r, :c, :] = np.dstack([img1, img1, img1])
    output_img[:r1, c:c + c1, :] = np.dstack([img2, img2, img2])

    # Go over all of the matching points and extract them
    for match in matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        (x1, y1) = keypoints1[img1_idx].pt
        (x2, y2) = keypoints2[img2_idx].pt

        # Draw circles on the keypoints
        cv2.circle(output_img, (int(x1), int(y1)), 4, (0, 255, 255), 1)
        cv2.circle(output_img, (int(x2) + c, int(y2)), 4, (0, 255, 255), 1)

        # Connect the same keypoints
        cv2.line(output_img, (int(x1), int(y1)), (int(x2) + c, int(y2)), (0, 255, 255), 1)

    return output_img


def align_image_panorama(img1,img2):


    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #cv2.imshow("q",img1_gray)
    #cv2.imshow("qdd",img2_gray)
    # Create our ORB detector and detect keypoints and descriptors
    orb = cv2.ORB_create(nfeatures=2000)

    # Find the key points and descriptors with ORB
    keypoints1, descriptors1 = orb.detectAndCompute(img1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2_gray, None)
    #cv2.imshow("qnjnq",cv2.drawKeypoints(img1, keypoints1, None, (255, 0, 255))
    #cv2.imshow("qqjbi",cv2.drawKeypoints(img2, keypoints2, None, (255, 0, 255)))
    # Create a BFMatcher object.
    # It will find all of the matching keypoints on two images
    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)

    # Find matching points
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    #print(keypoints1[0].pt)
    #print(keypoints1[0].size)
    #print("Descriptor of the first keypoint: ")
    #print(descriptors1[0])
    #print(matches_sorted[0].distance)
    '''
    all_matches = []
    #find all match
    for m, n in matches:
        all_matches.append(m)

    img3 = draw_matches(img1_gray, keypoints1, img2_gray, keypoints2, all_matches[:30])
    cv2.imshow("rr",img3)
    '''
    # Finding the best matches 0.6 threshold
    good = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good.append(m)
    #cv2.imshow("1_withkp",cv2.drawKeypoints(img1, [keypoints1[m.queryIdx] for m in good], None, (255, 0, 255)))
    #cv2.imshow("2_withkp",cv2.drawKeypoints(img2, [keypoints2[m.trainIdx] for m in good], None, (255, 0, 255)))

    # Set minimum match condition
    MIN_MATCH_COUNT = 10

    if len(good) > MIN_MATCH_COUNT:
        # Convert keypoints to an argument for findHomography
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Establish a homography
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        write_data(M)

        result = warpImages(img2, img1, M)# da chiamare ogni frame
        return result










def cv_stitcher(img1,img2):
    images = []


    images.append(img1)
    images.append(img2)

    # use built in stitcher
    stitcher = cv2.createStitcher()
    (status, stitched) = stitcher.stitch(images);
    return stitched


def alignImages(imPost, imPre, max_features, good_match_percent):
    #cri function
    ''' Function that performs the alignment between two images according to a set
    of features detected in both images that it tries to match automatically.
    It computes the homography between the two images using the keypoints detected
    and filtered according to the good_match_percent set by the user.

    INPUTS:
    - imPost: image to align to the second one. In the app it should be the POST image
    - imPre: reference image used as template. In the app it should be the PRE image
    - max_features: int number representing the number of ORB features to find.
                    Since these features get filtered later on it is best to set
                    this value around 200+ to ensure a good number of features to use
    - good_match_percent: a number < 0 representing the threshold to keep or discard a feature.
                          If lower, less features are kept.
    - debug: flag used to visualize image outputs
    - log: logger needed to print logs on terminal

    OUTPUTS:
    - imPostReg: the original POST image warped according to the computed homography
    - h: the computed homography matrix
    '''

    #log.debug('Callign alignment algorithm with max features to detect = ' + str(max_features) + ' and good match % = ' + str(good_match_percent))

    imPostGray = cv2.cvtColor(imPost, cv2.COLOR_BGR2GRAY)
    imPreGray = cv2.cvtColor(imPre, cv2.COLOR_BGR2GRAY)

    # detect ORB features and compute descriptors
    orb = cv2.ORB_create(max_features)
    keypoints1, descriptors1 = orb.detectAndCompute(imPostGray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(imPreGray, None)

    # match features
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # remove not so good matches
    # since matches is now sorted according to score (features with a lower match score are
    # near the bottom of the array), this filtering is easily done by selecting only
    # the first % of matches according to good_match_percent
    numGoodMatches = int(len(matches) * good_match_percent)
    matches = matches[:numGoodMatches]


    # draw the filtered matches on the images
    imMatches = cv2.drawMatches(imPost, keypoints1, imPre, keypoints2, matches, None)
    #visualize('matches', imMatches, (100, 100))

    # extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # use homography to align the POST image to the PRE one
    height, width, channels = imPre.shape
    imPostReg = cv2.warpPerspective(imPost, h, (width, height))

   # if debug == True:
        #visualize('aligned', imPostReg, (100, 100))

    return imPostReg,imMatches, h

def write_data(data):
    f = open('homografy.csv', 'a')
    writer = csv.writer(f)

    writer.writerow([data])


def returnCameraIndexes():
    # checks the first 10 indexes.
    index = 0
    arr = []
    i = 10
    while i > 0:
        #print("retry cap : ", index)
        cap = cv2.VideoCapture(index)
        print("cap status :" ,cap.isOpened())
        
        if cap.isOpened():
            print("is open! index =",index)
            arr.append(index)
            cap.release()
        index += 1
        i -= 1
    print(arr)
    return arr


def undistort(img):
    DIM=(640,480)
    K=np.array([[499.95795693114394, 0.0, 299.98932387422747], [0.0, 499.81738564423085, 233.07326875070703], [0.0, 0.0, 1.0]])    
    D=np.array([[-0.12616907524146279], [0.4164021464039151], [-1.6015342220517828], [2.094848806959125]])

    h,w = img.shape[:2]
    #start = time.time()
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    #end = time.time()
    #seconds = end - start
    #start1 = time.time()
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    #end1 = time.time()
    #seconds1 = end1 - start1
    #write_data([seconds,seconds1])
    return undistorted_img
#old stitcher

class Stitcher:
  def __init__(self):
  # determine if we are using OpenCV v3.X and initialize the
  # cached homography matrix
      print("stitcher class inizialization")
      self.isv3 = imutils.is_cv3()
      self.cachedH = None
  def detectAndDescribe(self, image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # check to see if we are using OpenCV 3.X
    #if self.isv3:
    # detect and extract features from the image
    descriptor = cv2.xfeatures2d.SIFT_create()
    (kps, features) = descriptor.detectAndCompute(image, None)
    # arrays
    kps = np.float32([kp.pt for kp in kps])
    # return a tuple of keypoints and features
    print("detect and describe completed..")
    return (kps, features)
  def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
        ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual
        # matches
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = []
    # loop over the raw matches
    for m in rawMatches:
    # ensure the distance is within a certain ratio of each
    # other (i.e. Lowe's ratio test)
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))
    if len(matches) > 4:
    # construct the two sets of points
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])
    # compute the homography between the two sets of points
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
    # return the matches along with the homograpy matrix
    # and status of each matched point
        print("dmatches, H, status completed..", len(matches))
        return (matches, H, status, ptsA, ptsB)
    
    # otherwise, no homograpy could be computed
    return None
  
  def stitch(self, images, ratio=0.85, reprojThresh=4.0):
    # unpack the images
    (imageB, imageA) = images
    # if the cached homography matrix is None, then we need to
    # apply keypoint matching to construct it
    if self.cachedH is None:
        # detect keypoints and extract
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)
        #print(kpsA[0][0])
        
        # match features between the two images
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
        # if the match is None, then there aren't enough matched
        # keypoints to create a panorama
        imA = imageA.copy()
        imB = imageB.copy()
        cv2.imwrite("img1.jpg",imA)
        cv2.imwrite("img2.jpg",imB)
        
        ptsA = M[3]
        ptsB = M[4]
        for i in range(0, len(ptsA)):
            imA = cv2.circle(imA, (int(ptsA[i][0]), int(ptsA[i][1])), 5, (0,0,255),1 )
        cv2.imwrite("kp.jpg",imA)
        for i in range(0,len(ptsB)):
            imB = cv2.circle(imB, (int(ptsB[i][0]), int(ptsB[i][1])), 5, (0,0,255),1 )
        cv2.imwrite("kpb.jpg",imB)
        
        if M is None:
          return None
        # cache the homography matrix
        self.cachedH = M[1]
        # apply a perspective transform to stitch the images together
        # using the cached homography matrix
    result = cv2.warpPerspective(imageA, self.cachedH,
      (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
    result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
    # return the stitched image
    
    return result 




def ID_to_ex_string(ID):
    # read from ex_info
    config = configparser.ConfigParser()
    config.read('exercise_info.ini')
    sections = config.sections()
    # print("sections are : {}".format(sections))

    for exercise in sections:
        ID_num = int(config[exercise]["ID"])

        if ID_num == ID:
            ex_string = exercise

            return ex_string

    # print("no ID found")
    ex_string = "0"
    return ex_string


def KP_to_render_from_config_file(segments):
    config_geometrical = configparser.ConfigParser()

    config_geometrical.read('config.ini')
    KPS_to_render = []
    # print(type(dictionary["segments"]))
    # print(type(dictionary["eva_range"]))
    for arto in segments:
        # print("analizing arto: {}".format(arto))

        kps = config_geometrical["ALIAS"][arto]

        kps = [int(x) for x in kps.split(",")]
        KPS_to_render.append(kps)

    # print(KPS_to_render)
    return KPS_to_render


def ex_string_to_config_param(ex_string):
    # read from ex_info
    config_sk = configparser.ConfigParser()
    config_sk.read('exercise_info.ini')
    sections = config_sk.sections()
    # print("sections are : {}".format(sections))

    for exercise in sections:

        if exercise == ex_string:
            segments = config_sk.get(exercise, 'segments_to_render')
            segments = segments.split(',')

            KP_2_render = KP_to_render_from_config_file(segments)
            return KP_2_render

    # print("no exercise found_cannot get config parameters for geometry analysis")
    KP_2_render = []
    return KP_2_render


def KP_renderer_on_frame(ex_string, kp, img):

    if not ex_string:
        print("no exercise // no rendering")
    else:

        kp_2_rend = ex_string_to_config_param(ex_string)

        #print("kp_2_rend : ", kp_2_rend)

        for segment in kp_2_rend:

            x = []
            y = []
            #print("len segment ; ", len(segment))
            for i in range(0, len(segment)-1, 2): #####|||||||VA inserito nello script di jetson
                x.append(kp[segment[i]])
                y.append(kp[segment[i + 1]])

            for i in range(int(len(segment)/2) - 1 ):
                cv2.line(img, (x[i], y[i]), (x[i + 1], y[i + 1]), (255, 255, 255), 3)

            for i in range(len(x)):
                cv2.circle(img, (x[i], y[i]), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (x[i], y[i]), 15, (0, 0, 255), 2)

def read_shared_mem_for_ex_string(mem_ex_value):
    if mem_ex_value == 0:
        ex_string = ""
        return ex_string
    else:

        ex_string = ID_to_ex_string(mem_ex_value)

        return ex_string


def landmarks2keypoints(landmarks, image): # deprecated
    image_width, image_height = image.shape[1], image.shape[0]
    keypoints = []
    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z
        keypoints.append([landmark.visibility, (landmark_x, landmark_y)])

    return keypoints


def landmarks2KP(landmarks, image):
    image_width, image_height = image.shape[1], image.shape[0]
    keypoints = []
    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z
        keypoints.append(landmark_x)
        keypoints.append(landmark_y)
    return keypoints


def skeletonizer(KP_global, EX_global, q):
    # printing process id
    print("ID of process running worker1: {}".format(os.getpid()))
    
    width = 480
    height = 240
    '''

    gst_str1 = ('nvarguscamerasrc sensor-id=0 ! ' + 'video/x-raw(memory:NVMM), ' +
          'width=(int)1280, height=(int)720, ' +
          'format=(string)NV12, framerate=(fraction)30/1 ! ' + 
          'nvvidconv flip-method=2 ! ' + 
          'video/x-raw, width=(int){}, height=(int){}, ' + 
          'format=(string)BGRx ! ' +
          'videoconvert ! appsink').format(width, height)
    gst_str2 = ('nvarguscamerasrc sensor-id=1 ! ' + 'video/x-raw(memory:NVMM), ' +
          'width=(int)1280, height=(int)720, ' +
          'format=(string)NV12, framerate=(fraction)30/1 ! ' + 
          'nvvidconv flip-method=2 ! ' + 
          'video/x-raw, width=(int){}, height=(int){}, ' + 
          'format=(string)BGRx ! ' +
          'videoconvert ! appsink').format(width, height)
    gst_str3 = "0"
    '''

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    port = 5000

    fs = sender.FrameSegment(s, port)
    #stitcher serve per fondere due immagini da due camere
    print("try creating stitcher...")
    stitcher = Stitcher()
    camera_index = returnCameraIndexes()
    print("length camera index : ",len(camera_index))
    if len(camera_index) == 2:
        print("2 camera system")
    
        cap = cv2.VideoCapture(camera_index[0])
        cap1 = cv2.VideoCapture(camera_index[1])
        frame_width2 = int(cap.get(3))
        frame_height2 = int(cap.get(4))
        print(frame_width2,frame_height2)

        frame_width1 = int(cap1.get(3))
        frame_height1 = int(cap1.get(4))
    elif len(camera_index) == 1:
        print("1 camera system")
        cap = cv2.VideoCapture(camera_index[0])
        frame_width2 = int(cap.get(3))
        frame_height2 = int(cap.get(4))
        print(frame_width2,frame_height2)

    
    
    else:
        print("not enough camera aviable: camera numer = ",len(camera_index))
        return 0

    #cap = cv2.VideoCapture(gst_str1, cv2.CAP_GSTREAMER)

    #cap1 = cv2.VideoCapture(gst_str2, cv2.CAP_GSTREAMER) 
    
    #print("now i show you")
    
    #frame_width = int(cap1.get(3))
    #frame_height = int(cap1.get(4))*2

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    print("start pose config")
    with mp_pose.Pose(
            static_image_mode=False,  # false for prediction
            upper_body_only=False,
            smooth_landmarks=True,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8) as pose:
        print("start loop")
        
        if len(camera_index) == 2:
            print("is cap0 opened ?: ",cap.isOpened())
            print("is cap1 opened ?: ",cap1.isOpened())
        
        elif len(camera_index) == 1:
            print("is cap0 opened ?: ",cap.isOpened())
        
        
        while cap.isOpened():

            start = time.time()
            ex_string = read_shared_mem_for_ex_string(EX_global.value)
            ID = ex_string_to_ID(ex_string)
            
            
            #image=undistort(image)
            #image1=undistort(image1)
            #print("read image succes")
            if len(camera_index) == 2:
                if ID >= 50 and ID <= 60 :
                    success, image = cap1.read()
                else:
                    success, image = cap.read()
                    
                image=undistort(image)
            else:
                success, image = cap.read()
                image=undistort(image)
                
                
                
                
                
                
                #image=undistort(image)
                #image1=undistort(image1)
                #image1 = cv2.rotate(image1,cv2.ROTATE_180)
                #image1 = cv2.rotate(image1,cv2.ROTATE_90_COUNTERCLOCKWISE)
                #image = cv2.rotate(image,cv2.ROTATE_90_COUNTERCLOCKWISE)
                #cv2.putText(image, 'img 1', (300, 200), cv2.FONT_HERSHEY_COMPLEX, 3, (255,0,255), 3)
                #cv2.putText(image1, 'img 2', (300, 200), cv2.FONT_HERSHEY_COMPLEX, 3, (255,0,255), 3)
                #image1=undistort(image1)
                

            if not success:
                print("Ignoring empty camera0 frame.")
                # If loading a video, use 'break' instead of 'continue'.
                return False
            if len(camera_index) == 2:
                if not success1:
                    print("Ignoring empty camera2 frame.")
                    return False  
            #image = cv2.rotate(image,cv2.ROTATE_180)
            #image1 = cv2.rotate(image1,cv2.ROTATE_180)
            
             
            #due tipi di stichetr diversi quado le camere saranno montate stai pronto e usane uno.
            '''
            
            #stitcher = cv2.createStitcher(False)
            #sti = stitcher.stitch((image,image1))

            '''
            #correzione fisheye necessita calibrazione
            #image=undistort(image)
            #image1=undistort(image1)
            #print("undistorted")

            
            
            

            #2camere smontate____
            #sti = np.concatenate((image,image1), axis= 1)
            #2camere montate stitcher___
            #print("calling stitcher function...")
            if len(camera_index) == 2:
                #sti = cv_stitcher(image,image1)
                #cv2.imwrite("sti.jpg",sti)
                #break
                #sti = np.concatenate((image,image1), axis= 1)
                #sti = align_image_panorama(image, image1)
                sti = image
            
                #sti = stitcher.stitch([image, image1])
                #sti = np.concatenate((image,image1[0:frame_width1, 0:frame_height1]), axis= 1)
                #sti, postM, h  = alignImages(image,image1,250,0.5)
                #monocamera___
                #sti = image
                
            else:
                sti = image
                
            #sti = cv2.rotate(sti,cv2.ROTATE_90_CLOCKWISE)
    
            #sti = cv2.flip(sti, 1)
            



            if sti is None:
                print("[INFO] homography could not be computed")
                break
            
            #cv2.imshow('MediaPipeconc', conc)
            
            #assert status == 0 # Verify returned status is 'success'

            
            # render in front of ex_string
            if ex_string != "":


                #sti = np.concatenate((image,image1[550:720, 0:480]), axis= 0)

                #sti = cv2.cvtColor(sti, cv2.COLOR_BGR2RGB)
                #print("sti creted")

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                sti.flags.writeable = False
                #print("preproc")
                results = pose.process(sti)




                # Draw the pose annotation on the image.
                sti.flags.writeable = True
                #sti = cv2.cvtColor(sti, cv2.COLOR_RGB2BGR)
                end = time.time()
                seconds = end - start
                fps = 1 / seconds
                cv2.putText(sti, 'FPS: {}'.format(int(fps)), (frame_width2-300, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0))
                # Render detections
                mp_drawing.draw_landmarks(sti, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                          )

                # converting LM to KP
                if results.pose_landmarks is not None:
                    # svuoto queue
                    while not q.empty():
                        bit = q.get()
                    kp = landmarks2KP(results.pose_landmarks, sti)
                    if q.full():
                        print("impossible to insert data in full queue")
                    else:

                        q.put(kp)

                    # print(KP_global)

                    # print("KP global found : {}".format(len(KP_global)))
                    KP_renderer_on_frame(ex_string, kp, sti)



            # invio streaming
            fs.udp_frame(sti)
            #sender.send_status(5002, "KP_success")
            #print("udp completed img")
            

            cv2.imshow('MediaPipe Pose', sti)
            if cv2.waitKey(5) & 0xFF == 27:
                return False

        cap.release()
        s.close()

        cv2.destroyAllWindows()
