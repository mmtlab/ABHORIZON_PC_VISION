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


def undistort(img):
    DIM=(480, 240)
    K=np.array([[293.5116081746901, 0.0, 243.25387145248732], [0.0, 260.4055747091259, 104.82988114365413], [0.0, 0.0, 1.0]])
    D=np.array([[-0.03576268944984472], [0.057197056735017134], [-0.16392042989226446], [0.12922000339789327]])
    
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img


class Stitcher:
  def __init__(self):
  # determine if we are using OpenCV v3.X and initialize the
  # cached homography matrix
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
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,reprojThresh)
	# return the matches along with the homograpy matrix
	# and status of each matched point
        return (matches, H, status)
    # otherwise, no homograpy could be computed
    return None
  
  def stitch(self, images, ratio=0.75, reprojThresh=4.0):
    # unpack the images
    (imageB, imageA) = images
    # if the cached homography matrix is None, then we need to
    # apply keypoint matching to construct it
    if self.cachedH is None:
        # detect keypoints and extract
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)
        # match features between the two images
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
        # if the match is None, then there aren't enough matched
        # keypoints to create a panorama
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

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    port = 5000

    fs = sender.FrameSegment(s, port)
    stitcher = Stitcher()
    



    cap = cv2.VideoCapture(gst_str1, cv2.CAP_GSTREAMER)

    #cap1 = cv2.VideoCapture(gst_str2, cv2.CAP_GSTREAMER) 
    
    #print("now i show you")
    frame_width2 = int(cap.get(3))
    frame_height2 = int(cap.get(4))

    #frame_width1 = int(cap1.get(3))
    #frame_height1 = int(cap1.get(4))
    #frame_width = int(cap1.get(3))
    #frame_height = int(cap1.get(4))*2

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
	static_image_mode = False,
        # upper_body_only=upper_body_only,
        model_complexity=0,
        #enable_segmentation=enable_segmentation,#unespected
	#smooth_landmark= True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        while cap.isOpened() :

            start = time.time()
            success, image = cap.read()
            #success1, image1 = cap1.read()

            if not success:
                # print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                return False
            #if not success1:
                #return False  
            #image = cv2.rotate(image,cv2.ROTATE_180)
            #image1 = cv2.rotate(image1,cv2.ROTATE_180)
            
             
            #due tipi di stichetr diversi quado le camere saranno montate stai pronto e usane uno.
            '''
            
            #stitcher = cv2.createStitcher(False)
            #sti = stitcher.stitch((image,image1))

            '''
            image=undistort(image)
            #image1=undistort(image1)
            #print("undistorted")

            
            
            

            #camere smontate
            #sti = np.concatenate((image,image1), axis= 1)
            #camere montate stitcher
            #sti = stitcher.stitch([image1, image])
            #monocamere
            #
            sti = image  



            if sti is None:
                print("[INFO] homography could not be computed")
                break
            
            #cv2.imshow('MediaPipeconc', conc)
            
            #assert status == 0 # Verify returned status is 'success'
            
                
            


            #sti = np.concatenate((image,image1[550:720, 0:480]), axis= 0)
            
            sti = cv2.cvtColor(cv2.flip(sti, 1), cv2.COLOR_BGR2RGB)
            #print("sti creted")
            
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            vis.flags.writeable = False
            #print("preproc")
            results = pose.process(sti)
            #print("posproc")
            

            # Draw the pose annotation on the image.
            sti.flags.writeable = True
            sti = cv2.cvtColor(sti, cv2.COLOR_RGB2BGR)
            end = time.time()
            seconds = end - start
            fps = 1 / seconds
            cv2.putText(sti, 'FPS: {}'.format(int(fps)), (frame_width2 - 190, 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                        255)
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

                ex_string = read_shared_mem_for_ex_string(EX_global.value)
                # render in front of ex_string
                if ex_string != "":
                    KP_renderer_on_frame(ex_string, kp, sti)

            # invio streaming
            fs.udp_frame(sti)
            #sender.send_status(5002, "KP_success")
            

            cv2.imshow('MediaPipe Pose', sti)
            if cv2.waitKey(5) & 0xFF == 27:
                return False

        cap.release()
        s.close()

        cv2.destroyAllWindows()
