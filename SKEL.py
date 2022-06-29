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
from datetime import datetime
from numpy.linalg import norm
import EVA
import statistics
import logging

camera_index_secondary = 1
camera_index_primary = 0
recording = False
writing = True
showing = False
printing_FPS = False
real_time_camera = True
model = 1


logging2 = logging.getLogger('SKEL')
logging2.setLevel(logging.INFO)
fh2 = logging.FileHandler('./log/SKEL.log')
fh2.setLevel(logging.DEBUG)
logging2.addHandler(fh2)

#logging.basicConfig(filename='SKEL_LOG.log', filemode='a', level=logging.DEBUG)
logging2.info(".............................................")
logging2.info("____!!!!!!!_____starting time____!!!!!!!_____: %s",datetime.now())
logging2.info(".............................................")


# camera_index_secondary = "/home/abhorizon/ABHORIZON_PC_VISION/data/video_subject_h_ex_1b.avi"


def write_data_csv(exercise,time,data):
    """
    write data 2 CSV,auto start and close when an exercise is done

    :param exercise: the name of the exercise for the title formatting
    :param time: timestamp of the start point of the excercise for the title of the excercise
    :param data: write to a csv file input data (the keypoints 13 of the body)

    :return: nothing
    """
    #print("filename_execution")
    filename = "./data/SKELETON_" + exercise + "_" + time.strftime("%m-%d-%Y_%H:%M:%S") +".csv"
    #print("filename:",filename)
    f = open(filename, 'a')
    writer = csv.writer(f)


    writer.writerow(data)
    f.close()




def brightness(img):
    """
    Calculate the brightness of the current frame

    :param img: the current frame of the camera

    :return: the brightness of the image calculated from the img channels
    """
    # funzione per il calcolo della luminosità dell immagine

    if len(img.shape) == 3:
        # Colored RGB or BGR (*Do Not* use HSV images with this function)
        # create brightness with euclidean norm
        return np.average(norm(img, axis=2)) / np.sqrt(3)
    else:
        # Grayscale
        return np.average(img)


def ex_string_to_ID(ex_string):
    """
    convert a string (of the exercixe) into a integer ID

    :param ex_string: the string describing the exercise

    :return: the id associated to the input string
    """
    # converte una stringa esercizio nel suo ID caratteristico
    # read from ex_info
    config = configparser.ConfigParser()
    config.read('exercise_info.ini')
    sections = config.sections()
    # print("sections are : {}".format(sections))
    ex_string = ex_string.rstrip("\n")
    for exercise in sections:

        if exercise == ex_string:
            # config.get("test", "foo")

            ID = int(config.get(exercise, 'ID'))
            return ID

    # print("no exercise found")
    ID = 0
    return ID


def writeCSVdata(ID, landmarks):
    """
    write data 2 CSV, 

    :param data: write to a csv file input data (append to the end)

    :return: nothing
    """
    # scrive su un file csv i dati estratti dalla rete Neurale
    file = open('./data/subject_n_ex_m.csv', 'a')
    writer = csv.writer(file)
    now = datetime.now()
    time = now.strftime("%d/%m/%Y %H:%M:%S")

    keypoints = []
    keypoints.append(ID)
    keypoints.append(time)

    for index, landmark in enumerate(landmarks.landmark):
        keypoints.append(landmark.x)
        keypoints.append(landmark.y)
        keypoints.append(landmark.z)
    writer.writerow(keypoints)
    file.close()


def returnCameraIndexes():
    """
    checks the first 10 indexes of cameras usb connected


    :return: an array of the opened camera index
    """
    # checks the first 10 indexes of cameras usb connected
    index = 0
    arr = []
    i = 10
    while i > 0:
        # print("retry cap : ", index)
        try:
            cap = cv2.VideoCapture(index)
        except:
            logging2.warning("camera index %s not aviable",index)
        # print("cap status :" ,cap.isOpened())

        if cap.isOpened():
            logging2.info("is open! index = %s", index)
            arr.append(index)
            cap.release()
        index += 1
        i -= 1
    logging2.info(arr)
    return arr


class Undistorter:
    # crea una matrice di mapping a partire dai parametri della camera per correggere le distorsioni fisheye
    """
    Undistort the fisheye image

    :param nothing: 

    :return: nothing
    """
    def __init__(self):
        self.cachedM1 = None
        self.cachedM2 = None
        self.cachedM1180 = None
        self.cachedM2180 = None

    def undistortOPT(self, img):

        if self.cachedM1 is None or self.cachedM2 is None:
            logging2.debug("calculating map1 and map 2")
            # calc map1,map2
            DIM = (640, 480)

            K = np.array([[236.75101538649935, 0.0, 309.9021941455992], [0.0, 236.2446439123776, 241.05082505485547],
                          [0.0, 0.0, 1.0]])
            D = np.array(
                [[-0.035061160585193776], [0.0019371574878152896], [-0.0109780009702086], [0.003567547827103574]])
            '''

            K=np.array([[499.95795693114394, 0.0, 299.98932387422747], [0.0, 499.81738564423085, 233.07326875070703], [0.0, 0.0, 1.0]])    
            D=np.array([[-0.12616907524146279], [0.4164021464039151], [-1.6015342220517828], [2.094848806959125]])
            '''
            h, w = img.shape[:2]
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
            logging2.info("DONE! completed map1 and 2, dim =")

            if map1 is None or map2 is None:
                logging2.error("ERROR no map calculated")
                return None

            # cache the homography matrix
            self.cachedM1 = map1
            self.cachedM2 = map2
            logging2.debug("saved map1 and 2")

            # print("DONE! completed map1 and 2, dim = ", (self.cachedM1).shape, (self.cachedM2).shape)

        undistorted_img = cv2.remap(img, self.cachedM1, self.cachedM2, interpolation=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT)
        return undistorted_img

    def undistortOPT180(self, img):

        if self.cachedM1180 is None or self.cachedM2180 is None:
            logging2.debug("calculating m1, m2 180")

            K = np.array([[256.08951474686006, 0.0, 338.0670093090018], [0.0, 256.106658840997, 249.32266973335038],
                          [0.0, 0.0, 1.0]])
            D = np.array(
                [[-0.03532111776488767], [-0.015025999952290566], [0.00976541095811982], [-0.0033155746136321975]])

            # calc map1,map2
            DIM = (640, 480)

            h, w = img.shape[:2]
            map1180, map2180 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
            logging2.info("DONE! completed map1 and 2, dim = %s", map1180.shape, map2180.shape)

            if map1180 is None or map2180 is None:
                logging2.error("ERROR no map calculated")
                return None

            # cache the homography matrix
            self.cachedM1180 = map1180
            self.cachedM2180 = map2180
            logging2.debug("saved map1 and 2 180")

            # print("DONE! completed map1 and 2, dim = ", (self.cachedM1).shape, (self.cachedM2).shape)

        undistorted_img = cv2.remap(img, self.cachedM1180, self.cachedM2180, interpolation=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT)
        return undistorted_img



def ID_to_ex_string(ID):
    """
    convert an ID to the respective string of the exercise

    :param ID: the ID describing the exercise

    :return: the string associated to the input string
    """
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
    ex_string = ""
    return ex_string


def rendering_kp_on_frame(joints,kp,img):
    """
    render an array of bodyparts on the image, 

    :param joints: an array containing the bosy parts
    :param kp: the keypoints of the current frame detected
    :param img: the image matrix
    :return: the image annotated
    """

    # grafica i KP coinvolti nell esercizio sull immagine
    overlay = img.copy()
    alpha = 0.3
    if not joints:
        logging2.critical("no exercise // no rendering")
        return img
    else:

        for arto in joints:
            
            lines = int(len(arto)/3) #0,1 len = 2
            
            for jj in range(lines):
                P1 = (kp[arto[jj*lines]] , kp[arto[jj*lines + 1]])
                P2 = (kp[arto[jj*lines +2]], kp[arto[jj*lines + 3]])
                cv2.line(overlay, P1, P2, (55, 100, 255), 15)

            # cv2.circle(img, (kp[0], kp[1]), 3, (0, 0, 255), cv2.FILLED)
            # for i in range(len(x)):
            # cv2.circle(img, (x[i], y[i]), 1, (0, 0, 255), cv2.FILLED)
            # cv2.circle(img, (x[i], y[i]), 3, (0, 0, 255), 2)
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        return img


def read_shared_mem_for_ex_string(mem_ex_value):
    """
    read the multiprocessing memory every frame to check new command/exercise

    :param mem_ex_value: the ID describing the exercise saved on the multiproc memory

    :return: the string associated to the memory ID
    """
    # controlla la memoria condivisa e estrae l esercizio salvato
    if mem_ex_value == 0:
        ex_string = ""
        return ex_string
    else:

        ex_string = ID_to_ex_string(mem_ex_value)

        return ex_string





def landmarks2KP(landmarks, image):
    """
    convert mediapipe landmark object in an array of coordinate of the foundam,ental keypoints of the body,
    some joints are syntetized in one to simplify the skeleton render and repetability 
    :param landmarks: mediapipe object containing the coordinate, visibility and confidence of the detected joints
    :param image: the image Matrix, input of the neural network, its dimension is used to convert absolute coordinate (0-1) to
    pixel coordinate (0 -image dim)

    :return: the array containing the coordinate in pixel of the joints detected
    """
    image_width, image_height = image.shape[1], image.shape[0]
    keypoints = []
    keypoints_simply = []
    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z
        keypoints.append([landmark_x, landmark_y])

    keypoints_simply.append(np.median(keypoints[0:10], axis=0).astype(int))
    keypoints_simply.append(keypoints[11])
    keypoints_simply.append(keypoints[12])
    keypoints_simply.append(keypoints[13])
    keypoints_simply.append(keypoints[14])

    keypoints_simply.append(np.median([keypoints[15], keypoints[17], keypoints[19], keypoints[21]], axis=0).astype(int))
    keypoints_simply.append(np.median([keypoints[16], keypoints[18], keypoints[20], keypoints[22]], axis=0).astype(int))
    keypoints_simply.append(keypoints[23])
    keypoints_simply.append(keypoints[24])
    keypoints_simply.append(keypoints[25])
    keypoints_simply.append(keypoints[26])
    keypoints_simply.append(np.median([keypoints[27], keypoints[29], keypoints[31]], axis=0).astype(int))
    keypoints_simply.append(np.median([keypoints[28], keypoints[30], keypoints[32]], axis=0).astype(int))

    # print("kp instance 2.0:", len(keypoints_simply))

    # cv2.circle(image, (keypoints_simply[1]),5,(255, 100, 255),3)
    # cv2.circle(image, (keypoints_simply[2]), 5, (255, 100, 255), 3)
    # cv2.circle(image, (keypoints_simply[3]), 5, (255, 100, 255), 3)
    # cv2.circle(image, (keypoints_simply[4]), 5, (255, 100, 255), 3)

    kpp = []

    for u in range(len(keypoints_simply)):
        kpp.append(keypoints_simply[u][0])
        kpp.append(keypoints_simply[u][1])

    # print("kpppp:", len(kpp))
    return kpp


def skeletonizer(KP_global, EX_global, q, user_id):
    """
    Main function of the skeletonizer, perform camera management, neural network detection, rendering, and image streaming
    :param EX_global: ID of the current exercise/state saved in the multiprocessing memory
    :param q: the multiprocessing queue where kp are put by the skeletonizer and consumed  by the evaluator
    :param user_id: the identificative of the user that is using the machine to do the exercises

    :return: nothing
    """
    #header csv file generation
    header_csv = []
    for nkp in range(13):
        header_csv.append("x"+ str(nkp))
        header_csv.append("y"+ str(nkp))
    print(header_csv)
    #all_exercise = EVA.load_all_exercise_in_RAM()
    #flag for file csv generation
    inizialized_csv_file = False
    exercise_csv = ""
    time_csv = ""
    #user, dati dell utente, identificativo
    user = user_id.value
    
    # corpo del codice con ini camere e rete neurale
    # printing process id
    #inizializing parameters for the exercise and com
    logging2.info("ID of process running worker1: {}".format(os.getpid()))
    dictionary = {}
    ID = 0
    camera = 0
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    port = 5000
    fs = sender.FrameSegment(s, port)
    # stitcher serve per fondere due immagini da due camere
    logging2.info("try creating undistorter...")
    undistorter = Undistorter()
    camera_index = returnCameraIndexes()
    logging2.info("length camera index : %s", len(camera_index))
    if len(camera_index) == 2:
        logging2.info("2 camera system")
        try:
            cap = cv2.VideoCapture(camera_index[camera_index_primary])
            logging2.info("primary camera connected, CAP ok")

        except:
            logging2.info("primary camera index ERROR")
            
        if real_time_camera == True:
            try:
                cap1 = cv2.VideoCapture(camera_index[camera_index_secondary])
                logging2.info("secondary camera OK -inizialized cap-(LOW) (secondary sensor)")
            except:
                logging2.info("ERROR SECONDARY CAMERA _ LOW _CAM _ CAP _ ERROR)")

                
        else:
            try:
                path = "/home/abhorizon/ABHORIZON_PC_VISION/data/old_test/video_subject_z_ex_1.avi"
                cap1 = cv2.VideoCapture(path)
                logging2.info("video for offline evaluation : %s",path)
            except:
                logging2.error("video file not valid or not present in path: %s",path)

                    

        frame_width2 = int(cap.get(3))
        frame_height2 = int(cap.get(4))
        logging2.info("frame dimension: {}".format([frame_width2, frame_height2]))

        frame_width1 = int(cap1.get(3))
        frame_height1 = int(cap1.get(4))

    elif len(camera_index) == 1:
        logging2.error("ERROR 1 camera system")
        cap = cv2.VideoCapture(camera_index[0])
        frame_width2 = int(cap.get(3))
        frame_height2 = int(cap.get(4))
        logging2.info("frame dimension: {}".format(frame_width2, frame_height2))


    else:
        logging2.error("not enough camera aviable: camera numer = %s", len(camera_index))
        return 0
    if recording == True:
        out = cv2.VideoWriter('./data/video_subject_n_ex_m.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                          (frame_height1, frame_width1))

    # cap = cv2.VideoCapture(gst_str1, cv2.CAP_GSTREAMER)

    # cap1 = cv2.VideoCapture(gst_str2, cv2.CAP_GSTREAMER)

    # print("now i show you")

    # frame_width = int(cap1.get(3))
    # frame_height = int(cap1.get(4))*2

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    logging2.info("start pose config")
    with mp_pose.Pose(
            static_image_mode=False,  # false for prediction
            model_complexity=model,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.6) as pose:
        logging2.info("start loop")

        if len(camera_index) == 2:
            logging2.info("is cap0 opened ?:%s ", cap.isOpened())
            logging2.info("is cap1 opened ?:%s ", cap1.isOpened())

        elif len(camera_index) == 1:
            logging2.info("is cap0 opened ?:%s ", cap.isOpened())

        while cap.isOpened():
            if printing_FPS == True:
                start = time.time()
           
            if EX_global.value != 0:
                if bool(dictionary):
                    #print("dict ok")
                    pass
                else:
                    ex_string = read_shared_mem_for_ex_string(EX_global.value)
                    dictionary = EVA.ex_string_to_config_param(ex_string)
                    
                    logging2.info("creating dict: %s",dictionary)
                    ID = dictionary["ID"]
                    camera = int(dictionary["camera"])
                    logging2.info("EXERCISE ID: %s ", ID)
                    logging2.info("EXERCISE camera: %s ", camera)
                    continue




            else:
                dictionary = {}
                ex_string = ""
                ID = 0
                camera = 0
            #print("id", ID)
            # image=undistort(image)
            # image1=undistort(image1)
            # print("read image succes")
            
            
            if len(camera_index) == 2:
                if camera < 2:
                    success, image = cap1.read()  #capture low camera
                    if camera == 1:
                        #logging2.info("correction of distortion")
                        if real_time_camera == True:
                            image = undistorter.undistortOPT180(image) #correct distortion
                        pass
                    if real_time_camera == True:
                        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                        image = cv2.rotate(image, cv2.ROTATE_180) #da ripristinare 3 righe
                        #print("___________co0rrs")

                else:
                    success, image = cap.read()

                    # image=undistort(image)
                    image = undistorter.undistortOPT(image)
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                    # image = cv2.rotate(image,cv2.ROTATE_180)

            else:
                success, image = cap.read()
                image = undistorter.undistortOPT(image)
                #image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

            # image = image[180:frame_width1, 70:frame_height1-70]


            if not success:
                logging2.error("Ignoring empty camera0 frame.")
                # If loading a video, use 'break' instead of 'continue'.
                return False
            if len(camera_index) == 2:
                if not success:
                    logging2.error("Ignoring empty camera2 frame.")
                    return False
                    # image = cv2.rotate(image,cv2.ROTATE_180)
            # image1 = cv2.rotate(image1,cv2.ROTATE_180)

            # due tipi di stichetr diversi quado le camere saranno montate stai pronto e usane uno.
            '''

            #stitcher = cv2.createStitcher(False)
            #sti = stitcher.stitch((image,image1))

            '''
          
            if len(camera_index) == 2:
                
                sti = image
               
            else:
                sti = image
                # sti = cv2.rotate(sti,cv2.ROTATE_90_CLOCKWISE)

            alpha = 4
            beta = 12
            # sti = cv2.convertScaleAbs(sti, alpha=5, beta=2)
            # if brightness(sti) < 80:
            # print(brightness(sti))
            # sti = cv2.convertScaleAbs(sti, alpha=alpha, beta=beta)

            sti = cv2.flip(sti, 1)

            if sti is None:
                logging2.error("image null")
                break

            # cv2.imshow('MediaPipeconc', conc)

            # assert status == 0 # Verify returned status is 'success'

            # render in front of ex_string
            if ex_string == "":
                if inizialized_csv_file == True:
                    inizialized_csv_file = False
                    logging2.critical("no string in skel --> ergo stop signal from coordinator")

            else:
                            # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                if recording == True:
                    out.write(sti)
                sti.flags.writeable = False
                # end1 = time.time()
                # seconds1 = end1 - start
                #
                if printing_FPS == True:
                    end_pre_proc = time.time()
                    seconds_pre_proc = end_pre_proc - start

                # start2 = time.time()
                results = pose.process(sti)
                # end2 = time.time()
                # seconds2 = end2 - start2
                # print("p", results.pose_landmarks)

                # start3 = time.time()
                if printing_FPS == True:
                    start_post_proc = time.time()

                # Draw the pose annotation on the image.
                sti.flags.writeable = True
                # sti = cv2.cvtColor(sti, cv2.COLOR_RGB2BGR)

                # Render detections

                '''
                mp_drawing.draw_landmarks(sti, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                          )
                '''
                # converting LM to KP
                if results.pose_landmarks is not None:
                    #svuoto queue
                    #scrivo su csv
                    
                    # print("rendering...")
                    while not q.empty():
                        bit = q.get()

                    kp = landmarks2KP(results.pose_landmarks, sti)
                    if writing == True:
                        if inizialized_csv_file == False:
                            exercise_csv = "user_"+ str(user) + "_" + ex_string
                            time_csv = datetime.now()
                            write_data_csv(exercise_csv,time_csv,header_csv)
                            inizialized_csv_file = True
                        else:
                            write_data_csv(exercise_csv,time_csv,kp)
                        
                    # print("kp : ",kp)

                    if q.full():
                        logging2.error("impossible to insert data in full queue")
                    else:

                        q.put(kp)
                    

                    # print(KP_global)

                    # print("KP global found : {}".format(len(KP_global)))
                    sti = rendering_kp_on_frame(dictionary["segments_to_render"],kp,sti)
                    #KP_renderer_on_frame(ex_string, kp, sti)
                else:
                    logging2.debug("results is none:%s ", results.pose_landmarks)

            # invio streaming
            fs.udp_frame(sti)

            # sender.send_status(5002, "KP_success")
            # print("udp completed img")

            if showing == True:
                cv2.imshow('MediaPipe Pose', sti)
            if cv2.waitKey(5) & 0xFF == 27:
                return False
            if ex_string != "":

                # cv2.putText(sti, 'FPS: {}'.format(int(fps)), (200, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0))
                if printing_FPS == True:
                    end = time.time()
                    seconds = end - start
                    second_post_proc = end - start_post_proc
                    fps = 1 / seconds
                    pose_inference_time = seconds-(seconds_pre_proc+second_post_proc)
                    logging2.info("FPS:{}, total:{},pose:{},preproc:{},postproc:{}".format( fps, round(seconds,4) ,round(pose_inference_time,4) ,round(seconds_pre_proc,4) ,round(second_post_proc,4) ))
        cap.release()
        cap1.release()

        out.release()
        s.close()

        cv2.destroyAllWindows()
