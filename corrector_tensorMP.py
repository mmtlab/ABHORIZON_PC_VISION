import cv2
import mediapipe as mp
import time
import numpy as np
import math

import sys
import socket

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


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


def distance_calculator_2D_2Joint(kp1, kp2):
    # normalizer = math.sqrt(pow((kp[22]-kp[24]),2) + pow((kp[23]-kp[25]),2)) #spalle

    # print("number of arms: {}".format(len(kps_to_render)))

    #  x1, y1, x2, y2distance_calculator_2D_2Joint
    xa = kp1[0]
    ya = kp1[1]
    xb = kp2[0]
    yb = kp2[1]
    # ((kp[segment[0]], kp[segment[1]]), (kp[segment[2]], kp[segment[3]])

    distance = ((math.sqrt(pow((xa - xb), 2) + pow((ya - yb), 2))))
    # print("distance ", distance)
    return distance


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
        print("angle§ : {}".format(angle))

    return angle


0

# Curl counter variables
counter = 0
stage = None
count = 0
dir = 0
pTime = 0

# path2video = "/home/bernardo/Scaricati/video1.avi"
path2video = "/home/bernardo/PycharmProjects/tensor/videoacq/h2b.mp4"

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(path2video)
frame_width = int(cap.get(3))

frame_height = int(cap.get(4))

print(frame_width)
print(frame_height)

out = cv2.VideoWriter('ferrari_test.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (frame_width, frame_height))
# cap = cv2.VideoCapture(path2video)


frame_width = int(cap.get(3))

frame_height = int(cap.get(4))

print(frame_width)
print(frame_height)
ret, first = cap.read()
first_gray = None
#first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
#first_gray = cv2.GaussianBlur(first_gray, (21, 21), 0)
# backSub = cv2.createBackgroundSubtractorMOG2()
# backSub = cv2.createBackgroundSubtractorKNN()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (53, 53))
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7 ))
old_3_kp = None
old_3_kp_1 = None
old_3_kp_2 = None
#first_gray = None

# out = cv2.VideoWriter('Pose_test.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (frame_width,frame_height))
# out = cv2.VideoWriter('Pose_test.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20,  (300,480))
with mp_pose.Pose(
        static_image_mode=False,  # false for prediction
        upper_body_only=False,
        smooth_landmarks=True,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8) as pose:
    while cap.isOpened():
        delta_time = []

        start = time.time()
        success, image = cap.read()

        image = image[200:frame_height, 80:frame_width-80]
        alpha = 1
        beta = 12
        #image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        original = image
        # image = cv2.resize(image, (0, 0), fx=0.7, fy=0.7)
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break

        if first_gray is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            difference = cv2.absdiff(gray, first_gray)

            # Apply thresholding to eliminate noise
            thresh = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)[1]

            # miglioramento maschera:
            thresh = cv2.morphologyEx(thresh,cv2.MORPH_OPEN, kernel)
            thresh = cv2.dilate(thresh, kernel1 , iterations=2)
            fgMask = cv2.erode(thresh, kernel2 , iterations=2)

            #fgMask = thresh

            imask = fgMask > 0

            green = 255 * np.ones_like(image, np.uint8)
            green[imask] = image[imask]
            # print(",thresh, fgMask, imask,green",thresh, fgMask, imask,green)
            # cv2.imshow("thresh", green)

            # fgMask = backSub.apply(image)
            # fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
            #image = green

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = pose.process(image)

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.pose_landmarks is not None:
            kp = landmarks2KP(results.pose_landmarks, image)
            kpor = kp
            # print("kpo : ", len(kp))
            if old_3_kp is not None and old_3_kp_1 is not None and old_3_kp_2 is not None:
                for i in range(int(len(kp) / 4)):

                    if i > 4 and i < 12 :
                        cv2.circle(image, (kp[i * 4 + 2], kp[i * 4 + 3]), 2, (0, 255, 0), -1)
                        cv2.circle(image, (kp[i * 4 + 4], kp[i * 4 + 5]), 2, (0, 255, 255), -1)

                        oriz_dist = kp[i*4 + 2] - kp[i*4 + 4]
                        if oriz_dist < 20:

                            kp[i * 4 + 2] = int(np.median([old_3_kp[i * 4 + 2],old_3_kp_1[i * 4 + 2]]))
                            kp[i * 4 + 4] = int(np.median([old_3_kp[i*4 + 4],old_3_kp_1[i * 4 + 4]]))

                        cv2.circle(image, (kp[i * 4 + 2], kp[i * 4 + 3]), 6, (0, 255, 0), )
                        cv2.circle(image, (kp[i * 4 + 4], kp[i * 4 + 5]), 6, (0, 255, 255), 2)

            old_3_kp_2 = old_3_kp_1
            old_3_kp_1 = old_3_kp
            old_3_kp = kp

            #print("aggiornato", kpor)


            '''

        # Get coordinates
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x*frame_width,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y*frame_height]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x*frame_width, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y*frame_height]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x*frame_width, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y*frame_height]

        shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x*frame_width, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y*frame_height]


        #shoulder_dist = distance_calculator_2D_2Joint(shoulder,shoulder_right)
        #print("shguld dist", shoulder_dist)
        #print(shoulder, old_3_kp[0])
        #delta_time_shoulder = shoulder - old_3_kp[0]
        #print(delta_time_shoulder)
        delta_time_shoulder =[]
        for i in range(len(shoulder)):
            delta_time_shoulder.append(shoulder[i] - old_3_kp[0][i])
        print("delta_time_shoulder :" , delta_time_shoulder)



        old_3_kp = [shoulder,elbow, wrist]
        '''

        # Extract landmarks
        '''
    
        landmarks = results.pose_landmarks.landmark

        # Get coordinates
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        # Calculate angle
        angle = calculate_angle(shoulder, elbow, wrist)

        # Visualize angle
        cv2.putText(image, str(int(angle)),
                    tuple(np.multiply(elbow, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                    )

        # Curl counter logic
        if angle > 160:
            stage = "down"
        if angle < 30 and stage == 'down':
            stage = "up"
            counter += 1
            print(counter)
        if 1:
            per = np.interp(angle, (10,160), (100,0))
            bar = np.interp(angle, (10,160), (150, 380))
            # print(angle, per)

            # Check for the dumbbell curls
            color = (0, 0, 255)
            if per == 100:
                color = (0, 255, 0)
                if dir == 0:
                    count += 0.5
                    dir = 1
            if per == 0:
                color = (0, 255, 0)
                if dir == 1:
                    count += 0.5
                    dir = 0
            #print(count)

            # Draw Bar
            cv2.rectangle(image, (500, 150), (530, 380), color, 3)
            cv2.rectangle(image, (500, int(bar)), (530, 380), color, cv2.FILLED)
            cv2.putText(image, f'{int(per)} %', (500, 100), cv2.FONT_HERSHEY_PLAIN, 2,
                        color, 4)



    except:
        pass

        # Render curl counter
        # Setup status box
    cv2.rectangle(image, (0, 0), (120, 73), (205, 117, 106), -1)

    # Rep data
    #cv2.putText(image, 'REPS', (15, 12),
                #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, str(counter),
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

    # Stage data
    cv2.putText(image, 'RIPETIZIONI:', (15, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    #cv2.putText(image, stage,(60, 60),cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

    '''

        # Render detections

        '''
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                          mp_drawing.DrawingSpec(color=(100, 117, 66), thickness=2, circle_radius=2),
                          mp_drawing.DrawingSpec(color=(20, 66, 230), thickness=2, circle_radius=2)
                          )
    '''

        end = time.time()
        seconds = end - start
        fps = 1 / seconds
        cv2.putText(image, 'FPS: {}'.format(int(fps) - 12), (frame_width - 190, 30), cv2.FONT_HERSHEY_COMPLEX, 1, 255)
        # out.write(image)
        cv2.imshow('MediaPipe Pose', image)
        cv2.waitKey(0)


        #first_gray = original
        #first_gray = cv2.cvtColor(first_gray, cv2.COLOR_BGR2GRAY)
        #first_gray = cv2.GaussianBlur(first_gray, (21, 21), 0)


        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
out.release()
cv2.destroyAllWindows()
