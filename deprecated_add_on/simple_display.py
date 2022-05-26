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




cap = cv2.VideoCapture(0)

print("cap :",cap.isOpened())



frame_width2 = int(cap.get(3))
frame_height2 = int(cap.get(4))




#frame_width = int(cap1.get(3))
#frame_height = int(cap1.get(4))*2



while cap.isOpened():

   
    
    success, image = cap.read()
 
    


    cv2.imshow('MediaPipeconc', image)
   

    


    if cv2.waitKey(5) & 0xFF == 27:
        cv2.destroyAllWindows()
        

cap.release()


cv2.destroyAllWindows()
