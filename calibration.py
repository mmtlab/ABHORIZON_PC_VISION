import cv2
import time


import numpy as np
import os
import glob



def capture():

    width = 480
    height = 240

    gst_str1 = ('nvarguscamerasrc sensor-id=0 ! ' + 'video/x-raw(memory:NVMM), ' +
               'width=(int)1280, height=(int)720, ' +
               'format=(string)NV12, framerate=(fraction)30/1 ! ' + 
               'nvvidconv flip-method=2 ! ' + 
               'video/x-raw, width=(int){}, height=(int){}, ' + 
               'format=(string)BGRx ! ' +
               'videoconvert ! appsink').format(width, height)
    cap = cv2.VideoCapture(gst_str1, cv2.CAP_GSTREAMER) # video capture source camera (Here webcam of laptop) 
     # return a single frame in variable `frame`

    while(True):
        ret,frame = cap.read()
        cv2.imshow('img1',frame) #display the captured image
        if cv2.waitKey(1) & 0xFF == ord('y'): #save on pressing 'y' 
            a = str(time.time())
            cv2.imwrite('c1'+ a  + '.jpg' ,frame)
            cv2.destroyAllWindows()
        
            

    cap.release()
    
    
def calibrate():
        CHECKERBOARD = (6,9)
        subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
        objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        _img_shape = None
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        images = glob.glob('*.jpg')
        print(len(images))
 
        for fname in images:
            img = cv2.imread(fname)
            if _img_shape == None:
                _img_shape = img.shape[:2]
            else:
                assert _img_shape == img.shape[:2], "All images must share the same size."
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	    # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray,CHECKERBOARD,cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
            #print(corners)
	    # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
                imgpoints.append(corners)
        N_OK = len(objpoints)
        print(N_OK)
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        retval, K, D, rvecs, tvecs = cv2.fisheye.calibrate(objpoints,imgpoints,gray.shape[::-1],K,D,rvecs,tvecs,calibration_flags,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
        print("Found " + str(N_OK) + " valid images for calibration")
        print("DIM=" + str(_img_shape[::-1]))
        print("K=np.array(" + str(K.tolist()) + ")")
        print("D=np.array(" + str(D.tolist()) + ")")
# You should replace these 3 lines with the output in calibration step

def undistort(img):
    DIM=(480, 240)
    K=np.array([[293.5116081746901, 0.0, 243.25387145248732], [0.0, 260.4055747091259, 104.82988114365413], [0.0, 0.0, 1.0]])
    D=np.array([[-0.03576268944984472], [0.057197056735017134], [-0.16392042989226446], [0.12922000339789327]])
    
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img
'''
if __name__ == '__main__':
    width = 480
    height = 240

    gst_str1 = ('nvarguscamerasrc sensor-id=0 ! ' + 'video/x-raw(memory:NVMM), ' +
               'width=(int)1280, height=(int)720, ' +
               'format=(string)NV12, framerate=(fraction)30/1 ! ' + 
               'nvvidconv flip-method=2 ! ' + 
               'video/x-raw, width=(int){}, height=(int){}, ' + 
               'format=(string)BGRx ! ' +
               'videoconvert ! appsink').format(width, height)
    cap = cv2.VideoCapture(gst_str1, cv2.CAP_GSTREAMER) # video capture source camera (Here webcam of laptop) 
    

    while(True):
	    ret, frame = cap.read() # return a single frame in variable `frame`    
	    undistorted_img=undistort(frame)
	    cv2.imshow("undistorted", undistorted_img)
	    cv2.waitKey(0)
    cap.release() 
    cv2.destroyAllWindows()
'''
#capture()
calibrate()

