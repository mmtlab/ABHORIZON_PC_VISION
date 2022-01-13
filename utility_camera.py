#no more stitcher
import numpy as np
import imutils
import cv2

class Stitcher:
	def __init__(self):
		# determine if we are using OpenCV v3.X and initialize the
		# cached homography matrix
		self.isv3 = imutils.is_cv3()
		self.cachedH = None
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
  
  
stitcher = Stitcher()
while True:
  
  #estrai i due frame rutoali in modo che siano left e right
  result = stitcher.stitch([left, right])
  if result is None:
		print("[INFO] homography could not be computed")
		break
  cv2.imshow("Result", result)
	cv2.imshow("Left Frame", left)
	cv2.imshow("Right Frame", right)
  key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
# do a bit of cleanup
print("[INFO] cleaning up...")
cv2.destroyAllWindows()

