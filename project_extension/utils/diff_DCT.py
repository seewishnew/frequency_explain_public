
'''
python utils/diff_DCT.py --first images/original_02.png --second images/modified_02.png
'''

# import the necessary packages
from skimage.measure import compare_ssim
import argparse
import imutils
import cv2
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True,
	help="first input image")
ap.add_argument("-s", "--second", required=True,
	help="second")
args = vars(ap.parse_args())

# load the two input images
imageA = cv2.imread(args["first"])
imageB = cv2.imread(args["second"])
# convert the images to grayscale

# grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
# grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

grayA = imageA
grayB = imageB

# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
(score, diff) = compare_ssim(imageA, imageB, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))


# # threshold the difference image, followed by finding contours to
# # obtain the regions of the two input images that differ
# thresh = cv2.threshold(diff, 0, 255,
# 	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
# 	cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)


# # loop over the contours
# for c in cnts:
# 	# compute the bounding box of the contour and then draw the
# 	# bounding box on both input images to represent where the two
# 	# images differ
# 	(x, y, w, h) = cv2.boundingRect(c)
# 	cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
# 	cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
# # show the output images
# cv2.imshow("Original", imageA)
# cv2.imshow("Modified", imageB)
# cv2.imshow("Diff", diff)
# cv2.imshow("Thresh", thresh)
# cv2.waitKey(0)






from skimage.measure import compare_ssim
import cv2
import numpy as np


before = cv2.imread(args["first"])
after = cv2.imread(args["second"])


# Convert images to grayscale
before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

# Compute SSIM between two images
(score, diff) = compare_ssim(before_gray, after_gray, full=True)
print("Image similarity", score)

# The diff image contains the actual image differences between the two images
# and is represented as a floating point data type in the range [0,1] 
# so we must convert the array to 8-bit unsigned integers in the range
# [0,255] before we can use it with OpenCV
diff = (diff * 255).astype("uint8")

# Threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

mask = np.zeros(before.shape, dtype='uint8')
filled_after = after.copy()

for c in contours:
    area = cv2.contourArea(c)
    if area > 40:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(before, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.rectangle(after, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.drawContours(mask, [c], 0, (0,255,0), -1)
        cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)

cv2.imshow('before', before)
cv2.imshow('after', after)
cv2.imshow('diff',diff)
cv2.imshow('mask',mask)
cv2.imshow('filled after',filled_after)
cv2.waitKey(0)