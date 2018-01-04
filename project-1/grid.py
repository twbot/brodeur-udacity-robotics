#!/usr/bin/env python3

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2

def perspect_transform(img, src, dst):

	# Get transform matrix using cv2.getPerspectivTransform()
	M = cv2.getPerspectiveTransform(src, dst)
	# Warp image using cv2.warpPerspective()
	# keep same size as input image
	warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
	# Return the result
	return warped

def main():
	# filename = '/Users/twbot/Desktop/udacity-robotics/project-1/ex_imgs/grid.jpg'
	filename = '/Users/twbot/Downloads/sample1.jpg'
	img = mpimg.imread(filename)
	print(img.shape)
	plt.imshow(img)
	plt.show()

	scale = 10
	offset = 6
	# Define source and destination points
	source = np.float32([[15, 140], [118, 95], [200, 95], [301, 140]])
	destination = np.float32([
				[img.shape[1]/2, img.shape[0]-offset], 
				[img.shape[1]/2, img.shape[0]-scale-offset],
				[(img.shape[1]/2)+scale, img.shape[0]-scale-offset],
				[(img.shape[1]/2)+scale, img.shape[0]-offset]
				])

	warped = perspect_transform(img, source, destination)
	# Draw Source and destination points on images (in blue) before plotting
	cv2.polylines(img, np.int32([source]), True, (0, 0, 255), 3)
	cv2.polylines(warped, np.int32([destination]), True, (0, 0, 255), 3)
	# Display the original image and binary               
	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 6), sharey=True)
	f.tight_layout()
	ax1.imshow(img)
	ax1.set_title('Original Image', fontsize=40)

	ax2.imshow(warped, cmap='gray')
	ax2.set_title('Result', fontsize=40)
	plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
	plt.show()


if __name__ == '__main__':
	main()