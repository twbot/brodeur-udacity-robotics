#!/usr/bin/env python3

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def perspect_transform(img, src, dst):

	# Get transform matrix using cv2.getPerspectivTransform()
	M = cv2.getPerspectiveTransform(src, dst)
	# Warp image using cv2.warpPerspective()
	# keep same size as input image
	warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
	# Return the result
	return warped

def color_thresholding(img, rgb_thresh=(0,0,0)):
	color_select = np.zeros_like(img[:,:,0])
	thresh_vals = img> rgb_thresh
	thresh = (img[:,:,0] > rgb_thresh[0]) & (img[:,:,1] > rgb_thresh[1]) & (img[:,:,2] > rgb_thresh[2])
	color_select[thresh] = 1
	return color_select

def rover_coords(binary_img): 
    # Calculate pixel positions with reference to the rover 
    # position being at the center bottom of the image.  
	x, y = binary_img.nonzero()
	x_pixel = -(x-binary_img.shape[0]).astype(np.float32)
	y_pixel = -(y-binary_img.shape[1]/2).astype(np.float32)
	return x_pixel, y_pixel

def main():
	np.set_printoptions(threshold = np.inf)
	filename = '/Users/twbot/Desktop/grid.jpg'
	img = mpimg.imread(filename)
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
	warped_lines = np.copy(warped)

	# Draw Source and destination points on images (in blue) before plotting
	cv2.polylines(img, np.int32([source]), True, (0, 0, 255), 3)
	cv2.polylines(warped_lines, np.int32([destination]), True, (0, 0, 255), 3)
	
	# Display the original image and binary
	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 6), sharey=True)
	f.tight_layout()
	ax1.imshow(img)
	ax1.set_title('Original Image', fontsize=40)

	ax2.imshow(warped_lines, cmap='gray')
	ax2.set_title('Result', fontsize=40)
	plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
	plt.show()

	red_channel = 170
	blue_channel = 170
	green_channel = 170
	rgb_thresh = (red_channel, blue_channel, green_channel)

	color_sel = color_thresholding(warped, rgb_thresh = rgb_thresh)

	# Extract x and y positions of navigable terrain pixels
	# and convert to rover coordinates
	xpix, ypix = rover_coords(color_sel)

	# Plot the map in rover-centric coords
	fig = plt.figure(figsize=(5, 7.5))
	plt.plot(xpix, ypix, '.')
	plt.ylim(-160, 160)
	plt.xlim(0, 160)
	plt.title('Rover-Centric Map', fontsize=20)
	plt.show()

if __name__ == '__main__':
	main()