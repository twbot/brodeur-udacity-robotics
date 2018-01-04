#!/usr/bin/env python3

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from extra_functions import perspect_transform, color_thresholding, rover_coords

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