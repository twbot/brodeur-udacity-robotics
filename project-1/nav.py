#!/usr/bin/env python3

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

def first():
	filename = '/Users/twbot/Downloads/sample.jpg'
	image = mpimg.imread(filename)
	plt.imshow(image)
	plt.show()
	print(image.dtype, image.shape, np.min(image), np.max(image))

	red_channel = np.copy(image)
	red_channel[:, :, [1,2]] = 0

	green_channel = np.copy(image)
	green_channel[:, :, [0,2]] = 0

	blue_channel = np.copy(image)
	blue_channel[:, :, [0,1]] = 0

	real_channel = np.copy(image)
	real_channel[:, :, :] = 0

	fig = plt.figure(figsize=(12,3))
	plt.subplot(131)
	plt.imshow(red_channel)
	plt.subplot(132)
	plt.imshow(green_channel)
	plt.subplot(133)
	plt.imshow(blue_channel)
	plt.show()

def color_thresholding(img, rgb_thresh=(0,0,0)):
	color_select = np.zeros_like(img[:,:,0])
	thresh_vals = img> rgb_thresh
	thresh = (img[:,:,0] > rgb_thresh[0]) & (img[:,:,1] > rgb_thresh[1]) & (img[:,:,2] > rgb_thresh[2])
	color_select[thresh] = 1
	return color_select
	
def main():
	filename = '/Users/twbot/Downloads/sample.jpg'
	image = mpimg.imread(filename)

	first()
	red_channel = 170
	blue_channel = 170
	green_channel = 170
	rgb_thresh = (red_channel, blue_channel, green_channel)

	color_sel = color_thresholding(image, rgb_thresh = rgb_thresh)

	# Display the original image and binary               
	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(21, 7), sharey=True)
	f.tight_layout()
	ax1.imshow(image)
	ax1.set_title('Original Image', fontsize=40)

	ax2.imshow(color_sel, cmap='gray')
	ax2.set_title('Your Result', fontsize=40)
	plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
	plt.show()


if __name__ == '__main__':
	main()