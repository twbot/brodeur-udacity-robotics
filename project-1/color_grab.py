#!/usr/bin/env python3

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import glob

from extra_functions import navigable_area_thresholding, obstacle_thresholding, goal_thresholding

def split_rgb(img):
	plt.imshow(img)
	plt.show()
	print(img.dtype, img.shape, np.min(img), np.max(img))

	red_channel = np.copy(img)
	red_channel[:, :, [1,2]] = 0

	green_channel = np.copy(img)
	green_channel[:, :, [0,2]] = 0

	blue_channel = np.copy(img)
	blue_channel[:, :, [0,1]] = 0

	real_channel = np.copy(img)
	real_channel[:, :, :] = 0

	fig = plt.figure(figsize=(12,3))
	plt.subplot(131)
	plt.imshow(red_channel)
	plt.subplot(132)
	plt.imshow(green_channel)
	plt.subplot(133)
	plt.imshow(blue_channel)
	plt.show()

def find_navigable_terrain(img):
	red_channel = 170
	green_channel = 170
	blue_channel = 170
	rgb_thresh = (red_channel, green_channel, blue_channel)
	color_sel = navigable_area_thresholding(img, rgb_thresh)
	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(21, 7), sharey = True)
	f.tight_layout()
	ax1.imshow(img)
	ax1.set_title('Original Image', fontsize=40)

	ax2.imshow(color_sel, cmap='gray')
	ax2.set_title('Navigable Terrain', fontsize=40)
	plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
	plt.show()

def find_obstacles(img):
	red_channel = 170
	green_channel = 170
	blue_channel = 170
	rgb_thresh = (red_channel, green_channel, blue_channel)
	color_sel = obstacle_thresholding(img, rgb_thresh)
	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(21, 7), sharey = True)
	f.tight_layout()
	ax1.imshow(img)
	ax1.set_title('Original Image', fontsize=40)

	ax2.imshow(color_sel, cmap='gray')
	ax2.set_title('Navigable Terrain', fontsize=40)
	plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
	plt.show()

def find_goals(img, show_image = 1):
	red_channel = 120
	green_channel = 120
	blue_channel = 60
	rgb_thresh = (red_channel, green_channel, blue_channel)
	color_sel = goal_thresholding(img, rgb_thresh)
	
	if(show_image):
		f, (ax1, ax2) = plt.subplots(1, 2, figsize=(21, 7), sharey = True)
		f.tight_layout()
		ax1.imshow(img)
		ax1.set_title('Original Image', fontsize=30)

		ax2.imshow(color_sel, cmap='gray')
		ax2.set_title('Goal Location', fontsize=30)
		plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
		plt.show()

	return (img, color_sel, (np.sum(color_sel) > 0))


def find_pics_with_goal():
	img_dir = '/Users/twbot/Desktop/udacity-robotics/project-1/IMG/*'
	img_list = glob.glob(img_dir)
	for img in img_list:
		image = mpimg.imread(img)
		goal = find_goals(image, show_image=0)
		if(goal[2]):
			f, (ax1, ax2) = plt.subplots(1, 2, figsize=(21, 7), sharey=True)
			f.tight_layout()
			ax1.imshow(goal[0])
			ax1.set_title('Orig', fontsize=30)
			ax2.imshow(goal[1], cmap='gray')
			ax2.set_title('Goal', fontsize=30)
			plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
			plt.show()

def main():
	filename = '/Users/twbot/Downloads/sample1.jpg'
	image = mpimg.imread(filename)

	# find_navigable_terrain(image)
	# find_obstacles(image)
	# find_goals(image)

	find_pics_with_goal()

if __name__ == '__main__':
	main()