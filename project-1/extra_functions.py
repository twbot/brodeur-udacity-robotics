#!/usr/bin/env python3

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Define a function to convert from cartesian to polar coordinates
def to_polar_coords(xpix, ypix):
    # Calculate distance to each pixel
    dist = np.sqrt(xpix**2 + ypix**2)
    # Calculate angle using arctangent function
    angles = np.arctan2(ypix, xpix)
    return dist, angles

def perspect_transform(img):

    scale = 10
    offset = 6
    src = np.float32([[15, 140], [118, 95], [200, 95], [301, 140]])
    dst = np.float32([
                [img.shape[1]/2, img.shape[0]-offset], 
                [img.shape[1]/2, img.shape[0]-scale-offset],
                [(img.shape[1]/2)+scale, img.shape[0]-scale-offset],
                [(img.shape[1]/2)+scale, img.shape[0]-offset]
                ])
    # Get transform matrix using cv2.getPerspectivTransform()
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp image using cv2.warpPerspective()
    # keep same size as input image
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    # Return the result
    return warped

def perspect_transform(img, src, dest):
    
    # Get transform matrix using cv2.getPerspectivTransform()
    M = cv2.getPerspectiveTransform(src, dest)
    # Warp image using cv2.warpPerspective()
    # keep same size as input image
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    # Return the result
    return warped

def navigable_area_thresholding(img, rgb_thresh=(0,0,0)):
    color_select = np.zeros_like(img[:,:,0])
    thresh_vals = img> rgb_thresh
    thresh = (img[:,:,0] > rgb_thresh[0]) & (img[:,:,1] > rgb_thresh[1]) & (img[:,:,2] > rgb_thresh[2])
    color_select[thresh] = 1
    return color_select

def goal_thresholding(img, rgb_thresh=(0,0,0)):
    color_select = np.zeros_like(img[:,:,0])
    thresh_vals = img > rgb_thresh
    thresh = (img[:,:,0] > rgb_thresh[0]) & (img[:,:,1] > rgb_thresh[1]) & (img[:,:,2] < rgb_thresh[2])
    color_select[thresh] = 1
    return color_select

def obstacle_thresholding(img, rgb_thresh=(0,0,0)):
    color_select = np.zeros_like(img[:,:,0])
    thresh_vals = img > rgb_thresh
    thresh = (img[:,:,0] < rgb_thresh[0]) & (img[:,:,1] < rgb_thresh[1]) & (img[:,:,2] < rgb_thresh[2])
    color_select[thresh] = 1
    return color_select

def rover_coords(binary_img): 
    # Calculate pixel positions with reference to the rover 
    # position being at the center bottom of the image.  
    x, y = binary_img.nonzero()
    x_pixel = -(x-binary_img.shape[0]).astype(np.float32)
    y_pixel = -(y-binary_img.shape[1]/2).astype(np.float32)
    return x_pixel, y_pixel

# Define a function to apply a rotation to pixel positions
def rotate_pix(xpix, ypix, yaw):
    # TODO:
    # Convert yaw to radians
    # Apply a rotation
    yaw = yaw*(np.pi/180)
    xpix_rotated = xpix*np.cos(yaw)-ypix*np.sin(yaw)
    ypix_rotated = xpix*np.sin(yaw)+ypix*np.cos(yaw)
    # Return the result 
    return xpix_rotated, ypix_rotated
 
# Define a function to perform a translation
def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # TODO
    # Apply a scaling and a translation
    xpix_translated = xpos+(xpix_rot/scale)
    ypix_translated = ypos+(ypix_rot/scale)
    # Return the result  
    return xpix_translated, ypix_translated

# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Clip to world_size
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world