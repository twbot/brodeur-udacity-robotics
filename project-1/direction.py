#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
from extra_functions import perspect_transform, color_thresholding, rover_coords, to_polar_coords

picture = os.path.join('/Users/twbot/Desktop/', 'angle_example.jpg')
image = mpimg.imread(picture)
warped = perspect_transform(image) # Perform perspective transform
colorsel = color_thresholding(warped, rgb_thresh=(160, 160, 160)) # Threshold the warped image
xpix, ypix = rover_coords(colorsel)  # Convert to rover-centric coords
distances, angles = to_polar_coords(xpix, ypix) # Convert to polar coords
avg_angle = np.mean(angles) # Compute the average angle
avg_angle_degrees = avg_angle * (180/np.pi)
steering = np.clip(avg_angle_degrees, -15, 15)

# Do some plotting
fig = plt.figure(figsize=(12,9))
plt.subplot(221)
plt.imshow(image)
plt.subplot(222)
plt.imshow(warped)
plt.subplot(223)
plt.imshow(colorsel, cmap='gray')
plt.subplot(224)
plt.plot(xpix, ypix, '.')
plt.ylim(-160, 160)
plt.xlim(0, 160)
arrow_length = 100
x_arrow = arrow_length * np.cos(avg_angle)
y_arrow = arrow_length * np.sin(avg_angle)
plt.arrow(0, 0, x_arrow, y_arrow, color='red', zorder=2, head_width=10, width=2)
plt.show()