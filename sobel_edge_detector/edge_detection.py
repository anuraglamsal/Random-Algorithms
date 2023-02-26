import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('tiger.jpg', 0) 

#smoothing
gaussian_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])/16 
smooth_img = cv2.filter2D(image, -1, gaussian_kernel); 

#applying sobel masks to find partial derivatives 
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) 
f_wrt_x = cv2.filter2D(smooth_img, -1, sobel_x)
f_wrt_y = cv2.filter2D(smooth_img, -1, sobel_y)

#finding the magnitude of the gradient vector at each point (x, y)
mag = np.absolute(f_wrt_x) + np.absolute(f_wrt_y)

#plotting our images
plt.subplot(1, 3, 1)
plt.title("original grayscaled")
plt.imshow(image, 'gray')

plt.subplot(1, 3, 2)
plt.title("smoothed")
plt.imshow(smooth_img, 'gray')

plt.subplot(1, 3, 3)
plt.title("edge detection output")
plt.imshow(mag, 'gray')

plt.show()
