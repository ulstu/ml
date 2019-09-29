import cv2
import matplotlib.pyplot as plt
import numpy as np

# load image
source_image = cv2.imread('lena.bmp')
cv2.imshow('lena', source_image)

img = source_image
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
img = np.array(img, dtype=np.float64) / 255


plt.figure()
plt.imshow(img)
plt.show()

# get image matrix shape
height, width, channel = source_image.shape
print('SOURCE IMAGE SHAPE')
print('height', height)
print('width', width)
print('channel', channel)

# convert color from RGB to GRAY
# RGB is 3-dim array, GRAY is 2-dim array (channel is none)
result_image = cv2.cvtColor(source_image, cv2.COLOR_RGB2GRAY)
height, width = result_image.shape
print('GRAY IMAGE SHAPE')
print('height', height)
print('width', width)

# drawing images in window
# need to set win name and image for drawing
cv2.imshow('SOURCE', source_image)
cv2.imshow('RESULT', result_image)

# waiting windows closing
cv2.waitKey(0)



# save reslut into new file and change file format
cv2.imwrite('result.png', result_image)
