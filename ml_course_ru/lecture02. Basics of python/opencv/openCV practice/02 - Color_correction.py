import cv2

source_image = cv2.imread('low_contrast.jpg')
image = cv2.cvtColor(source_image, cv2.COLOR_RGB2Lab)
planes = cv2.split(image)

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(15, 15))
planes[0] = clahe.apply(planes[1])
print(len(planes))
for i in range(len(planes)):
    cv2.imshow('plane', planes[i])
    cv2.waitKey(0)

image = cv2.merge(planes)
correct_image = cv2.cvtColor(image, cv2.COLOR_Lab2RGB)
cv2.imshow('SOURCE', source_image)
cv2.imshow('RESULT', correct_image)
cv2.waitKey(0)
