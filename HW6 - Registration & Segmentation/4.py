import cv2
import numpy as np
import matplotlib.pyplot as plt

# Reading the images
img1 = cv2.imread('Color_MRI.png', 0)
img2 = cv2.imread('Color_MRI2.png', 0)
img1_cpy = img1.copy()
img2_cpy = img2.copy()

# Creating UI
def on_mouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN: 
        cv2.circle(params[0], (x, y), 5, (255, 0, 0), -1)
        cv2.imshow(params[1], params[0])
        params[2].append([x, y])
        print(params[2])

points1 = []  
points2 = []  

cv2.namedWindow('Image 1')
cv2.setMouseCallback('Image 1', on_mouse, (img1_cpy, 'Image 1', points1))
cv2.imshow('Image 1', img1_cpy)

cv2.namedWindow('Image 2')
cv2.setMouseCallback('Image 2', on_mouse, (img2_cpy, 'Image 2', points2))
cv2.imshow('Image 2', img2_cpy)

cv2.waitKey(0)

# find affine transformation
pts1 = np.float32(points1)
pts2 = np.float32(points2)
M = cv2.getAffineTransform(pts2,pts1)


# performing transformation on the second image
rows,cols = img2.shape
dst_image = cv2.warpAffine(img2, M, (cols,rows))

# Results:
fig, ax = plt.subplots(1, 3, figsize=(10, 5))
ax[0].imshow(img1, cmap='gray', vmin=0, vmax=255)
ax[0].set_title('MRI1')
ax[1].imshow(img2, cmap='gray', vmin=0, vmax=255)
ax[1].set_title('MRI2')
ax[2].imshow(dst_image, cmap='gray', vmin=0, vmax=255)
ax[2].set_title('Transformed')
plt.show();

# Joint histogram:
def hist2d(img1, img2, title):
    H, xedges, yedges = np.histogram2d(img1.ravel(), img2.ravel(), bins=256, range=[[0, 256], [0, 256]], density=True)
    H = H.T
    fig = plt.figure(figsize=(7, 2))
    ax = fig.add_subplot(132, title=title, aspect='equal')
    X, Y = np.meshgrid(xedges, yedges)
    ax.pcolormesh(X, Y, H)
    plt.show()

hist2d(img1, img2, 'Image1 vs. Image2')
hist2d(img1, dst_image, 'Image1 vs. Transformed')
hist2d(img1, img1, 'Image1 vs. Image1')

