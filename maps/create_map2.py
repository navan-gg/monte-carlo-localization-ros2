import numpy as np
import cv2

size = 400
img = np.ones((size, size), dtype=np.uint8) * 254   # free space

# ----- outer boundary -----
img[0:5, :] = 0
img[-5:, :] = 0
img[:, 0:5] = 0
img[:, -5:] = 0

def hollow_square(img, cx, cy, half, thickness=3):
    x1 = cx - half
    x2 = cx + half
    y1 = cy - half
    y2 = cy + half

    img[y1:y1+thickness, x1:x2] = 0
    img[y2-thickness:y2, x1:x2] = 0
    img[y1:y2, x1:x1+thickness] = 0
    img[y1:y2, x2-thickness:x2] = 0


# ----- symmetric hollow squares -----
hollow_square(img, 200, 120, 40)   # top
hollow_square(img, 200, 280, 40)   # bottom
hollow_square(img, 120, 200, 40)   # left
hollow_square(img, 280, 200, 40)   # right

cv2.imwrite("map2.pgm", img)
