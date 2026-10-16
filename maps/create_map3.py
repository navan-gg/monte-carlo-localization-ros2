import numpy as np
import cv2

size = 400
img = np.ones((size, size), dtype=np.uint8) * 254   # free space

# ----- outer boundary -----
img[0:5, :] = 0
img[-5:, :] = 0
img[:, 0:5] = 0
img[:, -5:] = 0


def vertical_wall(x, y1, y2, thickness=4):
    img[y1:y2, x:x+thickness] = 0


def horizontal_wall(y, x1, x2, thickness=4):
    img[y:y+thickness, x1:x2] = 0


# ----- two walls crossing at the centre, splitting the map into 4 rooms -----

vertical_wall(200, 0, 400)


horizontal_wall(200, 0, 400)



cv2.imwrite("map3.pgm", img)
