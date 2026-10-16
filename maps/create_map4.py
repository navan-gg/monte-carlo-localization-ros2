import cv2

# map4 = map.pgm plus two obstacles that break its 180-degree symmetry.
# Without them a laser scan from (x, y) looks the same as one from (-x, -y),
# so a laser-only filter such as AMCL cannot tell the two poses apart.

img = cv2.imread("map.pgm", cv2.IMREAD_GRAYSCALE)

RES = 0.05
ORIGIN = -10.0
SIZE = img.shape[0]


def block(x1, x2, y1, y2):
    """Fill the rectangle [x1, x2] x [y1, y2] (metres, map frame) as occupied."""
    c1 = int((x1 - ORIGIN) / RES)
    c2 = int((x2 - ORIGIN) / RES)
    r1 = SIZE - int((y2 - ORIGIN) / RES)
    r2 = SIZE - int((y1 - ORIGIN) / RES)
    img[r1:r2, c1:c2] = 0


block(-8.5, -6.5, 3.0, 5.0)    # square block on the left side
block(6.0, 9.0, -1.5, -1.2)    # short wall on the right side

cv2.imwrite("map4.pgm", img)
