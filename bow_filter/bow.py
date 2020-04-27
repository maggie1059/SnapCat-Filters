import numpy as np
import cv2
from skimage.transform import resize
import imutils

def bow_filter(image, coords, bow_filter_path="bow.png", output_image_path="cat_bow.png")
    bgd = image
    fgd = cv2.imread(bow_filter_path, cv2.IMREAD_UNCHANGED)

    left_ear_tip = 4
    right_ear_tip = 7

    x_left = coords[left_ear_tip][0]
    y_left = coords[left_ear_tip][1]
    x_right = coords[right_ear_tip][0]
    y_right = coords[right_ear_tip][1]
    alpha = 1.0

    dx = float(x_right - x_left)
    dy = float(y_right - y_left)

    size = int(np.linalg.norm([dx, dy]))
    fgd = resize(fgd, (size, size, 4))

    if dy >= 0:
        angle = np.arctan(dy/dx)*180./np.pi
    else:
        angle = 360 - np.arctan(abs(dy)/dx)*180./np.pi

    fgd = imutils.rotate_bound(fgd, angle)

    y_left = y_left - 70

    y_offset = 0
    if y_left < 0:
        y_offset = -y_left
    elif y_left > bgd.shape[0]-1:
        y_offset = -(y_left - bgd.shape[0] + 1)

    x_offset = 0
    if x_left < 0:
        x_offset = -x_left
    elif x_left > bgd.shape[1]-1:
        x_offset = -(x_left - bgd.shape[1] + 1)

    roi = bgd[np.clip(y_left, 0, bgd.shape[0]-1):np.clip((y_left+fgd.shape[0]), 0, bgd.shape[0]-1), np.clip(x_left, 0, bgd.shape[1]-1):np.clip((x_left+fgd.shape[1]), 0, bgd.shape[1]-1)]

    for r in range(roi.shape[0]):
        for c in range(roi.shape[1]):
            if fgd[r + y_offset][c + x_offset][3] > 0:
                for i in range(3):
                    roi[r][c][i] = alpha*fgd[r + y_offset][c + x_offset][i]*255. + (1-alpha)*roi[r][c][i]

    bgd[np.clip(y_left, 0, bgd.shape[0]-1):np.clip((y_left+fgd.shape[0]), 0, bgd.shape[0]-1), np.clip(x_left, 0, bgd.shape[1]-1):np.clip((x_left+fgd.shape[1]), 0, bgd.shape[1]-1)] = roi
    cv2.imwrite(output_image_path, bgd)

if __name__ == "__main__":
    cat_image_path = "cat.jpg"
    coord_path = "coords"
    image = cv2.imread(cat_image_path)
    with open(coord_path) as f:
        coords = np.array([int(x) for x in f.read().split()[1:]]).reshape((9, 2))
    bow_filter(image, coords)

