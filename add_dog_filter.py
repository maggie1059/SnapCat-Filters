import numpy as np
import cv2
from skimage.transform import resize
from imutils import rotate_bound


def add_ear_filter(image, coords, left, right, top, filter_path, which_ear):
    """
    left = index for bottom left of ear in coords list
    right = index for bottom right of ear in coords list
    top = index for top of ear in coords list"""

    bgd = image
    fgd = cv2.imread(filter_path, cv2.IMREAD_UNCHANGED)

    ear_bottom_left = left
    ear_bottom_right = right
    ear_top = top

    x_left = coords[ear_bottom_left][0]
    y_left = coords[ear_bottom_left][1]
    x_right = coords[ear_bottom_right][0]
    y_right = coords[ear_bottom_right][1]
    x_top = coords[ear_top][0]
    y_top = coords[ear_top][1]
    alpha = 1.0

    dx = float(x_right - x_left)
    dy = float(y_right - y_left)

    width = int(np.linalg.norm([dx, dy]))
    # calculate distance of midpoint of left and right to top of ear
    if which_ear == "left":
        bottom_midpoint = coords[ear_bottom_left] + 0.5 *(coords[ear_bottom_left] - coords[ear_bottom_right])
    else:
        bottom_midpoint = coords[ear_bottom_right] + 0.5 *(coords[ear_bottom_right] - coords[ear_bottom_left])
    height = int(np.linalg.norm(coords[ear_top]-bottom_midpoint))
    fgd = resize(fgd, (height,int(width*1.3), 4))
    resized_size = fgd.shape[0]

    angle = np.arctan(abs(dy)/dx)

    if dy >= 0:
        rotate_angle = angle*180./np.pi
    else:
        rotate_angle = 360 - angle*180./np.pi

    fgd = rotate_bound(fgd, rotate_angle)
    rotated_size = fgd.shape[0]
    
    if dy < 0:
        vertical_shift = rotated_size - resized_size + fgd.shape[0] / 3
        horizontal_shift = int(angle*(rotated_size - resized_size + fgd.shape[1]/2.3) * np.sin(angle))
        x_left = x_left - int(np.sin(angle)*vertical_shift)
        y_left = y_left - int(np.sin(angle)*horizontal_shift)
        x_left = x_left + int(np.cos(angle)*horizontal_shift)
        y_left = y_left - int(np.cos(angle)*vertical_shift)
    else:
        vertical_shift = int(1.8*(rotated_size - resized_size) - angle*fgd.shape[1]/1.5)
        horizontal_shift = -(rotated_size - resized_size) - fgd.shape[1] / 20
        x_left = x_left + int(np.sin(angle)*vertical_shift)
        y_left = y_left + int(np.sin(angle)*horizontal_shift)
        x_left = x_left + int(np.cos(angle)*horizontal_shift)
        y_left = y_left - int(np.cos(angle)*vertical_shift)
    
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

    if which_ear == "left":
        roi_top_y = int(y_top - height * 0.4)
        roi_top_x = int(x_top - height * 0.6)
    else:
        roi_top_y = int(y_top - height * 0.4)
        roi_top_x = int(x_top - height * 0.6)
        
    roi = bgd[np.clip(roi_top_y, 0, bgd.shape[0]-1):np.clip((roi_top_y+fgd.shape[0]), 0, bgd.shape[0]-1), np.clip(roi_top_x, 0, bgd.shape[1]-1):np.clip((roi_top_x+fgd.shape[1]), 0, bgd.shape[1]-1)]

    for r in range(roi.shape[0]):
        for c in range(roi.shape[1]):
            if fgd[r + y_offset][c + x_offset][3] > 0: # if it's not transparent
                for i in range(3):
                    roi[r][c][i] = alpha*fgd[r + y_offset][c + x_offset][i]*255. + (1-alpha)*roi[r][c][i]

    # copy back into original image
    bgd[np.clip(roi_top_y, 0, bgd.shape[0]-1):np.clip((roi_top_y+fgd.shape[0]), 0, bgd.shape[0]-1), np.clip(roi_top_x, 0, bgd.shape[1]-1):np.clip((roi_top_x+fgd.shape[1]), 0, bgd.shape[1]-1)] = roi

    return bgd

def add_nose_filter(image, coords, filter_path="filters/dog_nose.png"):
    bgd = image
    fgd = cv2.imread(filter_path, cv2.IMREAD_UNCHANGED)

    left_eye = 0
    right_eye = 1
    nose = 2

    x_left = coords[left_eye][0]
    y_left = coords[left_eye][1]
    x_right = coords[right_eye][0]
    y_right = coords[right_eye][1]
    x_nose = coords[nose][0]
    y_nose = coords[nose][1]
    alpha = 1.0

    dx = float(x_right - x_left)
    dy = float(y_right - y_left)

    width = int(np.linalg.norm([dx, dy]))
    midpoint = coords[left_eye] + 0.5 *(coords[left_eye] - coords[right_eye])
    height = int(np.linalg.norm(coords[nose]-midpoint))

    fgd = resize(fgd, (int(height*.6),int(width*1.2), 4))
    resized_size = fgd.shape[0]

    angle = np.arctan(abs(dy)/dx)

    if dy >= 0:
        rotate_angle = angle*180./np.pi
    else:
        rotate_angle = 360 - angle*180./np.pi

    fgd = rotate_bound(fgd, rotate_angle)
    rotated_size = fgd.shape[0]
    
    if dy < 0:
        vertical_shift = rotated_size - resized_size + fgd.shape[0] / 3
        horizontal_shift = int(angle*(rotated_size - resized_size + fgd.shape[1]/2.3) * np.sin(angle))
        x_left = x_left - int(np.sin(angle)*vertical_shift)
        y_left = y_left - int(np.sin(angle)*horizontal_shift)
        x_left = x_left + int(np.cos(angle)*horizontal_shift)
        y_left = y_left - int(np.cos(angle)*vertical_shift)
    else:
        vertical_shift = int(1.8*(rotated_size - resized_size) - angle*fgd.shape[1]/1.5)
        horizontal_shift = -(rotated_size - resized_size) - fgd.shape[1] / 20
        x_left = x_left + int(np.sin(angle)*vertical_shift)
        y_left = y_left + int(np.sin(angle)*horizontal_shift)
        x_left = x_left + int(np.cos(angle)*horizontal_shift)
        y_left = y_left - int(np.cos(angle)*vertical_shift)
    
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

    roi_top_y = int(y_nose - fgd.shape[0]*.7)
    roi_top_x = int(x_nose - fgd.shape[1]/2)

    roi = bgd[np.clip(roi_top_y, 0, bgd.shape[0]-1):np.clip((roi_top_y+fgd.shape[0]), 0, bgd.shape[0]-1), np.clip(roi_top_x, 0, bgd.shape[1]-1):np.clip((roi_top_x+fgd.shape[1]), 0, bgd.shape[1]-1)]


    for r in range(roi.shape[0]):
        for c in range(roi.shape[1]):
            if fgd[r + y_offset][c + x_offset][3] > 0: 
                for i in range(3):
                    roi[r][c][i] = alpha*fgd[r + y_offset][c + x_offset][i]*255. + (1-alpha)*roi[r][c][i]

    # copy back into original image
    bgd[np.clip(roi_top_y, 0, bgd.shape[0]-1):np.clip((roi_top_y+fgd.shape[0]), 0, bgd.shape[0]-1), np.clip(roi_top_x, 0, bgd.shape[1]-1):np.clip((roi_top_x+fgd.shape[1]), 0, bgd.shape[1]-1)] = roi

    return bgd


def dog_filter(image, coords, output_image_path="cat_dog.png"):
    bgd = add_ear_filter(image, coords, 3, 5, 4, "filters/dog_left_ear.png", "left")
    bgd = add_ear_filter(image, coords, 6, 8, 7, "filters/dog_right_ear.png", "right")
    bgd = add_nose_filter(image, coords)
    
    cv2.imwrite(output_image_path, bgd)

if __name__ == "__main__":
    for i in range(5):
        print(i)
        cat_image_path = "cats/cat_" + str(i) + ".jpg"
        coord_path = "cats/coords_" + str(i)
        image = cv2.imread(cat_image_path)
        with open(coord_path) as f:
            coords = np.array([int(x) for x in f.read().split()[1:]]).reshape((9, 2))
        dog_filter(image, coords, output_image_path="cat_dog_" + str(i) + ".png")

