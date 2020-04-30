import numpy as np
import cv2
from skimage.transform import resize
#from imutils import rotate_bound
from convenience import rotate_bound

def bow_filter(image, coords, bow_filter_path="filters/bow.png", output_image_path="cat_bow.png"):
    """
    Overlays a bow filter over an image of a cat. 
    :param image - np.array of a cat image.
    :param coords - np.array of keypoint coordinates of shape (9, 2).
    :param bow_filter_path - filepath to the bow image.
    :param output_image_path - filepath to store the filtered image.
    :return np.array of the filtered image.
    """
    # Store original image and filter.
    bgd = image
    fgd = cv2.imread(bow_filter_path, cv2.IMREAD_UNCHANGED)

    # Coordinate indices of ear tips.
    left_ear_tip = 4
    right_ear_tip = 7
    
    # Obtain x, y coordinates of ear tips.
    x_left = coords[left_ear_tip][0]
    y_left = coords[left_ear_tip][1]
    x_right = coords[right_ear_tip][0]
    y_right = coords[right_ear_tip][1]
    
    # Controls blending of filter with background.
    alpha = 1.0

    dx = float(x_right - x_left)
    dy = float(y_right - y_left)
    
    # Flip the image to handle rotations larger than 90 degrees.
    flip = False
    if dx < 0:
        flip = True
        bgd = np.flipud(bgd)
        y_left = bgd.shape[0]-1-y_left
        y_right = bgd.shape[0]-1-y_right
        x_left, x_right = x_right, x_left
        y_left, y_right = y_right, y_left
    
    # Resize the filter based on distance between ear tips.
    size = int(np.linalg.norm([dx, dy]))
    fgd = resize(fgd, (size, size, 4))
    resized_size = fgd.shape[0]
    
    # Incline angle of line connecting the ear tips in radians.
    angle = np.arctan(abs(dy)/abs(dx))
    
    # Rotate the filter clockwise (angle in degrees).
    rotate_angle = angle*180./np.pi
    if dy < 0:
        rotate_angle = 360 - rotate_angle

    fgd = rotate_bound(fgd, rotate_angle)
    rotated_size = fgd.shape[0]
    
    # Position the filter.
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
    
    # Offsets for indexing the filter (in case parts of the filter are out of bounds).
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
    
    # Determine bounds for filter placement in the original image.
    roi_top = np.clip(y_left, 0, bgd.shape[0]-1)
    roi_bottom = np.clip((y_left + fgd.shape[0]), 0, bgd.shape[0]-1)
    roi_left = np.clip(x_left, 0, bgd.shape[1]-1)
    roi_right = np.clip((x_left + fgd.shape[1]), 0, bgd.shape[1]-1)
    roi = bgd[roi_top:roi_bottom, roi_left:roi_right]
    
    # Overlay filter on the original image.
    for r in range(roi.shape[0]):
        for c in range(roi.shape[1]):
            # Alpha channel separates filter from its transparent background.
            if fgd[r + y_offset][c + x_offset][3] > 0:
                roi[r, c, :] = alpha*fgd[r + y_offset, c + x_offset, :-1]*255. + (1-alpha)*roi[r, c, :]
    
    # Flip image back if necessary.
    if flip:
        bgd = np.flipud(bgd)

    cv2.imwrite(output_image_path, bgd)
    return bgd

if __name__ == "__main__":
    for i in range(9):
        print(i)
        cat_image_path = "cats/cat_" + str(i) + ".jpg"
        coord_path = "cats/coords_" + str(i)
        image = cv2.imread(cat_image_path)
    
        #image = np.flipud(image)

        with open(coord_path) as f:
            coords = np.array([int(x) for x in f.read().split()[1:]]).reshape((9, 2))
        
        #temp = np.copy(coords[4])
        #coords[4] = coords[7]
        #coords[7] = temp
        #coords[4][1] = image.shape[0] - 1 - coords[4][1]
        #coords[7][1] = image.shape[0] - 1 - coords[7][1]

        bow_filter(image, coords, output_image_path="cat_bow_" + str(i) + ".png")

