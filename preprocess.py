# import numpy as np
# import pandas as pd
# import math




# # Data Augmentation by mirroring the images
# def augment(img, points):
#     f_img = img[:, ::-1]        # Mirror the image
#     for i in range(0,len(points),2):        # Mirror the key point coordinates
#         x_renorm = (points[i]+0.5)*96       # Denormalize x-coordinate
#         dx = x_renorm - 48          # Get distance to midpoint
#         x_renorm_flipped = x_renorm - 2*dx      
#         points[i] = x_renorm_flipped/96 - 0.5       # Normalize x-coordinate
#     return f_img, points

# aug_imgs_train = []
# aug_points_train = []
# for i, img in enumerate(imgs_train):
#     f_img, f_points = augment(img, points_train[i])
#     aug_imgs_train.append(f_img)
#     aug_points_train.append(f_points)
    
# aug_imgs_train = np.array(aug_imgs_train)
# aug_points_train = np.array(aug_points_train)

# # Combine the original data and augmented data
# imgs_total = np.concatenate((imgs_train, aug_imgs_train), axis=0)       
# points_total = np.concatenate((points_train, aug_points_train), axis=0)