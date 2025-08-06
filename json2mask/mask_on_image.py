import os
import cv2
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from constant import *

# Create a list which contains every file name in masks folder
mask_list = os.listdir(MASK_DIR)

# Remove hidden files if any
for f in mask_list:
    if f.startswith('.'):
        mask_list.remove(f)

# For every mask image
for mask_name in tqdm.tqdm(mask_list):
    # Name without extension
    mask_name_without_ex = mask_name.split('.')[0]

    # Access required folders
    mask_path = os.path.join(MASK_DIR, mask_name)
    image_path = os.path.join(IMAGE_DIR, mask_name_without_ex + '.jpeg')
    image_out_path = os.path.join(IMAGE_OUT_DIR, mask_name)

    # Read mask and corresponding original image
    #########################################
    # CODE
    #########################################
    # Read mask as grayscale
    mask = cv2.imread(mask_path, 0)

    # Read original image
    image = cv2.imread(image_path)

    # Check if both files exist
    if mask is None or image is None:
        print(f"Could not read files for {mask_name_without_ex}")
        continue

    # Change the color of the pixels on the original image that corresponds
    # to the mask part and create new image
    #########################################
    # CODE
    #########################################
    # Create a copy of original image
    masked_image = image.copy()

    # Apply mask: where mask is white (255), make the image green
    # Create a green overlay
    green_overlay = np.zeros_like(image)
    green_overlay[:, :] = [0, 255, 0]  # Green color in BGR

    # Create mask condition (where mask pixels are white/255)
    mask_condition = mask == 255

    # Apply green color where mask is white with some transparency
    alpha = 0.5  # Transparency factor
    masked_image[mask_condition] = (1 - alpha) * image[mask_condition] + alpha * green_overlay[mask_condition]

    # Write output image into IMAGE_OUT_DIR folder
    #########################################
    # CODE
    #########################################
    cv2.imwrite(image_out_path, masked_image)

    # Visualize created image if VISUALIZE option is chosen
    if VISUALIZE:
        #########################################
        # CODE
        #########################################
        # Create subplot for visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Mask
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Mask')
        axes[1].axis('off')

        # Masked image
        axes[2].imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
        axes[2].set_title('Masked Image')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()
        plt.pause(1)  # Show for 1 second