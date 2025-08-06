import json
import os
import numpy as np
import cv2
import tqdm
from constant import JSON_DIR, MASK_DIR

# Create a list which contains every file name in "jsons" folder
json_list = os.listdir(JSON_DIR)

""" tqdm Example Start"""
iterator_example = range(1000000)
for i in tqdm.tqdm(iterator_example):
    pass
""" tqdm Example End"""

# For every json file
for json_name in tqdm.tqdm(json_list):
    # Access and open json file as dictionary
    json_path = os.path.join(JSON_DIR, json_name)
    json_file = open(json_path, 'r')

    # Load json data
    json_dict = json.load(json_file)

    # Create an empty mask whose size is the same as the original image's size
    #########################################
    # CODE
    #########################################
    # Get image width and height from JSON
    image_height = json_dict["size"]["height"]
    image_width = json_dict["size"]["width"]

    # Create empty mask (black image)
    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    # For every objects
    for obj in json_dict["objects"]:
        # Check the objects 'classTitle' is 'Freespace' or not.
        if obj['classTitle'] == 'Freespace':
            #########################################
            # CODE
            #########################################
            # Get the points from the polygon
            points = obj['points']['exterior']

            # Convert points to numpy array format for cv2
            # Points come as [x, y] pairs, convert to proper format
            polygon_points = np.array(points, dtype=np.int32)

            # Fill the polygon area on mask with white (255)
            cv2.fillPoly(mask, [polygon_points], color=255)

    # Write mask image into MASK_DIR folder
    #########################################
    # CODE
    #########################################
    # Create output filename (replace .json with .png)
    mask_name = json_name.replace('.json', '.png')
    mask_path = os.path.join(MASK_DIR, mask_name)

    # Save the mask image
    cv2.imwrite(mask_path, mask)

    # Close JSON file
    json_file.close()