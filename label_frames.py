#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import cv2
import argparse
import logging
from tqdm import tqdm

# setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def label_frames(input_dir, output_dir):
    """
    Label frames with their filename (without extension).
    """
    # ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # loop over all files in the input directory
    for filename in tqdm(os.listdir(input_dir), desc='Labeling frames', unit='frame'):
        # construct full file path
        file_path = os.path.join(input_dir, filename)

        # load the image
        image = cv2.imread(file_path)
        if image is None:
            logging.warning(f"Could not load image: {filename}")
            continue

        # label the image
        label = os.path.splitext(filename)[0]  # filename without extension
        cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # save the image
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, image)
        logging.info(f"Labeled image saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frame Labeller")
    parser.add_argument("input", help="Directory containing frames")
    parser.add_argument("output", help="Directory to save labeled frames")
    args = parser.parse_args()

    label_frames(args.input, args.output)

