#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import imgaug.augmenters as iaa
import cv2
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing images to augment.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save augmented images.')
    return parser.parse_args()

def augment_images(image_dir, output_dir):
    # Define a few augmentations
    augmenters = iaa.Sequential([
        iaa.Rotate((-25, 25)),
        iaa.Scale((0.5, 1.5)),
        iaa.Fliplr(0.5),  # horizontally flip 50% of the images
    ])

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Loop over each image in the input directory
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = cv2.imread(os.path.join(image_dir, filename))
            img_aug = augmenters.augment_image(img)
            cv2.imwrite(os.path.join(output_dir, 'aug_' + filename), img_aug)

def main():
    args = parse_args()
    augment_images(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()

