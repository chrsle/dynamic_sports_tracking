#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import argparse
from detection_model_utils import load_model, run_inference_for_single_image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model.')
    parser.add_argument('--test_image', type=str, required=True, help='Path to the test image.')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load the trained model
    model = load_model(args.model_path)

    # Load the test image
    test_image = cv2.imread(args.test_image)

    # Run inference on the test image
    output_dict = run_inference_for_single_image(model, test_image)

    # Print the output_dict
    print(output_dict)

if __name__ == "__main__":
    main()

