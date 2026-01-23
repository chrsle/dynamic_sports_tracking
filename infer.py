#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import argparse
import numpy as np
from detection_model_utils import load_model, load_label_map
from recognize_goals import recognize_goals

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video', type=str, required=True, 
                        help='Path to the input video file.')
    return parser.parse_args()

def main():
    args = parse_args()
    model, category_index = load_model(), load_label_map()

    # Load the video.
    cap = cv2.VideoCapture(args.input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Convert the image from BGR color (which OpenCV uses) to RGB color.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_expanded = np.expand_dims(frame_rgb, axis=0)

        # Perform detection
        output_dict = model(frame_expanded)

        # Recognize goals
        goal_event = recognize_goals(output_dict, category_index)

        if goal_event:
            print(f"Goal detected at frame {frame_count}")

        frame_count += 1

    # Release the video file
    cap.release()

if __name__ == '__main__':
    main()

