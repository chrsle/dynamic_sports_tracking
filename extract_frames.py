#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import cv2
import argparse
import logging
from tqdm import tqdm

def setup_logging():
    """
    Setup logging configuration
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def create_directory(path):
    """
    Create directory if it does not exist
    """
    if not os.path.exists(path):
        os.makedirs(path)

def extract_frames(video_path, output_path, seconds):
    """
    Extracts frames from a video.
    """
    # Ensure output directory exists
    create_directory(output_path)

    # Open the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        logging.error(f"Cannot open video: {video_path}")
        return

    # Get the frames per second of the video
    fps = video.get(cv2.CAP_PROP_FPS)

    # Determine the frame interval based on the desired seconds between each extracted frame
    frame_interval = int(fps * seconds)

    frame_count = 0
    while True:
        # Read the next frame
        ret, frame = video.read()
        
        # If the frame was not successfully read, then we have reached the end of the video
        if not ret:
            break

        # If this is an interval frame, write it to a JPEG file
        if frame_count % frame_interval == 0:
            output_filename = f"{output_path}/frame{frame_count}.jpg"
            cv2.imwrite(output_filename, frame)
            logging.info(f"Frame saved: {output_filename}")

        frame_count += 1

    # Release the video file
    video.release()

if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser(description="Frame Extractor")
    parser.add_argument("video", help="Path to the video file")
    parser.add_argument("output", help="Directory to save the frames")
    parser.add_argument("--seconds", type=int, default=1, help="Seconds between each extracted frame")
    args = parser.parse_args()

    extract_frames(args.video, args.output, args.seconds)

