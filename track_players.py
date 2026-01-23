#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video', type=str, required=True, help='Path to the input video.')
    return parser.parse_args()

def main():
    args = parse_args()

    # Create a tracker object
    tracker = cv2.TrackerCSRT_create()

    # Open the video file
    video = cv2.VideoCapture(args.input_video)

    # Read the first frame
    success, frame = video.read()
    if not success:
        print("Failed to read video")
        sys.exit(1)

    # Select a bounding box
    bbox = cv2.selectROI("Tracking", frame, False)
    tracker.init(frame, bbox)

    while True:
        # Read a new frame
        success, frame = video.read()
        if not success:
            break

        # Update the tracking result
        success, bbox = tracker.update(frame)

        if success:
            # Draw the tracked object
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        else:
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)

        # Display the frame with the tracked object
        cv2.imshow("Tracking", frame)

        # Exit if ESC key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Release the video capture and window
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

