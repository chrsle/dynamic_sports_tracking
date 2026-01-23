#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2

def create_trackers(algorithm='CSRT'):
    """
    Function to create an object tracker. CSRT is chosen by default, but this can be changed.
    """
    if algorithm == 'CSRT':
        tracker = cv2.TrackerCSRT_create()
    elif algorithm == 'KCF':
        tracker = cv2.TrackerKCF_create()
    elif algorithm == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    else:
        raise ValueError("Invalid tracking algorithm")

    return tracker

def initialize_tracker(tracker, frame, bbox):
    """
    Function to initialize a tracker with a frame and a bounding box.
    """
    tracker.init(frame, bbox)

def update_tracker(tracker, frame):
    """
    Function to update a tracker with a new frame.
    """
    success, bbox = tracker.update(frame)
    return success, bbox

def draw_tracking_result(frame, bbox, success):
    """
    Function to draw the result of tracking on a frame.
    """
    if success:
        # Draw the tracked object
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    else:
        cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

    return frame
