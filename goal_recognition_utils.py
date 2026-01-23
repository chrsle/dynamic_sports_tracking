#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import cv2
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

def load_model(model_path):
    """
    Load a trained model from the specified path.
    """
    return tf.saved_model.load(model_path)

def load_label_map(label_map_path):
    """
    Load a label map from a .pbtxt file.
    """
    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
    return label_map_util.create_category_index(categories)

def process_video(model, category_index, video_path):
    """
    Process a video file and return the frames with goal detections visualized.
    """
    cap = cv2.VideoCapture(video_path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(frame_rgb)
        detections = model(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        frame_rgb_with_detections = frame_rgb.copy()
        viz_utils.visualize_boxes_and_labels_on_image_array(
            frame_rgb_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30,
            agnostic_mode=False)

        yield cv2.cvtColor(frame_rgb_with_detections, cv2.COLOR_RGB2BGR)

    cap.release()

