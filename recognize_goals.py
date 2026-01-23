#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import cv2
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

class GoalRecognizer:
    def __init__(self, model_path, label_map_path):
        self.detection_model = tf.saved_model.load(model_path)
        self.category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)

    def recognize_goal(self, image_np):
        input_tensor = tf.convert_to_tensor(image_np)
        detections = self.detection_model(input_tensor)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        image_np_with_detections = image_np.copy()
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            self.category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30,
            agnostic_mode=False)

        return image_np_with_detections

if __name__ == "__main__":
    recognizer = GoalRecognizer('exported-models/my_model/saved_model', 'label_map.pbtxt')

    cap = cv2.VideoCapture('test_footage.mp4')
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = recognizer.recognize_goal(frame_rgb)

        cv2.imshow('frame', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

