#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from object_detection.utils import dataset_util, label_map_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

def load_image_into_numpy_array(image):
    """
    Load an image into a numpy array.
    """
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def create_tf_example(example):
    """
    Create a tf.Example from a dictionary example.
    """
    # TODO: Fill in this function with your own code for creating a tf.Example
    pass

def create_tf_record(output_filename, examples):
    """
    Create a TFRecord file from a list of examples.
    """
    writer = tf.io.TFRecordWriter(output_filename)
    for example in examples:
        tf_example = create_tf_example(example)
        writer.write(tf_example.SerializeToString())
    writer.close()

def load_label_map(label_map_path):
    """
    Load a label map from a .pbtxt file.
    """
    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index

def load_config(config_path):
    """
    Load a pipeline config from a .config file.
    """
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)
    return pipeline_config

