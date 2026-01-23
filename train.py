#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder
from object_detection.utils import dataset_util

def load_config(config_path):
    config = config_util.get_configs_from_pipeline_file(config_path)
    return config

def main():
    # Specify the paths to your pipeline config and training data.
    pipeline_config_path = 'path/to/your/pipeline.config'
    train_record_path = 'path/to/your/train.record'
    val_record_path = 'path/to/your/val.record'
    label_map_path = 'path/to/your/label_map.pbtxt'

    # Load the configuration.
    configs = load_config(pipeline_config_path)
    model_config = configs['model']
    train_config = configs['train_config']
    train_input_config = configs['train_input_config']
    eval_config = configs['eval_config']
    eval_input_configs = configs['eval_input_configs'][0]

    # Update the input configurations with paths to TFRecords and label map.
    train_input_config.label_map_path = label_map_path
    train_input_config.tf_record_input_reader.input_path[:] = [train_record_path]
    eval_input_configs.label_map_path = label_map_path
    eval_input_configs.tf_record_input_reader.input_path[:] = [val_record_path]

    # Build the model.
    detection_model = model_builder.build(model_config=model_config, is_training=True)

    # Create a Tensorflow session.
    sess = tf.Session()

    # Restore from a checkpoint if specified in the pipeline config.
    if train_config.fine_tune_checkpoint:
        var_map = detection_model.restore_map(
            fine_tune_checkpoint_type=train_config.fine_tune_checkpoint_type,
            load_all_detection_checkpoint_vars=train_config.load_all_detection_checkpoint_vars
        )
        available_var_map = (variables_helper.get_variables_available_in_checkpoint(
            var_map, train_config.fine_tune_checkpoint, include_global_step=False))
        init_saver = tf.train.Saver(available_var_map)
        init_saver.restore(sess, train_config.fine_tune_checkpoint)

    # Specify additional training parameters (e.g., learning rate, batch size)...

    # Start training...
    
    # Save the trained model.
    saver = tf.train.Saver()
    saver.save(sess, 'path/to/save/model')

if __name__ == '__main__':
    main()

