#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.protos import eval_pb2
from object_detection.metrics import coco_evaluation
from object_detection.builders import eval_builder

def main():
    # Specify the paths to your pipeline config and evaluation data.
    pipeline_config_path = 'path/to/your/pipeline.config'
    eval_record_path = 'path/to/your/eval.record'
    label_map_path = 'path/to/your/label_map.pbtxt'
    model_dir = 'path/to/your/trained/model'

    # Load the configuration.
    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    model_config = configs['model']
    eval_config = configs['eval_config']
    eval_input_configs = configs['eval_input_configs'][0]

    # Update the input configurations with paths to TFRecords and label map.
    eval_input_configs.label_map_path = label_map_path
    eval_input_configs.tf_record_input_reader.input_path[:] = [eval_record_path]

    # Build the model.
    detection_model = model_builder.build(model_config=model_config, is_training=False)
    detection_model.restore_map(fine_tune_checkpoint_type='detection', load_all_detection_checkpoint_vars=True)

    # Load checkpoint of the trained model
    checkpoint = tf.train.Checkpoint(model=detection_model)
    checkpoint.restore(tf.train.latest_checkpoint(model_dir)).expect_partial()

    # Load label map data (for plotting).
    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=100, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Create evaluator
    evaluator_options = eval_pb2.EvaluatorOptions()
    evaluator_type = eval_config.metrics_set[0]
    evaluator = eval_builder.build(evaluator_options, input_config=eval_input_configs, model_config=model_config, categories=categories)

    # Run evaluation
    results = evaluator.evaluate(create_input_dict_fn, model_dir, graph_hook_fn=detection_model.get_graph_rewriter_fn())

    # Print the results
    for key, value in results.items():
        print(f'{key}: {value}')

if __name__ == '__main__':
    main()

