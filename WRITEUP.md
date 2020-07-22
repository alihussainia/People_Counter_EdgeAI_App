# Project Write-Up

## Custom Layers

Custom layers are a necessary and important to have feature of the OpenVINO™ Toolkit, although one shouldn’t have to use it very often, if at all, due to all of the supported layers. 

The list of supported layers relates to whether a given layer is a custom layer. Any layer not in that list is automatically classified as a custom layer by the Model Optimizer.

To actually add custom layers, there are a few differences depending on the original model framework. 

You have three options for TensorFlow* models with custom layers:

- Register those layers as extensions to the Model Optimizer. In this case, the Model Optimizer generates a valid and optimized Intermediate Representation.
- If you have sub-graphs that should not be expressed with the analogous sub-graph in the Intermediate Representation, but another sub-graph should appear in the model, the Model Optimizer provides such an option. This feature is helpful for many TensorFlow models. To read more, see Sub-graph Replacement in the Model Optimizer.
- Experimental feature of registering definite sub-graphs of the model as those that should be offloaded to TensorFlow during inference. In this case, the Model Optimizer produces an Intermediate Representation that:

 - Can be inferred only on CPU 
 - Reflects each sub-graph as a single custom layer in the Intermediate Representation


## Comparing Model Performance

### Model No.1:

#### ssdlite_mobilenet_v2_coco

Source:

http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz

Code:

`python mo_tf.py --input_model frozen_inference_graph.pb
--tensorflow_use_custom_operations_config ssd_v2_support.json
--tensorflow_object_detection_api_pipeline_config ssd_mobilenet_v2_coco.config
--data_type FP16`

Metrics:

- Detected Persons : 681
- Probability Threshold : 0.6 
- Speed / time: 2m21s.


### Model No.2:

#### ssd_mobilenet_v2_coco

Source:

http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz

Code:

`python /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels`

Metrics:

- Detected Persons : 734
- Probability Threshold : 0.6
- Speed / Time : 3m10s 


### With and Without OpenVINO: 

inference_tf.py, is a Tensorflow based revised version of the inference.py and its Network class. 

Source:

(https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb)



Metrics (Without and With OpenVINO):

- Model1: ssd_mobilenet_v2_coco:
    - Probability Threshold : 0.6
    - Accuracy : 601 vs 681
    - Model Size : 17M vs 8.7M
    - Total count of persons : 9 vs 6
    - CPU usage : 71% vs 54%
    - Script Running Time : 3m10s vs 2m23s
    - Average presence time : 19s vs 23s
    - Total inference time : 225s vs 42s
    - Inference time frame : 159ms vs 32ms

- Model2: ssdlite_mobilenet_v2_coco:
    - Probability Threshold : 0.6
    - Accuracy : 565 vs 734
    - Model Size : 66M vs 64M
    - Total count of persons : 8 vs 6
    - CPU usage : 67% vs 38% 
    - Script Running Time : 4m10s vs 3m11s 
    - Average presence time : 19s vs 21s
    - Total inference time : 144s vs 43s
    - Inference time by frame : 102ms vs 29ms

Conclusion:

Based on the above results, it can easily be concluded that converting to IR makes a positive difference in the model performance.


Note:

Model accuracy is based upon total number of detections. 


###  Edge vs Cloud Services

- Network Dependency: Low vs High.

- Cost Effective: Free vs $9.99 per ML studio workspace per month $1 per studio experimentation hour on Azure Machine Learning Standard Cloud Services.


### Potential Applications and Importance

Potential Applications:

- Number of customers daily visiting the store.

- To know how many persons are present in a room like school class.

Importance:

- Employer can predict the most widely visited sections of the departmental stores and thus can use those sections primarily for any non-performing goods marketing.

- To find the number of absent students in each class and identify the class with most absentees in order to further investigate the reason why most students are absent in that class. 


### Assess Effects on End User Needs

Since my model was not specifically trained to recognize people, but it did quite well on still person, but didn't perform well when a person tried to move. So I would say that on one hand if Lighting, model accuracy, and camera focal length/image size matters, the best training of the model matters too, so that it can perform very well even in the low resource environments.
