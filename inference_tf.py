#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import logging as log
import numpy as np
import tensorflow as tf

class NetworkTf:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### Initialize any class variables desired ###
        self.detection_graph = None
        self.default_graph = None
        self.sess = None
        self.image_tensor = None
        self.detection_boxes = None
        self.detection_scores = None
        self.detection_classes = None
        self.num_detections = None
        self.output_blob = None

    def load_model(self, model, device=None, cpu_extension=None):
        '''
        Load the model given IR files.
        Synchronous requests made within.
        '''
        ### Load the model ###
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        
        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)
                
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        
        self.output_blob = np.empty([1, 1, 100, 7])
            
        ### Return the loaded inference plugin ###
        return

    def get_input_shape(self):
        '''
        Gets the input shape of the network
        '''
        ### Return the shape of the input layer ###
        return [0,0,1280,720] # TODO if we want to generalize this script
    
    def exec_net(self, image):
        p_frame = np.expand_dims(image, axis=0)
        ### Start a synchronous request ###
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: p_frame})
        # we convert it to an output of shape (1, 1, 100, 7)
        for index, box in enumerate(boxes[0]):
            self.output_blob[0][0][index][0] = 0
            self.output_blob[0][0][index][1] = classes[0][index]
            self.output_blob[0][0][index][2] = scores[0][index]
            self.output_blob[0][0][index][3] = box[1]
            self.output_blob[0][0][index][4] = box[0]
            self.output_blob[0][0][index][5] = box[3]
            self.output_blob[0][0][index][6] = box[2]
        return

    def wait(self):
        '''
        Wait for the request to be complete.
        '''
        # TODO : here it's synchronous
        return 0

    def get_output(self):
        ### Extract and return the output results
        return self.output_blob