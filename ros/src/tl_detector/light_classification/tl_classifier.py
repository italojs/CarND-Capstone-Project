from styx_msgs.msg import TrafficLight

import numpy as np
import tensorflow as tf
from PIL import Image
import time
# from matplotlib import pyplot as plt
# import cv2

# import yolo_v3
#import yolo_v3_tiny

from utils import load_coco_names, draw_boxes, get_boxes_and_inputs, get_boxes_and_inputs_pb, non_max_suppression,load_graph, letter_box_image, convert_to_original_size

from filter_functions_img import red_filter, yellow_filter, green_filter, plotHist, color_isolate, matrix_scalar_mul, matrix_multiplication, max_idx, min_idx, max_idx_rank, is_bimodal, yaxis_hists, feature_value, analyze_color,state_predict

import os

options = {
    'input_image': '000208.jpg',
    'output_image': 'output1.jpg',
    'image_size': 416,
    'labels': 'coco.names',
    'frozen_model': 'frozen_darknet_yolov3_model.pb',
    'tiny': False,
    'data_format': 'NCHW',
    'ckpt_file': 'saved_model_big/model.ckpt',
    'iou': 0.4,
    'thresh': 0.5,
    'gpu': 0.5
}


class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=options['gpu'])

        print ('GPU options defined')

        self.config = tf.ConfigProto(
            gpu_options=gpu_options,
            log_device_placement=False,
        )

        dir_path = os.path.dirname(os.path.abspath(__file__))
        self.classes = load_coco_names(os.path.join(dir_path,options['labels']))

        t0 = time.time()
        self.frozenGraph = load_graph(os.path.join(dir_path,options['frozen_model']))

        self.sess =  tf.Session(graph=self.frozenGraph, config=self.config)
        print("Loaded graph in {:.2f}s".format(time.time()-t0))
        pass

    def get_classification(self, cv_image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction

        image = Image.fromarray(cv_image)
        img_resized = letter_box_image(image, options['image_size'], options['image_size'], 128)
        img_resized = img_resized.astype(np.float32)

        boxes, inputs = get_boxes_and_inputs_pb(self.frozenGraph)

        # with tf.Session(graph=self.frozenGraph, config=self.config) as sess:
        t0 = time.time()
        detected_boxes = self.sess.run(boxes, feed_dict={inputs: [img_resized]})
        filtered_boxes = non_max_suppression(detected_boxes,
                                            confidence_threshold=options['thresh'],
                                            iou_threshold=options['iou'])
        print("Predictions found in {:.2f}s".format(time.time() - t0))
        inp = filtered_boxes.get(9)
        inp_new = dict()
        inp_new[9] = inp

        if (inp_new[9] != None):
            if (len(inp_new[9])>0):
                for cls, bboxs in inp_new.items():
                    for box, score in bboxs:
                        box = convert_to_original_size(box, (options['image_size'], options['image_size']),
                                                    np.array(image.size),True)
                # print(inp_new)
                a = analyze_color(inp_new, cv_image)
                # print(a)
                light_color = state_predict(a)
                print("the light color is {}".format(light_color))
                if light_color:
                    if light_color == 'YELLOW':
                        return TrafficLight.YELLOW
                    elif light_color == 'RED':
                        return TrafficLight.RED
                    elif light_color == 'GREEN':
                        return TrafficLight.GREEN

        return TrafficLight.UNKNOWN
