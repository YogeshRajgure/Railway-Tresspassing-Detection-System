import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import os
# from main import content
# print(content)
#Fweights = 'checkpoints/label'
#def name_of_weight():


# with open(os.path.join("projects", "path_to_weight_for_current_project.txt"), 'r') as f:
#     project_path = f.read()

# os.path.join(project_path, 'saved_model.pb')
# weight_saved_here = "custom_yolov4_small_bottle.weights"
Fweights = os.path.join("_custom_tensor_weights_from_yolo","coco_tiny")   #os.path.join("./_custom_tensor_weights_from_yolo", (weight_saved_here).split(".")[0])  # converted .pb file directory

Fclasses = 'data/classes/custom.names'


# Flags
Ftiny = True
Fmodel = 'yolov4'
Fsize = 416
Fframework = 'tf'
Foutput_format = 'XVID'
Fiou = 0.50
Fscore = 0.30
Fdont_show = True

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(Fclasses, Ftiny, Fmodel)

saved_model_loaded = tf.saved_model.load(Fweights, tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']



def format_boxes(boxes, height, width, threshold):
    # out_boxes -> coordinates
    # out_scores -> scores
    # out_classes -> classes
    # num_boxes -> number of predictions
    out_boxes, out_scores, out_classes, num_boxes = list(boxes)
    rois = []

    for itr in range(num_boxes[0]):
        if out_scores[0][itr] > threshold:
            xmin = int(out_boxes[0][itr][1] * width)
            ymin = int(out_boxes[0][itr][0] * height)
            xmax = int(out_boxes[0][itr][3] * width)
            ymax = int(out_boxes[0][itr][2] * height)
            class_id = int(out_classes[0][itr])
            rois.append([xmin, ymin, xmax, ymax, class_id])

    return rois


def detect_box(frame):

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image_data = cv2.resize(frame, (Fsize, Fsize))
    image_data = image_data / 255.

    images_data = []
    images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)

    batch_data = tf.constant(images_data)
    pred_bbox = infer(batch_data)

    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                                                                                     scores=tf.reshape(pred_conf,
                                                                                                       ( tf.shape(pred_conf)[0],
                                                                                                         -1,
                                                                                                         tf.shape(pred_conf)[-1]
                                                                                                        )
                                                                                                       ),
                                                                                     max_output_size_per_class=50,
                                                                                     max_total_size=50,
                                                                                     iou_threshold=Fiou,
                                                                                     score_threshold=Fscore
                                                                                )

    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    # height, width, _ = frame.shape
    # rois = format_boxes(pred_bbox, height, width, threshold)
    #
    # return rois
    return pred_bbox