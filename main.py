import numpy as np
import os
import winsound
import cv2
import yaml
from yaml.loader import SafeLoader
import argparse

# Construct the argument parser
ap = argparse.ArgumentParser()
# Add the arguments to the parser
ap.add_argument("-c", "--config", required=True,
   help="config file ")
args = vars(ap.parse_args())

#config_path = "configs/old_trim.yaml"
config_path = args['config']
print('#'*50)
print(config_path)

with open(config_path) as config_file:
    content = yaml.load(config_file.read(), Loader=SafeLoader)




dict_names = {
            0:"person",
            6:"train",
            9:"traffic_light",
            16:"dog",
            17:"horse",
            18:"sheep",
            19:"cow"
            }

from Inference_for_obj_counting import tensor_logic

input_video = os.path.join(content['paths']['video_folder'],content['paths']['video_name'])
# input_video = os.path.join("data","old_lady_train.mp4")
cap = cv2.VideoCapture(input_video)
ret, img = cap.read()
#rows, cols, _ = img.shape

count = 0

area_of_interest = [[(content['polygon']['x1'],content['polygon']['y1']),
                     (content['polygon']['x2'],content['polygon']['y2']),
                     (content['polygon']['x3'],content['polygon']['y3']),
                     (content['polygon']['x4'],content['polygon']['y4'])
                    ]]
new_area_of_interest = [[(content['polygon']['x1']//2,content['polygon']['y1']//2),
                         (content['polygon']['x2']//2,content['polygon']['y2']//2),
                         (content['polygon']['x3']//2,content['polygon']['y3']//2),
                         (content['polygon']['x4']//2,content['polygon']['y4']//2)
                        ]]
# this is because frame rate is changing
while True:
    count+=1
    if count%7!=0:  # skip frames condition
        continue

    ret, frame = cap.read()

    if not ret:
        break

    # draw the boundaries for railway route
    cv2.polylines(frame, np.array(area_of_interest, np.int32),
                  isClosed=True,
                  color=tuple(content['color']['polygon']),
                  thickness=content['thickness']['polygon'])
    # drop the unwanted part of the frame
    #frame = frame[:, 285:cols]
    height, width, _ = frame.shape
    # resize the frame
    frame = cv2.resize(frame, (width//2, height//2))
    height, width, _ = frame.shape
    # out_boxes -> coordinates
    # out_scores -> scores
    # out_classes -> classes
    # num_boxes -> number of predictions
    out_boxes, out_scores, out_classes, num_boxes = list(tensor_logic.detect_box(frame))
    # "detections" will store the detection coordinates for every object detected in the ongoing frame
    #detections = []


    for pred in range(num_boxes[0]):
        if int(out_classes[0][pred]) in [0,6,9,16,17,18,19]:
            if out_scores[0][pred] > content['params']['threshold']:
                xmin = int(out_boxes[0][pred][1] * width)
                ymin = int(out_boxes[0][pred][0] * height)
                xmax = int(out_boxes[0][pred][3] * width)
                ymax = int(out_boxes[0][pred][2] * height)
                class_id = int(out_classes[0][pred])
                # detections.append([xmin, ymin, xmax, ymax, class_id])
                # print(class_id, out_scores[0][pred])

                center = (int((xmin + xmax) / 2), int((ymax + ymax) / 2))

                result = cv2.pointPolygonTest(np.array(new_area_of_interest, np.int32), center, False) # false as we do not want the distance bet poly and obj
                # print("*"*20)
                # print(result)

                if result >=0: # draw box in red color
                    # draw the bounding boxes for all the detections
                    rect_pt_1 = (xmin, ymin)
                    rect_pt_2 = (xmax, ymax)

                    cv2.rectangle(frame,
                                  rect_pt_1, rect_pt_2,
                                  color=tuple(content['color']['rect_in']),
                                  thickness=content['thickness']['rect_in'])
                    # draw a centroid
                    cv2.circle(frame, center, radius=2,
                               color=tuple(content['color']['centroid_in']),
                               thickness=content['thickness']['centroid_in'])
                    # put text
                    cv2.putText(frame, dict_names[class_id],
                                (xmin + 5, ymin - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color=tuple(content['color']['text_in']),
                                thickness=content['thickness']['text_in'],
                                lineType=cv2.LINE_AA)
                    ##################################################################
                    # ALERT THE TRESSPASSING WITH SOUND
                    winsound.Beep(500, 100)
                    #winsound.PlaySound('alert.wav', winsound.SND_ASYNC)

                else: # draw the box in green color
                    # draw the bounding boxes for all the detections
                    rect_pt_1 = (xmin, ymin)
                    rect_pt_2 = (xmax, ymax)
                    cv2.rectangle(frame,
                                  rect_pt_1, rect_pt_2,
                                  color=tuple(content['color']['rect']),
                                  thickness=content['thickness']['rect'])

                    # draw a centroid
                    cv2.circle(frame, center,
                               radius=2,
                               color=tuple(content['color']['centroid']),
                               thickness=content['thickness']['centroid'])

                    # put text
                    cv2.putText(frame,
                                dict_names[class_id],
                                (xmin + 5, ymin - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color=tuple(content['color']['text']),
                                thickness=content['thickness']['text'],
                                lineType=cv2.LINE_AA)

    # show image
    cv2.imshow("img", frame)
    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()


#Inferencing.main_inferencing(project_path)

































































































