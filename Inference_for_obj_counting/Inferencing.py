import os
import time
from datetime import datetime
import cv2

from Inference_for_obj_counting import tensor_logic


# from Resolute_utils import utils
# from Resolute_Dash_Analysis import new_dash



# resolution = ()
# result_video_path = 'result_vid'
#out_file_name = os.path.join(result_video_path, 'result_' + '_' + video_file)




def draw_roi(frame, rois, color_towel, color_defect):

    for roi in rois:

        ## creating the centroid
        center = (int((roi[0] + roi[2]) / 2),
                  int((roi[1] + roi[3]) / 2))
        cv2.circle(frame,
                   center,
                   2,
                   (0, 0, 255),
                   1)

        ## creating the bounding box
        if roi[4] == 0:  # box
            rect_pt_1 = (roi[0], roi[1])
            rect_pt_2 = (roi[2], roi[3])
            cv2.rectangle(frame,
                          rect_pt_1,
                          rect_pt_2,
                          color_towel,
                          thickness=2)
            bbox_message = 'label'

        ## Putting text
        cv2.putText(frame,
                    bbox_message,
                    (roi[0] + 30, roi[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    lineType=cv2.LINE_AA)


def draw_counting_lines(frame, roi_left, roi_upper, roi_right, roi_lower):
    cv2.line(frame, (roi_left, roi_upper), (roi_right, roi_upper), (0, 255, 0), 1)  # green roi
    cv2.line(frame, (roi_left, roi_lower), (roi_right, roi_lower), (0, 255, 0), 1)  # blue roi

    cv2.line(frame, (roi_left, roi_upper), (roi_left, roi_lower), (0, 0, 255), 1)  # red roi
    cv2.line(frame, (roi_right, roi_upper), (roi_right, roi_lower), (0, 0, 255), 1)  # red roi


def analyse(frame, rois, total_packet_count, box_threshold, frame_no, prev_bottle):

    analysis = {}
    roi_upper = 578
    roi_lower = 600
    roi_left = 0
    roi_right = 2000
    boxes = 0
    date = time.asctime(time.gmtime())[3:-14]
    timer = time.asctime(time.gmtime())[10:-5]

    # drawing 2 counting logic lines
    draw_counting_lines(frame, roi_left, roi_upper, roi_right, roi_lower)

    # roi is a alias for a detected object
    # so correct saying is
    # for dectction_box in many_detected_boxes
    for roi in rois:
        # creating the centroid
        centroid = (int((roi[0] + roi[2]) / 2), int((roi[1] + roi[3]) / 2))

        # if centroid is inside the roi region
        if roi[4] == 0 and roi_upper < centroid[1] < roi_lower and roi_left < centroid[0] < roi_right and roi[
            4] == 0:  # boxes
            boxes = boxes + 1
            print("bottles",boxes, prev_bottle)

    if boxes == 0:
        box_threshold = 0

    if box_threshold > 1 and prev_bottle >= 1:
        box_threshold = 0

    if boxes > 0:
        box_threshold = box_threshold + 1

    if box_threshold == 1:  # change
        total_packet_count = total_packet_count + boxes
        #         print("Total count",total_packet_count)

    if box_threshold > 1 and 0 < prev_bottle < boxes:
        total_packet_count = total_packet_count + (boxes - prev_bottle)

    prev_bottle = boxes

    analysis["Packets"] = total_packet_count * 1
    analysis["Date"] = date
    analysis["Date"] = date
    analysis["Time"] = timer

    return analysis, frame, rois, total_packet_count, box_threshold, frame_no, prev_bottle


def process_lines(project_path, video_name, out_folder,
                  draw_object = True, video_out = True):
    try:
        rotate_90_clockwise = 0
        prev_bottle = 0

        # color = (B,G,R)
        color_object = (0,0,255)
        color_defect = (0,0,255)
        counter = 1
        fpers = None
        total_packet_count = 0
        frame_no = None
        box_threshold = 0
        logo_path = "logo_new2.jpg"
        logo = cv2.imread(logo_path)
        print("*"*50)# take from config
        print(os.path.join(project_path, video_name))
        print("*" * 50)

        # cap = utils.get_video(project_path, video_name)
        print("video opened")

        while cap.isOpened():
            ret, frame = cap.read()
            start_time = time.time()

            if ret:
                if counter % 1==0:

                    if rotate_90_clockwise ==1:
                        frame = cv2.resize(frame, (1080, 1920))
                        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    else:
                        frame = cv2.resize(frame, (1920, 1080))

                    st_time = time.time()
                    rois = tensor_logic.detect_box(frame, 0.25, project_path)
                    print("box detected..")

                    analysis, frame, rois, total_packet_count, box_threshold, frame_no, prev_bottle = analyse(
                                                                                                    frame,
                                                                                                    rois,
                                                                                                    total_packet_count,
                                                                                                    box_threshold,
                                                                                                    frame_no,
                                                                                                    prev_bottle)
                    if draw_object:
                        draw_roi(frame, rois, color_object, color_defect)

                    # whohle dashboard part comes from this line
                    # frame = new_dash.add_area(frame, logo)
                    # new_dash.write_analysis(frame, analysis)

                    frame_no = counter
                    print(f"Frame no: {counter}")
                    cv2.imwrite(os.path.join(out_folder, str(counter)+'.jpg'), frame)

                    # if video_out:
                    #     utils.output_frames(frame)
            else:
                cap.release()

            counter += 1

            if (time.time() - start_time) != 0:
                fpers = round(1.0 / (time.time() - start_time), 2)
            else:
                fpers = 0

            # new_dash.put_fps_in_dash(frame, fpers)
            print("FPS :", fpers)
        print("done")

    except Exception as e:
        print(e)


def main_inferencing(project_path):
    # logo_path = "logo_new2.jpg"
    # data_path = os.path.join('video_data')
    # processed_frames_path = 'out'
    # utils.is_path(processed_frames_path)

    #data_path = os.path.join(project_path)


    # result_video_path = 'result_vid'
    # utils.is_path(result_video_path)

    locations = os.listdir(project_path)
    print(locations)
    videos = [i for i in locations if i.startswith("input_video")]
    print("*"*50)
    print(videos)
    print("*" * 50)

    for video in videos:

        output_video_name = os.path.join(project_path, 'result_video' + '_' + video.split("input_video_")[1])

        processed_frames_path = os.path.join(project_path, 'out', 'box')
        os.makedirs(processed_frames_path, exist_ok=True)

        start = datetime.now()

        print(project_path,"\nto process lines")
        process_lines(project_path, video, processed_frames_path)

        # utils.make_video(processed_frames_path, output_video_name, fps = 20)

        stop = datetime.now()
        print(f"Time taken: {stop-start}")





if __name__ == "__main__":

    print("This file is not meant for stand alone usage,\
     funciotns from this file are being used for inferencing..")


#
# locations = os.listdir(project_path)
#     print("*"*50)
#     print(locations)
#     print("*" * 50)
#
#     for location in locations:
#
#         #video_files = os.listdir(os.path.join(data_path, location))
#         video_files = ["small_bottles.MOV"]
#         print(video_files)
#
#         for video_file in video_files:
#
#             out_file_name = os.path.join(result_video_path, 'result_' + '_' + video_file)
#
#             out_folder = "out/box"
#             utils.is_path(out_folder)
#
#             start = datetime.now()
#
#             print(data_path,"\nto process lines")
#             process_lines(data_path, video_file, out_folder, weight_folder_name)
#
#             utils.make_video(out_folder, out_file_name, fps = 30)
#
#             stop = datetime.now()
#             print(f"Time taken: {stop-start}")