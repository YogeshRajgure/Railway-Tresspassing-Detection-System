from flask import Flask,render_template,Response, session, request, redirect, url_for
import cv2
import os
from Inference_for_obj_counting import tensor_logic
from utils.connector import mysql_connect as mysql
import winsound
import numpy as np
import yaml
from yaml.loader import SafeLoader


app = Flask(__name__)
app.secret_key = "super secret keys"


threshold = 0.6
dict_names = {
            0:"person",
            6:"train",
            9:"traffic_light",
            16:"dog",
            17:"horse",
            18:"sheep",
            19:"cow"
            }

input_video_1 = os.path.join("static", "videos", "people_crossing.mp4")
input_video_2 = os.path.join("static", "videos", "old_lady_train_Trim.mp4")
input_video_3 = os.path.join("static", "videos", "empty_crossing.mp4")
input_video_4 = os.path.join("static", "videos", "walking_on_sides.mp4")
# input_video_5 = os.path.join("static", "videos", "empty_crossing.mp4")
# input_video_6 = os.path.join("static", "videos", "walking_on_sides.mp4")
input_video_5 = 0
input_video_6 = 1

# input_video = os.path.join("data","old_lady_train.mp4")


# cap = cv2.VideoCapture(input_video)
#
# # this is because frame rate is changing
# area_of_interest = [[(288,566), (288,395), (1900,545), (1900,1068)]]
# new_area_of_interest = [[(288//2,566//2), (288//2,395//2), (1900//2,545//2), (1900//2,1068//2)]]




class Window:
    def __init__(self, video_loc, config_path):

        with open(config_path) as config_file:
            self.content = yaml.load(config_file.read(), Loader=SafeLoader)

        self.video_loc = video_loc
        self.cap = cv2.VideoCapture(self.video_loc)
        self.area_of_interest = [[(self.content['x1'], self.content['y1']),
                             (self.content['x2'], self.content['y2']),
                             (self.content['x3'], self.content['y3']),
                             (self.content['x4'], self.content['y4'])]]
        self.new_area_of_interest = [[(self.content['x1'] // 2, self.content['y1'] // 2),
                                 (self.content['x2'] // 2, self.content['y2'] // 2),
                                 (self.content['x3'] // 2, self.content['y3'] // 2),
                                 (self.content['x4'] // 2, self.content['y4'] // 2)]]

    def generate_frames(self):

        cap = self.cap
        # this is because frame rate is changing
        area_of_interest = self.area_of_interest
        new_area_of_interest = self.new_area_of_interest
        count = 0

        while True:
            count += 1
            if count % 7 != 0:  # skip frames condition
                continue

            ## read the camera frame
            ret, frame = cap.read()
            if not ret:
                break

            # draw the boundaries for railway route
            cv2.polylines(frame, np.array(area_of_interest, np.int32), True, (15, 220, 10), 6)
            # drop the unwanted part of the frame
            # frame = frame[:, 285:cols]
            height, width, _ = frame.shape
            # resize the frame
            frame = cv2.resize(frame, (width // 2, height // 2))
            height, width, _ = frame.shape
            # out_boxes -> coordinates
            # out_scores -> scores
            # out_classes -> classes
            # num_boxes -> number of predictions
            out_boxes, out_scores, out_classes, num_boxes = list(tensor_logic.detect_box(frame))
            # "detections" will store the detection coordinates for every object detected in the ongoing frame
            detections = []
            result = -1
            for pred in range(num_boxes[0]):
                if int(out_classes[0][pred]) in [0, 6, 9, 16, 17, 18, 19]:
                    if out_scores[0][pred] > threshold:
                        xmin = int(out_boxes[0][pred][1] * width)
                        ymin = int(out_boxes[0][pred][0] * height)
                        xmax = int(out_boxes[0][pred][3] * width)
                        ymax = int(out_boxes[0][pred][2] * height)
                        class_id = int(out_classes[0][pred])
                        detections.append([xmin, ymin, xmax, ymax, class_id])
                        # print(class_id, out_scores[0][pred])

                        center = (int((xmin + xmax) / 2), int((ymax + ymax) / 2))

                        result = cv2.pointPolygonTest(np.array(new_area_of_interest, np.int32), center,
                                                      False)  # false as we do not want the distance bet poly and obj
                        # print("*"*20)
                        # print(result)

                        if result >= 0:  # draw box in red color
                            # draw the bounding boxes for all the detections
                            rect_pt_1 = (xmin, ymin)
                            rect_pt_2 = (xmax, ymax)

                            cv2.rectangle(frame,
                                          rect_pt_1, rect_pt_2,
                                          color=(0, 0, 255), thickness=2)
                            # draw a centroid
                            cv2.circle(frame, center,
                                       radius=2, color=(255, 0, 0), thickness=5)
                            # put text
                            cv2.putText(frame,
                                        dict_names[class_id],
                                        (xmin + 5, ymin - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (0, 0, 0), 1,
                                        lineType=cv2.LINE_AA)
                            # ALERT THE TRESSPASSING WITH SOUND
                            winsound.Beep(500, 100)
                            # winsound.PlaySound('alert.wav', winsound.SND_ASYNC)

                        else:  # draw the box in green color
                            # draw the bounding boxes for all the detections
                            rect_pt_1 = (xmin, ymin)
                            rect_pt_2 = (xmax, ymax)
                            cv2.rectangle(frame,
                                          rect_pt_1, rect_pt_2,
                                          color=(255, 0, 0), thickness=2)

                            # draw a centroid
                            cv2.circle(frame, center,
                                       radius=2, color=(0, 0, 255), thickness=5)

                            # put text
                            cv2.putText(frame,
                                        dict_names[class_id],
                                        (xmin + 5, ymin - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (0, 0, 0), 1,
                                        lineType=cv2.LINE_AA)

            # show image
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # result

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



@app.route("/")
def index():
    return render_template("index.html")

@app.route('/multi_screen_x', methods=['GET','POST'])
def multi_screen_x():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Checking the above credentials with mysql db
        db = mysql.connect()
        cursor = db.cursor()
        cursor.execute('SELECT * FROM user WHERE username=%s AND password=%s', (username, password))
        record = cursor.fetchone()
        db.close()

        img_1 = os.path.join('static', 'images', '1.jpeg')
        # img_2 = this we are taking as video file
        img_3 = os.path.join('static', 'images', '2.jpeg')
        img_4 = os.path.join('static', 'images', '1.jpeg')
        img_5 = os.path.join('static', 'images', '2.jpeg')
        img_6 = os.path.join('static', 'images', '1.jpeg')
        img = [img_1, img_3, img_4, img_5, img_6]

        if record:
            session['loggedin'] = True
            session['username'] = record[1]
            return render_template('multi_screen_x.html', img = img)
        else:
            return render_template('index.html')

    return render_template('index.html')


@app.route('/video_1')
def video_1():
    return Response(v_1.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_2')
def video_2():
    return Response(v_2.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_3')
def video_3():
    return Response(v_3.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_4')
def video_4():
    return Response(v_4.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_5')
def video_5():
    return Response(v_5.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_6')
def video_6():
    return Response(v_6.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/multi_screen_redirected', methods=['GET','POST'])
def multi_screen_redirected():
    img_1 = os.path.join('static', 'images', '1.jpeg')
    # img_2 = this we are taking as video file
    img_3 = os.path.join('static', 'images', '2.jpeg')
    img_4 = os.path.join('static', 'images', '1.jpeg')
    img_5 = os.path.join('static', 'images', '2.jpeg')
    img_6 = os.path.join('static', 'images', '1.jpeg')
    img = [img_1, img_3, img_4, img_5, img_6]
    return render_template('multi_screen_x.html', img = img)

@app.route('/big_screen/<int:vid_id>')
def big_screen(vid_id):
    return render_template('big_screen.html', vid = 'video_'+str(vid_id))

@app.route("/logout")
def logout():
    session.pop('loggedin',None)
    session.pop('username', None)
    # On logout clear the session and redirect back to login
    return redirect(url_for('index'))



if __name__ == "__main__":
    v_1 = Window(video_loc=input_video_1, config_path=os.path.join('configs','people_crossing_v_1.yaml'))
    v_2 = Window(video_loc=input_video_2, config_path=os.path.join('configs','old_lady_v_2.yaml'))
    v_3 = Window(video_loc=input_video_3, config_path=os.path.join('configs','empty_crossing_v_3.yaml'))
    v_4 = Window(video_loc=input_video_4, config_path=os.path.join('configs','walking_on_sides_v_4.yaml'))
    v_5 = Window(video_loc=input_video_5, config_path=os.path.join('configs','people_crossing_v_1.yaml'))
    v_6 = Window(video_loc=input_video_6, config_path=os.path.join('configs','old_lady_v_2.yaml'))
    app.run(debug=True)