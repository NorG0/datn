#!/usr/bin/env python
# coding: utf-8

# In[2]:


import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import time
import os
import datetime
from threading import Thread
from queue import Queue


# In[3]:


mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions


# In[4]:


import csv
import os
import numpy as np
import pickle


# In[5]:


import pandas as pd
from sklearn.model_selection import train_test_split


# In[6]:


with open('body_language_lr.pkl', 'rb') as f:
    model = pickle.load(f)


# In[7]:


model


# In[8]:


# user = open('user.txt',"r")
user= 'long'
# user

# with open ('user.csv', mode = 'a', newline='') as f:
#                         csv_writer=csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#                         csv_writer.writerow(body_language_class_list)


# In[9]:


class Camera:
    def __init__(self, mirror=False):
        self.data = None
        self.cam = cv2.VideoCapture(0)

        self.WIDTH = 640
        self.HEIGHT = 480

        self.center_x = self.WIDTH / 2
        self.center_y = self.HEIGHT / 2
        self.touched_zoom = False
        self.image_queue = Queue()
        self.video_queue = Queue()

        self.scale = 1
        self.__setup()

        self.mirror = mirror

    def __setup(self):
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.WIDTH)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.HEIGHT)
        time.sleep(2)

    def get_location(self, x, y):
        self.center_x = x
        self.center_y = y
        self.touched_zoom = True

    def stream(self):
        # streaming thread 함수
        def streaming():
            # 실제 thread 되는 함수
            self.ret = True
            while self.ret:
                self.ret, np_image = self.cam.read()
                if np_image is None:
                    continue
                if self.mirror:
                    # 거울 모드 시 좌우 반전
                    np_image = cv2.flip(np_image, 1)
                if self.touched_zoom:
                    np_image = self.__zoom(np_image, (self.center_x, self.center_y))
                else:
                    if not self.scale == 1:
                        np_image = self.__zoom(np_image)
                self.data = np_image
                k = cv2.waitKey(1)
                if k == ord('q'):
                    self.release()
                    break

        Thread(target=streaming).start()

    def __zoom(self, img, center=None):
        # zoom하는 실제 함수
        height, width = img.shape[:2]
        if center is None:
            #   중심값이 초기값일 때의 계산
            center_x = int(width / 2)
            center_y = int(height / 2)
            radius_x, radius_y = int(width / 2), int(height / 2)
        else:
            #   특정 위치 지정시 계산
            rate = height / width
            center_x, center_y = center

            #   비율 범위에 맞게 중심값 계산
            if center_x < width * (1-rate):
                center_x = width * (1-rate)
            elif center_x > width * rate:
                center_x = width * rate
            if center_y < height * (1-rate):
                center_y = height * (1-rate)
            elif center_y > height * rate:
                center_y = height * rate

            center_x, center_y = int(center_x), int(center_y)
            left_x, right_x = center_x, int(width - center_x)
            up_y, down_y = int(height - center_y), center_y
            radius_x = min(left_x, right_x)
            radius_y = min(up_y, down_y)

        # 실제 zoom 코드
        radius_x, radius_y = int(self.scale * radius_x), int(self.scale * radius_y)

        # size 계산
        min_x, max_x = center_x - radius_x, center_x + radius_x
        min_y, max_y = center_y - radius_y, center_y + radius_y

        # size에 맞춰 이미지를 자른다
        cropped = img[min_y:max_y, min_x:max_x]
        # 원래 사이즈로 늘려서 리턴
        new_cropped = cv2.resize(cropped, (width, height))

        return new_cropped

    def zoom_out(self):
        # scale 값을 조정하여 zoom-out
        if self.scale < 1:
            self.scale += 0.4
        if self.scale == 1:
            self.center_x = self.WIDTH
            self.center_y = self.HEIGHT
            self.touched_zoom = False

    def zoom_in(self):
        # scale 값을 조정하여 zoom-in
        if self.scale > 0.2:
            self.scale -= 0.4
    def save_picture(self):
        # 이미지 저장하는 함수
        ret, img = self.cam.read()
        if ret:
            now = datetime.datetime.now()
            date = now.strftime('%Y%m%d')
            hour = now.strftime('%H%M%S')
            user_id = '00001'
            filename = './images/cvui_{}_{}_{}.png'.format(date, hour, user_id)
            cv2.imwrite(filename, img)
            self.image_queue.put_nowait(filename)

    def show(self):
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while self.cam.isOpened():
                self.ret,frame = self.cam.read()
                frame = self.data
                
                # Recolor Feed
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False    

                # Make Detections
                results = holistic.process(image)
                image.flags.writeable = True   
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # 1. Draw face landmarks
                mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                         mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                         mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                         )

                # 2. Right hand
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                         mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                         mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                         )

                # 3. Left Hand
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                         mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                         mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                         )

                # 4. Pose Detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                         mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                         mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                         )
                # Export coordinates
                try:
                    # Extract Pose landmarks
                    pose = results.pose_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                    # Extract Face landmarks
                    face = results.face_landmarks.landmark
                    face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

                    # Concate rows
                    row = pose_row+face_row


                    # Make Detections
                    X = pd.DataFrame([row])
                    body_language_class = model.predict(X)[0]
                    body_language_prob = model.predict_proba(X)[0]
                    body_language_class_list = body_language_class.split()
                        
                    print(body_language_class, body_language_prob)

                    # Grab ear coords
                    coords = tuple(np.multiply(
                                    np.array(
                                        (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                         results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                                , [640,480]).astype(int))

                    cv2.rectangle(image, 
                                  (coords[0], coords[1]+5), 
                                  (coords[0]+len(body_language_class)*20, coords[1]-30), 
                                  (245, 117, 16), -1)
                    cv2.putText(image, body_language_class, coords, 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # Get status box
                    cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)

                    # Display Class
                    cv2.putText(image, 'CLASS'
                                , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, body_language_class.split(' ')[0]
                                , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # Display Probability
                    cv2.putText(image, 'PROB'
                                , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                                , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    # Save Results to CSV
                    body_language_class_list.insert(0,user)
                    with open ('rs.csv', mode = 'a', newline='') as f:
                        csv_writer=csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(body_language_class_list)

                except:
                    pass
                
                if frame is not None:
                    cv2.imshow('SMS', image)
                
                key = cv2.waitKey(1)
                if key == ord('q'):
                    # q : close
                    self.release()
                    cv2.destroyAllWindows()
                    break

                elif key == ord('z'):
                    # z : zoom - in
                    self.zoom_in()

                elif key == ord('x'):
                    # x : zoom - out
                    self.zoom_out()
                
                
                    
    def release(self):
        self.cam.release()
        cv2.destroyAllWindows()


# In[ ]:





# In[10]:


cam=Camera(mirror=True)


# In[11]:


cam.stream()


# In[12]:


cam.show()


# DEUBUGGG



