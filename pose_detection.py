#!/usr/bin/env python
# coding: utf-8

# In[32]:


import cv2
import time
import numpy as np 
#import Holistic
import mediapipe as mp
# window에서만 지원 
#from win10toast import ToastNotifier
import math
import matplotlib.pyplot as plt 


# In[3]:


# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_pose = mp.solutions.pose

# #파일 위치 미리 지정
# input_video_path = "./data/ex3_input.mp4"
# default_video_path = "./data/ex3_default.mp4"
# save_video_path = './output/ex3_output.mp4'

# cap = cv2.VideoCapture(input_video_path)
# # webcam으로 할 경우엔 input_video_path 대신 0을 넣어줄 것
# # cap = cv2.VideoCapture(0)

# # holistic = Holistic.HolisticDetector()
# mp_holistic = mp.solutions.holistic


# In[51]:


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

#파일 위치 미리 지정
input_video_path = "./data/ex4_input.mp4"
#default_video_path = "./data/ex3_default.mp4"
save_video_path = './output/ex4_holistic_output.mp4'
#save_default_video_path = './output/ex3_holistic_default_output.mp4'

cap = cv2.VideoCapture(input_video_path)
#cap_d = cv2.VideoCapture(default_video_path)
# webcam으로 할 경우엔 input_video_path 대신 0을 넣어줄 것
# cap = cv2.VideoCapture(0)
#holistic = Holistic.HolisticDetector()

#재생할 파일의 넓이와 높이
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#video controller
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter(save_video_path, fourcc, 30.0, (int(width), int(height)))
#out_d = cv2.VideoWriter(save_default_video_path, fourcc, 30.0, (int(width), int(height)))  ####

######
# records_d = {}
# records_d['fn'], records_d['us'], records_d['rs'] = [], [], []

# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#     while cap_d.isOpened():
#         success, image = cap_d.read()
#         #print(success, image)
#         # Webcam 사용 시 이 코드 살리기 
#         # if not success:
#         #     print("카메라를 찾을 수 없습니다.")
#         #     # 웹캠을 불러올 경우는 'continue', 동영상을 불러올 경우 'break'를 사용합니다.
#         if success:  
#             image.flags.writeable = False
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             results_d = holistic.process(image)
    
#             # 포즈 주석을 이미지 위에 그립니다.
#             image.flags.writeable = True
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#             #image = cv2.blur(image, (75, 75))
#             mp_drawing.draw_landmarks(image, results_d.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
#             mp_drawing.draw_landmarks(image, results_d.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
#                                       landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
#             face_lm, pose_lm = save_landmarks(results_d, image, min_lm=True)
#             fn, sh_angle, rs = summary_records(face_lm, pose_lm)
            
#             records_d['fn'].append(fn)
#             records_d['us'].append(sh_angle)
#             records_d['rs'].append(rs)
            
#             out.write(image)
#         else: 
#             break

# cap_d.release()
# out_d.release()
########

records = {}
records['fn'], records['us'], records['rs'] = [], [], []

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        success, image = cap.read()
        #print(success, image)
        # Webcam 사용 시 이 코드 살리기 
        # if not success:
        #     print("카메라를 찾을 수 없습니다.")
        #     # 웹캠을 불러올 경우는 'continue', 동영상을 불러올 경우 'break'를 사용합니다.
        if success:  
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
    
            # 포즈 주석을 이미지 위에 그립니다.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            #image = cv2.blur(image, (75, 75))
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
            face_lm, pose_lm = save_landmarks(results, image, min_lm=True)
            fn, sh_angle, rs = summary_records(face_lm, pose_lm)
            
            records['fn'].append(fn)
            records['us'].append(sh_angle)
            records['rs'].append(rs)
            
            out.write(image)
        else: 
            break

cap.release()
out.release()


# In[5]:


def save_landmarks(results, image, min_lm=False):
    lms = [results.face_landmarks.landmark, results.pose_landmarks.landmark]
    face_lm, pose_lm = {}, {} 
    lm_list = [face_lm, pose_lm]
    h, w, c, = image.shape # height, width, channel
    
    for num in range(len(lms)):
        lm = lms[num]
        if min_lm == True:
            # Save only essential 5 landmarks for reducing required time
            # Essential landmarks: Face = 152, 133, 362 / Pose = 12, 11
            cds = lm
            if num == 0: # Face landmarks
                lm_list[num][152] = (cds[152].x, cds[152].y, cds[152].z)
                lm_list[num][133], lm_list[num][362] = (cds[133].x, cds[133].y, cds[133].z), (cds[362].x, cds[362].y, cds[362].z)
            else: # Pose landmarks
                lm_list[num][11], lm_list[num][12] = (cds[11].x, cds[11].y, cds[11].z), (cds[12].x, cds[12].y, cds[12].z)
        else:
            # Save all the landmarks
            for i, cd in enumerate(lm):
                cx, cy, cz = int(cd.x*w), int(cd.y*h), int(cd.z*(w+h)/2)
                cxyz = (cx, cy, cz)
                lm_list[num][i] = cxyz
    return face_lm, pose_lm


# In[25]:


def euclidean_dist(p1, p2): 
    dx = (p1[0] - p2[0]) ** 2
    dy = (p1[1] - p2[1]) ** 2
    eu_dist = (dx + dy) ** 0.5
    return eu_dist

def summary_records(face_lm, pose_lm):
    # Distace between two eyes
    a = euclidean_dist(face_lm[133], face_lm[362])
    
    # Length of neck     
    mid_sh = ((pose_lm[11][0]+pose_lm[12][0])/2, (pose_lm[11][1]+pose_lm[12][1])/2)
    b = euclidean_dist(face_lm[152], mid_sh)
    
    # Shoulder angle 
    c = euclidean_dist(pose_lm[11], pose_lm[12])
    if pose_lm[11][1] > pose_lm[12][1]:
        sh_flag = 'Right' # 오른쪽 어깨가 아래 
        c_x = euclidean_dist(pose_lm[12], (pose_lm[11][0], pose_lm[12][1]))
        c_y = euclidean_dist(pose_lm[11], (pose_lm[11][0], pose_lm[12][1]))
    
    else: # pose_lm[11][1] < pose_lm[12][1]
        sh_flag = 'Left'  # 왼쪽 어깨가 아래 
        c_x = euclidean_dist(pose_lm[11], (pose_lm[12][0], pose_lm[11][1]))
        c_y = euclidean_dist(pose_lm[12], (pose_lm[12][0], pose_lm[11][1])) 
    
    sh_angle = np.degrees(np.arccos(c_x/c))
    fn = b / a
    rs =  c / a
    return fn, sh_angle, rs


# In[ ]:




