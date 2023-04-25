import cv2
import os
import mediapipe as mp
import math
import numpy as np

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.3)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

def aligned_face(image): # 얼굴인식 후 양쪽눈을 기준으로 수평을 맞혀 사진 추출
  detection_results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
  if detection_results.detections:
    landmarks_results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if landmarks_results.multi_face_landmarks:
      landmarks = landmarks_results.multi_face_landmarks[0].landmark
      landmarks = np.array([[lmk.x * image.shape[1], lmk.y * image.shape[0]] for lmk in landmarks], dtype=np.int32)
      left_eye = landmarks[33]
      right_eye = landmarks[263]
      angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
      rotation_matrix = cv2.getRotationMatrix2D(((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2), angle, 1)
      aligned_face = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
      return aligned_face


def face_image(image): #얼굴인식 후 얼굴사진만 추출
  detection_results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
  if detection_results.detections:
            # 얼굴 랜드마크 수행
    landmarks_results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if landmarks_results.multi_face_landmarks:
                # 첫 번째 얼굴에 대해서만 처리
      landmarks = landmarks_results.multi_face_landmarks[0].landmark
      landmarks = np.array([[lmk.x * image.shape[1], lmk.y * image.shape[0]] for lmk in landmarks], dtype=np.int32)

                # 얼굴 좌표 계산
      x1 = int(np.min(landmarks[:, 0]))
      x2 = int(np.max(landmarks[:, 0]))
      y1 = int(np.min(landmarks[:, 1]))
      y2 = int(np.max(landmarks[:, 1]))
                
                # 얼굴 이미지 추출
      face_image = image[y1:y2, x1:x2]
      return face_image
    
# MEDIAPIPE 얼굴검출 확인

# import os
# import pandas as pd
# import requests
# import urllib
# from urllib.request import Request, urlopen
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# def URL_fetchImg(URL) : # url로 존재하는 파일들을 이미지파일로 변환 
#     """
#     함수 설명 : url로부터 이미지를 불러와 변환
#     인풋 : url 링크(string)
#     아웃풋 : 이미지(np.array)
#     """
#     try :
#         RESPONSE = requests.get(URL)
#         data = urllib.request.urlopen(URL).read() #type:bytes
#         encoded_img = np.frombuffer(data, dtype = np.uint8) # bytes -> unit8
#         image = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR) # 1dim -> 3dim(BRG)
#         return image #narray
#     except urllib.error.HTTPError:
#         print(f"HTTP error url : {URL}")
#         return ''
#     except requests.exceptions.MissingSchema:
#         print(f'missing schema url : {URL}')

        
# profile_url = 'https://sinor.s3.ap-northeast-2.amazonaws.com/userPhoto/1666484950598.jpg' # 배경사진
# profile_url = URL_fetchImg(profile_url)
# profile_img = face_image(profile_url)
# if profile_img is None:
#    print(f"No face detected in {profile_img}, skipping.")
# else :
#     profile_img = cv2.cvtColor(profile_img,cv2.COLOR_BGR2RGB)
#     plt.subplot(2, 2, 1)
#     plt.imshow(profile_img)
#     plt.show()