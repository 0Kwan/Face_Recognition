from util.pre_process import aligned_face, face_image
from util.test_url_img import URL_fetchImg
import torch
import torch.nn as nn
import PIL
import os
import os.path as osp
import numpy as np
import mediapipe as mp
from PIL import Image
from util.config import config as conf
from model import FaceMobileNet
import PIL.Image
import time
import matplotlib.pyplot as plt
import cv2
def preprocess(img, transform) -> torch.Tensor:
    im = PIL.Image.fromarray(img)
    im = transform(im)
    data = torch.unsqueeze(im, dim=0)   # shape: (1, 128, 128)
    return data


def featurize(image, transform, net, device) -> torch.Tensor:

    data = preprocess(image,transform)
    data = data.to(device) # add batch dimension
    net = net.to(device)
    with torch.no_grad():
        features = net(data) 
    return features.squeeze(0)

def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def threshold_search(y_score):
    y_score = np.asarray(y_score)
    th = 0.941
    if y_score >= th:
      y_test = '인증'
    else :
      y_test = '미인증'
    return y_test

def gender_detect(img_url):
    gender_list = ['Male', 'Female']
    gender_net = cv2.dnn.readNetFromCaffe(
        'checkpoints/deploy_gender.prototxt', 
        'checkpoints/gender_net.caffemodel')
    img = URL_fetchImg(img_url)
    face_img = face_image(img)
    blob = cv2.dnn.blobFromImage(face_img, scalefactor=1, size=(227, 227),
                                 mean=(78.4263377603, 87.7689143744, 114.895847746),
                                 swapRB=False, crop=False)
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_list[gender_preds[0].argmax()]
    return gender


def face_service(auth_url, profile_url):
    auth_img1 = URL_fetchImg(auth_url)
    profile_img1 = URL_fetchImg(profile_url)
    auth_img = face_image(auth_img1)
    if auth_img is None:
        return print(f"No face detected in {auth_img}, auth.")
    profile_img = face_image(profile_img1)
    if profile_img is None:
        return print(f"No face detected in {profile_img}, profile.")
    auth_img = aligned_face(auth_img)
    if auth_img is None:
        return print(f"No face detected in {auth_img}, auth.")
    profile_img = aligned_face(profile_img)
    if profile_img is None:
        return print(f"No face detected in {profile_img}, profile.")
    model = FaceMobileNet(conf.embedding_size)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(conf.test_model, map_location=conf.device))
    model.eval()
    profile_features = featurize(profile_img, conf.test_transform, model, conf.device).cpu().numpy()
    auth_features = featurize(auth_img, conf.test_transform, model, conf.device).cpu().numpy()
    similarity = cosin_metric(profile_features, auth_features)
    result = threshold_search(similarity)
    return result, auth_img1, profile_img1, similarity

start_time = time.time()
profile_url = 'https://sinor.s3.ap-northeast-2.amazonaws.com/userPhoto/1669949261758.jpg' # 배경사진
auth_url = 'https://sinor.s3.ap-northeast-2.amazonaws.com/userAuthPhoto/auth_5361669949120230.jpg'  # 얼굴
result, auth_img1, profile_img1, similarity= face_service(auth_url , profile_url)
print(f'service time : {round(time.time() - start_time, 3)}')
print(result,similarity)
auth_gender = gender_detect(auth_url)
profile_gender = gender_detect(profile_url)
print(f'auth_gender : {auth_gender}, profile_gender : {profile_gender}')
male_gender_url = 'https://sinor.s3.ap-northeast-2.amazonaws.com/userAuthPhoto/auth_1261666411883874.jpg'
male_gender = gender_detect(male_gender_url)
print(male_gender)
male_img = URL_fetchImg(male_gender_url)
auth_img1 = cv2.cvtColor(auth_img1,cv2.COLOR_BGR2RGB)
profile_img1 = cv2.cvtColor(profile_img1,cv2.COLOR_BGR2RGB)
male_img = cv2.cvtColor(male_img,cv2.COLOR_BGR2RGB)
plt.subplot(2, 2, 1)
plt.imshow(auth_img1)
plt.subplot(2, 2, 2)
plt.imshow(profile_img1)
plt.subplot(2, 2, 3)
plt.imshow(male_img)
plt.show()
# cv2.namedWindow('Images', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Images', 300, 200) # Optional: resize the window
# cv2.moveWindow('Images', 0, 0) # Optional: move the window to a specific position
# cv2.imshow('Images', cv2.vconcat([cv2.hconcat([auth_img1, profile_img1])]))
# cv2.waitKey()
# cv2.destoryAllWindows()