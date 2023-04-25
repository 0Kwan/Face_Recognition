import os
import pandas as pd
import requests
import urllib
from urllib.request import Request, urlopen
import numpy as np
import cv2



def URL_fetchImg(URL) : # url로 존재하는 파일들을 이미지파일로 변환 
    """
    함수 설명 : url로부터 이미지를 불러와 변환
    인풋 : url 링크(string)
    아웃풋 : 이미지(np.array)
    """
    try :
        RESPONSE = requests.get(URL)
        data = urllib.request.urlopen(URL).read() #type:bytes
        encoded_img = np.frombuffer(data, dtype = np.uint8) # bytes -> unit8
        image = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR) # 1dim -> 3dim(BRG)
        return image #narray
    except urllib.error.HTTPError:
        print(f"HTTP error url : {URL}")
        return ''
    except requests.exceptions.MissingSchema:
        print(f'missing schema url : {URL}')


# 엑셀에 url로 존재하는 파일들을 이미지형태로 저장시켜주는 코드 

# df = pd.read_excel('./data/AI_DATA_0403_1.xlsx',sheet_name = 'Sheet1', header = 1)

# for i, r in df['profiles'].items():
#     result = r.split(',')
#     df.loc[i, 'profile1'] = result[0]
#     if len(result) == 2:
#         df.loc[i, 'profile2'] = result[1]
#     if len(result) == 3:
#         df.loc[i, 'profile3'] = result[2]
#     if len(result) == 4:
#         df.loc[i, 'profile4'] = result[3]


# if not os.path.exists('images'):
#     os.makedirs('images')
# for idx, row in df.iterrows():
#     folder_name = str(row['id'])
#     if not os.path.exists(f'images/{folder_name}'):
#         os.makedirs(f'images/{folder_name}')

#     # auth 이미지 저장
#     auth_img = URL_fetchImg(row['auth'])
#     if isinstance(auth_img, np.ndarray):
#         cv2.imwrite(f"images/{folder_name}/{row['id']}_auth.jpg", auth_img)
        
#     # profile 이미지 저장
#     for i in range(1,5):
#         if f"profile{i}" in row:
#             profile_img = URL_fetchImg(row[f"profile{i}"])
#             if isinstance(profile_img, np.ndarray):
#                 cv2.imwrite(f"images/{folder_name}/{row['id']}_{i}.jpg", profile_img)


