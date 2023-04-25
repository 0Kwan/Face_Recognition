import os
#트레인데이터 폴더 분류 작업중 비어있는 폴더 제거 
for root, dirs, files in os.walk('./data/last_train_img'):
    for dir in dirs:
        dir_path = os.path.join(root, dir)
        if not os.listdir(dir_path):
            os.rmdir(dir_path)