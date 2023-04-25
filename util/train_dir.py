import os
import pandas as pd
import glob

#train할때 필요한 파일 형태로 변환시켜주는 코드

folder_path = './data/train_img'

# read in the list of file names
image_paths = glob.glob(os.path.join(folder_path, '*.*'))
image_paths = [os.path.basename(file_path) for file_path in image_paths]
image_paths.sort()
print(image_paths)

current_folder_name = ''
folder_counter = 0

for file_name in image_paths:
    
    prefix = file_name[:12]

    if prefix != current_folder_name:
        folder_counter += 1
        folder_name = str(folder_counter)
        os.mkdir(os.path.join(folder_path, folder_name))
        current_folder_name = prefix
    os.rename(os.path.join(folder_path, file_name),
              os.path.join(folder_path, folder_name, file_name))