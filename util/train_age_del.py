import os
import pandas as pd
import glob

#트레인데이터 저장중 발생한 필요없는 파일 제거

folder_path = './data/train_img'

# read in the list of file names
image_paths = glob.glob(os.path.join(folder_path, '*.*'))
image_paths.sort()
# create a new list to store the filtered image paths

for path in image_paths:
    # extract the filename from the path
    filename = os.path.basename(path)
    
    # check if the filename contains "AGE"
    if "AGE" in filename[6:10]:
        # if not, add the path to the filtered list
        os.remove(path)
