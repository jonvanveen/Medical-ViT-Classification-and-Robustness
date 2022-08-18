""" 
This code organizes the raw ISIC dataset into the Imagenet directory structure, 
with two folders (train/test), and 15 subfolders (for each diagnosis class in the dataset). 
This file setup was required for using the Swin-T and the Resnet50. This dataset contains
~71,000 images of benign and malignant skin lesions in .png format. 
"""

import os
import numpy as np
import csv
import shutil
import random

# Load data
with open('metadata.csv', mode='r') as csv_file:
    data_entries = csv.reader(csv_file,delimiter=',')
    rows = list(data_entries)
    random.shuffle(rows)
    train_len = np.ceil(0.8*len(rows))

raw_imgs = os.listdir('isic_raw')
# For ISIC dataset, only use the top 10 classes, for a more balanced
# dataset relative to the NIH Chest X-ray
label_list = ['nevus','melanoma','basal cell carcinoma','seborrheic keratosis','pigmented benign keratosis','actinic keratosis','squamous cell carcinoma','solar lentigo','vascular lesion','dermatofibroma']

file_count=0
for row in rows: 
    labelnum = 1
    for label in label_list: 
        # 80-20 train-test split

        if file_count<train_len: # training data
            class_path = 'imagenet/train/class'+str(labelnum)
            if label in row[9]:
                shutil.copy('isic_raw/'+row[0]+'.JPG', class_path)
                print("Sorted file #",file_count,", ",row[0]," into ",class_path)
        else: # testing data
            class_path = 'imagenet/val/class'+str(labelnum)
            if label in row[9]:
                shutil.copy('isic_raw/'+row[0]+'.JPG', class_path)
                print("Sorted file #",file_count,", ",row[0], " into ",class_path)
        
        labelnum+=1
    file_count+=1



