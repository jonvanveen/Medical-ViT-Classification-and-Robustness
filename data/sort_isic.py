""" 
This code organizes the raw NIH Chest X-ray dataset into two folders (train/test), and
15 subfolders (for each diagnosis class in the dataset). This file setup is required for
using the Swin Transformer, one of the models I'm investigating. This dataset contains
~112,000 frontal-view X-ray images of the chest in .png format. 
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
# For ISIC dataset, only using classes with 250+ samples, for 10 total classes
label_list = ['nevus','melanoma','basal cell carcinoma','seborrheic keratosis','pigmented benign keratosis','actinic keratosis','squamous cell carcinoma','solar lentigo','vascular lesion','dermatofibroma']

file_count=0
for row in rows: 
    labelnum = 1
    for label in label_list: # Multi-label images are sorted into multiple individual classes
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



