"""
Code to sort two classes for NIH Chest x-ray images for binary classification.
I coded this script as a simpler classification problem as a backup in case
the multi-class NIH dataset was not suitable for Swin-T, but this sorting
ultimately went unused.
"""

import os
import numpy as np
import csv
import shutil

with open('data_entries.csv', mode='r') as csv_file:
    data_entries = csv.reader(csv_file,delimiter=',')
    rows = list(data_entries)
    split = len(rows)*0.8

raw_imgs = os.listdir('raw_imgs')
label_list = ['Pneumonia','No Finding']

file_count=0
for row in rows: # training data
    labelnum = 1
    for label in label_list:
        
        # 80-20 split
        if file_count<split:
            class_path = 'imagenet/train/class'+str(labelnum)
            if label in row[1]:
                shutil.copy('raw_imgs/'+row[0], class_path)
                print("Sorted file #",file_count,", ",row[0]," into ",class_path)
        else:
            class_path = 'imagenet/val/class'+str(labelnum)
            if label in row[1]:
                shutil.copy('raw_imgs/'+row[0], class_path)
                print("Sorted file #",file_count,", ",row[0], " into ",class_path)
        labelnum+=1
    file_count+=1
