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

# Load data
with open('data_entries.csv', mode='r') as csv_file:
    data_entries = csv.reader(csv_file,delimiter=',')
    rows = list(data_entries)
    train_len = np.ceil(0.8*len(rows))

raw_imgs = os.listdir('raw_imgs')
# 15 classes
label_list = ['Atelectasis','Cardiomegaly','Effusion','Infiltration','Mass','Nodule','Pneumonia','Pneumothorax','Consolidation','Edema','Emphysema','Fibrosis','Pleural_Thickening','Hernia','No Finding']

file_count=0
# Map string category labels to ints in new csv file for swin_autoattack.py req's
with open('autoattack_labels.csv',mode='w') as autoattack_csv:
    csv_write = csv.writer(autoattack_csv, delimiter=',')

    for row in rows: 
        labelnum = 1
        for label in label_list: # Multi-label images are sorted into multiple individual classes

            if file_count<train_len:#89698: # training data
                class_path = 'imagenet/train/class'+str(labelnum)
                if label in row[1]:
                    shutil.copy('raw_imgs/'+row[0], class_path)
                    csv_write.writerow([ row[0], label_list.index(label) ])
                    print("Sorted file #",file_count,", ",row[0]," into ",class_path)
            
            else: # testing data
                class_path = 'imagenet/val/class'+str(labelnum)
                if label in row[1]:
                    shutil.copy('raw_imgs/'+row[0], class_path)
                    print("Sorted file #",file_count,", ",row[0], " into ",class_path)
                    csv_write.writerow([ row[0], label_list.index(label) ])

            labelnum+=1
        file_count+=1



