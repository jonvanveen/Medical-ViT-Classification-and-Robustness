
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

label_list = ['nevus','melanoma','basal cell carcinoma','seborrheic keratosis','pigmented benign keratosis','actinic keratosis','squamous cell carcinoma','solar lentigo','vascular lesion','dermatofibroma']
thous = rows[0:1700]

with open('isic_auto_train.csv', mode='w') as isic_auto_train:
    with open('isic_auto_test.csv', mode='w') as isic_auto_test:
        cnt=1
        for row in thous:
            if(cnt < 801):
                csv_write=csv.writer(isic_auto_train,delimiter=',')
                if (row[9] in label_list):
                    shutil.copy('isic_raw/'+row[0]+'.JPG', 'isic_auto_1000/train')
                    print(row[9])
                    row1 = label_list.index(row[9]) + 1
                    csv_write.writerow([str(row[0])+'.JPG', row1])
                    print(cnt)
                    cnt+=1
            elif(801 <= cnt < len(thous)+1):
                csv_writer=csv.writer(isic_auto_test,delimiter=',')
                if (row[9] in label_list):
                    shutil.copy('isic_raw/'+row[0]+'.JPG', 'isic_auto_1000/test')
                    print(row[9])
                    row1 = label_list.index(row[9]) + 1
                    csv_writer.writerow([str(row[0])+'.JPG', row1])
                    print(cnt)
                    cnt+=1


