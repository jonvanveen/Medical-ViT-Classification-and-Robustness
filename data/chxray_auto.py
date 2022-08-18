import os
import numpy as np
import csv
import shutil
import random

# Load data
with open('autoattack_labels.csv', mode='r') as csv_file:
    data_entries = csv.reader(csv_file,delimiter=',')
    rows = list(data_entries)

# Take 1000 images at random
random.shuffle(rows)
thous = rows[0:1000]

with open('chxray_auto_train.csv',mode='w') as autoattack_1000:
    csv_write=csv.writer(autoattack_1000, delimiter=',')
    with open('chxray_auto_test.csv',mode='w') as test:
        csv_writer=csv.writer(test, delimiter=',')
        
        cnt=1
        for row in thous:
            if(cnt < 801):
                shutil.copy('raw_imgs/'+row[0], 'chxray_auto_1000/train')
                csv_write.writerow([row[0],row[1]])
                cnt+=1
            elif(801<=cnt<len(thous)+1):
                shutil.copy('raw_imgs/'+row[0], 'chxray_auto_1000/test')
                csv_writer.writerow([row[0],row[1]])
                cnt+=1
