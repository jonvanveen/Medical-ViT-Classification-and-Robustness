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

with open('autoattack_1000.csv',mode='w') as autoattack_1000:
    csv_write=csv.writer(autoattack_1000, delimiter=',')

    for row in thous:
        shutil.copy('raw_imgs/'+row[0], 'auto_1000')
        csv_write.writerow([row[0],row[1]])
