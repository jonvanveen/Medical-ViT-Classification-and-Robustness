"""
This code was my attempted implementation at testing the adversarial robustness of Swin-T.

Due to time constraints and lack of Pytorch experience, I was unable to implement true
adversarial robustness on Swin-T in this project. See the project report for more details.
This code does not run, but is included in the repo to document what I attempted to do. 

The script adapts a combination of code from the following sources:

Pytorch Dataloader Tutorial https://pytorch.org/tutorials/beginner/data_loading_tutorial.html 
License: BSD
Author: Sasank Chilamkurthy

Microsoft Swin Transformer: https://github.com/microsoft/Swin-Transformer
Swin Transformer
Copyright (c) 2021 Microsoft
Licensed under The MIT License [see LICENSE for details]
Written by Ze Liu
"""

import os
import io
import argparse
import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import numpy as np
from models import build_model
from config import get_config
from skimage import io, transform
import gc
import torch.optim as optim

import sys
sys.path.insert(0,'..')

# Load custom dataset with class
# Dataset loader code adapted from Pytorch Tutorials: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html 
# License: BSD
# Author: Sasank Chilamkurthy
class LoadDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float32')

        if self.transform:
            image = self.transform(image)
            landmarks = self.transform(landmarks)

        sample = {'image': image, 'landmarks': landmarks}
        del image
        del landmarks
        gc.collect()
        torch.cuda.empty_cache()

        return sample


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, metavar="FILE")
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--csv', type=str)
    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--epsilon', type=float, default=8./255.)
    parser.add_argument('--model', type=str, default='./model_test.pt')
    parser.add_argument('--n_ex', type=int, default=1000)
    parser.add_argument('--individual', action='store_true')
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--log_path', type=str, default='./log_file.txt')
    parser.add_argument('--version', type=str, default='standard')
    parser.add_argument("--local_rank", type=int, default=0)
    # pass train/test csvs
    parser.add_argument('--csv_tr', type=str)
    parser.add_argument('--csv_ts', type=str)
    
    # optimize GPU usage
    gc.collect()
    torch.cuda.empty_cache()

    args, unparsed = parser.parse_known_args()
    print(args)
    config = get_config(args)

    # load model
    model = build_model(config)
    ckpt = torch.load(args.model)
    model.load_state_dict(ckpt,False) 
    
    model.cuda()
    model.eval()
    
    # load data
    transform_list = [transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor()]
    transform_chain = transforms.Compose(transform_list)
    item_tr = LoadDataset(csv_file=args.csv_tr, root_dir=args.data_dir+"/train", transform=transform_chain)
    item_ts = LoadDataset(csv_file=args.csv_ts, root_dir=args.data_dir+"/test", transform=transform_chain)
    train_loader = DataLoader(item_tr, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(item_ts, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # create save dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    print(x for (x,y) in test_loader)
    print(type(x) for (x,y) in test_loader)

    from art.attacks.evasion import AutoAttack
    from art.estimators.classification import PyTorchClassifier
    
    classifier = PyTorchClassifier(
        model=model,
        clip_values=None,
        loss=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(model.parameters(), lr=0.01),
        input_shape=(3, 224, 224),
        nb_classes=1000,
    )
    print("classifier init: ",classifier.nb_classes)
    print("Preparing data...")
    lxtr, lytr = [],[]
    for i, batch in enumerate(train_loader):
        x, y = batch["image"], batch["landmarks"]
        lxtr.append(x)
        lytr.append(y)
    print(lxtr)
    print(lytr)
        
    lxts, lyts = [],[]
    for i, batch in enumerate(test_loader):
        x, y = batch["image"], batch["landmarks"]
        lxts.append(x)
        lyts.append(y)
        
    x_train, y_train = torch.cat(lxtr, 0), torch.cat(lytr, 0) 
    print(torch.flatten(x_train).size())
    x_test, y_test = torch.cat(lxts, 0), torch.cat(lyts, 0)
    print("Data prepared.")
    
    # Fixing incorrect dimensions of tensor for input into ART classifier
    y_train = y_train[:799,:,0,0]
    x_ = torch.zeros(799,10)
    for i in range(0,799):
        x_[i][int(y_train[i])-1] = 1
    y_train = x_
    
    y_test = y_test[:220,:,0,0]
    x_ = torch.zeros(220,10)
    for i in range(0,220):
        x_[i][int(y_test[i])-1] = 1
    y_test = x_
    
    print("Fitting classifier...")
    classifier.fit(x_train.detach().cpu().numpy(), y_train.detach().cpu().numpy(), batch_size=64, training_mode=True, nb_epochs=5)
    x_train = x_train.detach().cpu().numpy()
    y_train = y_train.detach().cpu().numpy()
    
    logits = classifier.predict(x)
    
    predictions = classifier.predict(x_test.detach().cpu().numpy(),training_mode=False)
    print("classifier pred: ",classifier.nb_classes)
    print('predictions: ',predictions) 
    print('shape: ',predictions.shape)
    accuracy = np.sum(np.argmax(predictions[:10,:], axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("Accuracy on benign test examples: {}%".format(accuracy * 100))
    
    # Generate adversarial test examples
    attack = AutoAttack(estimator=classifier, eps=0.2)
    print('x test: ',type(x_test), x_test)
    x_test_adv = attack.generate(x=x_test)
    
    # Evaluate the ART classifier on adversarial test examples
    predictions = classifier.predict(x_test_adv)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
    
