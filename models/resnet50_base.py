# Code adapted with modifications from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html 
# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import sklearn
from timm.utils import accuracy, AverageMeter

cudnn.benchmark = True
plt.ion()   # interactive mode

# Try to speed up GPU w/ clearing memory
# torch.cuda.empty_cache()
# import gc
# gc.collect()

# Transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()#,
    ]),
    'val': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()#,
    ]),
}

# Train on NIH chest x-ray Dataset
data_dir = '~/Data/chxray/imagenet'
# Train on ISIC Dataset
data_dir = '~/Data/isic/isic_org'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                  data_transforms[x])
                  for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16, # batch size so small due to memory constraints
              shuffle=True, num_workers=2)
              for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# AUC calculation
from sklearn.metrics import roc_auc_score
def cal_auc(outputs, targets):
    outputs = outputs.squeeze()
    targets = targets.squeeze()

    if outputs.ndim == 1:
        # try-except block to avoid incorrect ValueError
        try:
            auc = roc_auc_score(targets, outputs)
        except ValueError:
            pass
    else:
        n_classes = outputs.shape[1]
        auc = 0
        for i in range(n_classes):
            try:
                label_auc = roc_auc_score(targets==i, outputs[:,i])
            except ValueError:
                pass
            auc += label_auc
        auc /= n_classes
    return auc

def train_model(model, criterion, optimizer, scheduler, num_epochs=200):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    #all_output, all_target = np.array([]), np.array([])

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    #del outputs

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                acc1, acc5 = accuracy(outputs,labels,topk=(1,5))

                #all_output.append(outputs.detach().cpu().numpy())
                #all_target.append(labels.detach().cpu().numpy())
                #all_output = np.append(all_output, outputs.detach().cpu().numpy())
                #all_target = np.append(all_target, labels.detach().cpu().numpy())

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            time_elapsed = time.time() - since
            print("Top 1 acc: ",acc1)
            print("Top 5 acc: ",acc5)
            print("This epoch takes ", time_elapsed)

            #all_output = np.concatenate(all_output)
            #all_target = np.concatenate(all_target)
            #auc = cal_auc(all_output, all_target)
            #print(f"* AUC: {auc:.5f}")

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            
            torch.save(model.state_dict(), "results_isic/epoch_"+str(epoch)+"_isic.pth")
            print("Saved model at ",epoch," epochs")

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, len(class_names))

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=200)
