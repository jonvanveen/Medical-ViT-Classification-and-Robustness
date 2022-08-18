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
        #image = image[:, :, 0] # doesn't solve problem
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float32')#.reshape(-1, 2) # reshape caused error
        #sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            #sample = self.transform(sample)
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

    # additional unused parser args to satisfy swin script args
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    
    gc.collect()
    torch.cuda.empty_cache()

    args, unparsed = parser.parse_known_args()
    print(args)
    config = get_config(args)

    # load model
    model = build_model(config)
    #model.head = nn.Linear(in_features=768, out_features=10, bias=True)
    ckpt = torch.load(args.model)
    model.load_state_dict(ckpt,False) # False is risky - added to avoid mismatched state_dicts
  
    # fix the head
    # model.head = nn.Linear(in_features=768, out_features=10, bias=True)
    
    model.cuda()
    model.eval()
    
    # Transforms
# data_transforms = {
#     'train': transforms.Compose([
#         transforms.Resize((224,224)),
#         transforms.ToTensor()#,
#     ]),
#     'val': transforms.Compose([
#         transforms.Resize((224,224)),
#         transforms.ToTensor()#,
#     ]),
# }

# # Train on ISIC Dataset
# data_dir = '/content/drive/MyDrive/MLSP_Masters/ECE_697/data/isic/isic_org'

# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
#                   data_transforms[x])
#                   for x in ['train', 'val']}

# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, # batch size so small due to memory constraints
#               shuffle=True, num_workers=2)
#               for x in ['train', 'val']}

# dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# class_names = image_datasets['train'].classes

    # load data
    transform_list = [transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor()]
    transform_chain = transforms.Compose(transform_list)
    #item = LoadDataset(csv_file=args.csv, root_dir=args.data_dir, transform=transform_chain)#, download=True)
    item_tr = LoadDataset(csv_file=args.csv_tr, root_dir=args.data_dir+"/train", transform=transform_chain)
    item_ts = LoadDataset(csv_file=args.csv_ts, root_dir=args.data_dir+"/test", transform=transform_chain)
    train_loader = DataLoader(item_tr, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(item_ts, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # create save dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    print(x for (x,y) in test_loader)
    print(type(x) for (x,y) in test_loader)

    # load attack    
    # from autoattack import AutoAttack
    # adversary = AutoAttack(model, norm=args.norm, eps=args.epsilon, log_path=args.log_path,
    #     version=args.version)
    
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

    
    # x = torch.randn(2, 3, 224, 224)
    # print(classifier.predict(x).shape)
    # exit()

    print("Preparing data...")
    
    # l = [x for (x, y) in test_loader]
    # print("l: ",l)
    # l = [y for (x, y) in test_loader]
    # print("l: ",l)
    # x_test = torch.cat(l, 0)
    # y_test = torch.cat(l, 0)
    
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
        
    x_train, y_train = torch.cat(lxtr, 0), torch.cat(lytr, 0) #torch.FloatTensor(lxtr), torch.FloatTensor(lytr)
    print(torch.flatten(x_train).size())
    x_test, y_test = torch.cat(lxts, 0), torch.cat(lyts, 0)#torch.FloatTensor(lxts), torch.FloatTensor(lyts)
    print("Data prepared.")
    # print('xtest: ',x_test)
    print('xtest shape ', x_test.shape)
    
    # print(y_train)
    print(y_train.size())
    # print("")
    y_train = y_train[:799,:,0,0]
    x_ = torch.zeros(799,10)
    for i in range(0,799):
        x_[i][int(y_train[i])-1] = 1
    y_train = x_
    
    print(y_test.shape)
    y_test = y_test[:220,:,0,0]
    x_ = torch.zeros(220,10)
    for i in range(0,220):
        x_[i][int(y_test[i])-1] = 1
    y_test = x_
    print(y_test.size())
    #print(y_test)
    
    print("Fitting classifier...")
    # classifier.fit(x_train.detach().cpu().numpy(), y_train.detach().cpu().numpy(), batch_size=64, training_mode=True, nb_epochs=1)
    x_train = x_train.detach().cpu().numpy()
    y_train = y_train.detach().cpu().numpy()
    
    print('x_train ',np.shape(x_train))
    print('y_train ',np.shape(y_train))
    
    logits = classifier.predict(x)
    print(logits.shape)
    print(logits.argmax(1)) # or 0
    print(y)
    
    
    classifier.fit(x_train, y_train, batch_size=64, training_mode=True, nb_epochs=5)
    print("classifier fit: ",classifier.nb_classes)
    print("Classifier fitted.")
    
    predictions = classifier.predict(x_test.detach().cpu().numpy(),training_mode=False)
    print("classifier pred: ",classifier.nb_classes)
    print('predictions: ',predictions) 
    print('shape: ',predictions.shape)
    #accuracy = np.sum((abs(predictions - y_test.detach().cpu().numpy())/2))/np.max(y_test.shape)
    accuracy = np.sum(np.argmax(predictions[:10,:], axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("Accuracy on benign test examples: {}%".format(accuracy * 100))
    
    # Generate adversarial test examples
    attack = AutoAttack(estimator=classifier, eps=0.2)
    print('x test: ',type(x_test), x_test)
    x_test_adv = attack.generate(x=x_test)#.detach().cpu().numpy())
    
    # Evaluate the ART classifier on adversarial test examples
    predictions = classifier.predict(x_test_adv)
    #accuracy = np.sum((abs(predictions - y_test.detach().cpu().numpy())/2))/np.max(y_test.shape)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
    
    #l = [x for (x, y) in test_loader]
    #print("l: ",l,type(l[0]))
    # l = [y for (x, y) in test_loader]
    # y_test = torch.cat(l, 0)

    # example of custom version
    # if args.version == 'custom':
    #     adversary.attacks_to_run = ['apgd-ce', 'fab']
    #     adversary.apgd.n_restarts = 2
    #     adversary.fab.n_restarts = 2

    # run attack and save images
    # with torch.no_grad():
    #     if not args.individual:
    #         adv_complete = adversary.run_standard_evaluation(x_test[:args.n_ex], y_test[:args.n_ex],
    #             bs=args.batch_size)

    #         torch.save({'adv_complete': adv_complete}, '{}/{}_{}_1_{}_eps_{:.5f}.pth'.format(
    #             args.save_dir, 'aa', args.version, adv_complete.shape[0], args.epsilon))

    #     else:
    #         # individual version, each attack is run on all test points
    #         adv_complete = adversary.run_standard_evaluation_individual(x_test[:args.n_ex],
    #                 y_test[:args.n_ex], bs=args.batch_size)

    #         torch.save(adv_complete, '{}/{}_{}_individual_1_{}_eps_{:.5f}_plus_{}_cheap_{}.pth'.format(
    #             args.save_dir, 'aa', args.version, args.n_ex, args.epsilon))
