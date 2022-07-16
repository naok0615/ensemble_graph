# coding: utf-8

import os
import random
import scipy.io
import pickle
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from os.path import join
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import default_loader


class worker_initializer():
    def __init__(self, manualSeed):
        self.manualSeed = manualSeed

    def worker_init_fn(self, worker_id):
        random.seed(self.manualSeed+worker_id)


# Stanford Dogs
class dogs(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root      = root
        self.train     = train
        self.transform = transform

        split = self.load_split()

        self.images_folder      = join(self.root, 'Images')
        self.annotations_folder = join(self.root, 'Annotation')
        
        self._breeds = list_dir(self.images_folder)
        self._images = [(annotation+'.jpg', idx) for annotation, idx in split]

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        path, target  = self._images[index]
        path  = join(self.images_folder, path)
        image = Image.open(path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, target

    def load_split(self):
        if self.train:
            split  = scipy.io.loadmat(join(self.root, 'train_list.mat'))['annotation_list']
            labels = scipy.io.loadmat(join(self.root, 'train_list.mat'))['labels']
        else:
            split  = scipy.io.loadmat(join(self.root, 'test_list.mat'))['annotation_list']
            labels = scipy.io.loadmat(join(self.root, 'test_list.mat'))['labels']

        split  = [item[0][0] for item in split]
        labels = [item[0]-1 for item in labels]
        return list(zip(split, labels))

def StanfordDogs(args):
    manualSeed = args.manualSeed
    args       = args.dataloader
    
    train_transform = transforms.Compose([transforms.RandomResizedCrop(224, ratio=(1, 1.3)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),])
    test_transform  = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),])
    
    train_dataset = dogs(root=args.data_path,
                         train=True,
                         transform=train_transform)
    test_dataset  = dogs(root=args.data_path,
                         train=False,
                         transform=test_transform)
    
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=args.train_shuffle,
                              num_workers=args.workers,
                              pin_memory=False,
                              drop_last=args.train_drop_last,
                              worker_init_fn=worker_initializer(manualSeed).worker_init_fn)
    test_loader  = DataLoader(test_dataset,
                              batch_size=args.batch_size,
                              shuffle=args.test_shuffle,
                              num_workers=args.workers,
                              pin_memory=False,
                              drop_last=args.test_drop_last,
                              worker_init_fn=worker_initializer(manualSeed).worker_init_fn)
    
    print("Training set stats:", len(train_dataset))
    print("Testing set stats :", len(test_dataset))
    
    return train_loader, test_loader

def StanfordDogs_split(args):
    manualSeed = args.manualSeed
    args       = args.dataloader
    
    train_transform = transforms.Compose([transforms.RandomResizedCrop(224, ratio=(1, 1.3)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),])
    test_transform  = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),])
    
    train_dataset = dogs(root=args.data_path,
                         train=True,
                         transform=train_transform)
    test_dataset  = dogs(root=args.data_path,
                         train=True,
                         transform=test_transform)
    
    if os.path.isfile('./split_index/SDogs/index_001.pkl'):
        print("load index for split")
        with open('./split_index/SDogs/index_001.pkl', 'rb') as f:
            train_index, test_index = pickle.load(f)
    else:
        print("mask index for split")
        target_list = [ train_dataset._images[i][1] for i in range(len(train_dataset._images)) ]
        train_index, test_index = train_test_split( list(range(len(target_list))), test_size=0.5, stratify=target_list )
        save_index  = ([train_index, test_index])
        with open('./split_index/SDogs/index_001.pkl', 'wb') as f:
            pickle.dump(save_index, f)
        
    print("Training set stats:", len(train_index))
    print("Testing set stats :", len(test_index))
    
    train_dataset_split = Subset(train_dataset, train_index)
    test_dataset_split  = Subset(test_dataset,  test_index)
    
    train_loader = DataLoader(train_dataset_split,
                              batch_size=args.batch_size,
                              shuffle=args.train_shuffle,
                              num_workers=args.workers,
                              pin_memory=False,
                              drop_last=args.train_drop_last,
                              worker_init_fn=worker_initializer(manualSeed).worker_init_fn)
    test_loader  = DataLoader(test_dataset_split,
                              batch_size=args.batch_size,
                              shuffle=args.test_shuffle,
                              num_workers=args.workers,
                              pin_memory=False,
                              drop_last=args.test_drop_last,
                              worker_init_fn=worker_initializer(manualSeed).worker_init_fn)

    return train_loader, test_loader


# Stanford Cars
class Cars(Dataset):
    file_list = {
        'imgs': ('http://imagenet.stanford.edu/internal/car196/car_ims.tgz', 'car_ims.tgz'),
        'annos': ('http://imagenet.stanford.edu/internal/car196/cars_annos.mat', 'cars_annos.mat')
    }

    def __init__(self, root, train=True, transform=None):
        self.root      = root
        self.train     = train
        self.transform = transform
        self.loader    = default_loader

        loaded_mat = scipy.io.loadmat(join(self.root, self.file_list['annos'][1]))
        loaded_mat = loaded_mat['annotations'][0]
        
        self._images = []
        for item in loaded_mat:
            if self.train != bool(item[-1][0]):
                path  = str(item[0][0])
                label = int(item[-2][0]) - 1
                self._images.append((path, label))
                
    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        path, target = self._images[index]
        path  = join(self.root, path)
        image = self.loader(path)
        
        if self.transform is not None:
            image = self.transform(image)

        return image, target


def StanfordCars(args):
    manualSeed = args.manualSeed
    args       = args.dataloader
    
    train_transform = transforms.Compose([transforms.RandomResizedCrop(224, ratio=(1, 1.3)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),])
    test_transform  = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),])
        
    train_dataset = Cars(root=args.data_path,
                         train=True,
                         transform=train_transform)
    test_dataset  = Cars(root=args.data_path,
                         train=False,
                         transform=test_transform)
    
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=args.train_shuffle,
                              num_workers=args.workers,
                              pin_memory=False,
                              drop_last=args.train_drop_last,
                              worker_init_fn=worker_initializer(manualSeed).worker_init_fn)
    test_loader  = DataLoader(test_dataset,
                              batch_size=args.batch_size,
                              shuffle=args.test_shuffle,
                              num_workers=args.workers,
                              pin_memory=False,
                              drop_last=args.test_drop_last,
                              worker_init_fn=worker_initializer(manualSeed).worker_init_fn)
    
    print("Training set stats:", len(train_dataset))
    print("Testing set stats :", len(test_dataset))
    
    return train_loader, test_loader

def StanfordCars_split(args):
    manualSeed = args.manualSeed
    args       = args.dataloader
    
    train_transform = transforms.Compose([transforms.RandomResizedCrop(224, ratio=(1, 1.3)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),])
    test_transform  = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),])
    
    train_dataset = Cars(root=args.data_path,
                         train=True,
                         transform=train_transform)
    test_dataset  = Cars(root=args.data_path,
                         train=True,
                         transform=test_transform)
    
    if os.path.isfile('./split_index/SCars/index_001.pkl'):
        print("load index for split")
        with open('./split_index/SCars/index_001.pkl', 'rb') as f:
            train_index, test_index = pickle.load(f)
    else:
        print("make index for split")
        target_list = [ train_dataset._images[i][1] for i in range(len(train_dataset._images)) ]
        train_index, test_index = train_test_split( list(range(len(target_list))), test_size=0.5, stratify=target_list )
        save_index  = ([train_index, test_index])
        with open('./split_index/SCars/index_001.pkl', 'wb') as f:
            pickle.dump(save_index, f)
        
    print("Training set stats:", len(train_index))
    print("Testing set stats :", len(test_index))
    
    train_dataset_split = Subset(train_dataset, train_index)
    test_dataset_split  = Subset(test_dataset,  test_index)
    
    train_loader = DataLoader(train_dataset_split,
                              batch_size=args.batch_size,
                              shuffle=args.train_shuffle,
                              num_workers=args.workers,
                              pin_memory=False,
                              drop_last=args.train_drop_last,
                              worker_init_fn=worker_initializer(manualSeed).worker_init_fn)
    
    test_loader = DataLoader(test_dataset_split,
                             batch_size=args.batch_size,
                             shuffle=args.test_shuffle,
                             num_workers=args.workers,
                             pin_memory=False,
                             drop_last=args.test_drop_last,
                             worker_init_fn=worker_initializer(manualSeed).worker_init_fn)

    return train_loader, test_loader


# CUB-200-2011
class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    file_id  = '1hbzc_P1FuxMkcabkgn9ZKinBwW683j45'
    filename = 'CUB_200_2011.tgz'
    tgz_md5  = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None):
        self.root      = root
        self.train     = train
        self.transform = transform
        self.loader    = default_loader

        images = pd.read_csv(join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ', names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(join(self.root, 'CUB_200_2011', 'image_class_labels.txt'), sep=' ', names=['img_id', 'target'])
        train_test_split   = pd.read_csv(join(self.root, 'CUB_200_2011', 'train_test_split.txt'), sep=' ', names=['img_id', 'is_training_img'])
        
        data      = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        class_names = pd.read_csv(join(self.root, 'CUB_200_2011', 'classes.txt'), sep=' ', names=['class_name'], usecols=[1])
        self.class_names = class_names['class_name'].to_list()
        
        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path   = join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        image  = self.loader(path)

        if self.transform is not None:
            image = self.transform(image)
            
        return image, target

def CUB2011(args):
    manualSeed = args.manualSeed
    args       = args.dataloader
    
    train_transform = transforms.Compose([transforms.RandomResizedCrop(224, ratio=(1, 1.3)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),])
    test_transform  = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),])
                    
    train_dataset = Cub2011(root=args.data_path,
                            train=True,
                            transform=train_transform)
    test_dataset  = Cub2011(root=args.data_path,
                            train=False,
                            transform=test_transform)
    
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=args.train_shuffle,
                              num_workers=args.workers,
                              pin_memory=False,
                              drop_last=args.train_drop_last,
                              worker_init_fn=worker_initializer(manualSeed).worker_init_fn)
    test_loader  = DataLoader(test_dataset,
                              batch_size=args.batch_size,
                              shuffle=args.test_shuffle,
                              num_workers=args.workers,
                              pin_memory=False,
                              drop_last=args.test_drop_last,
                              worker_init_fn=worker_initializer(manualSeed).worker_init_fn)
    
    print("Training set stats:", len(train_dataset))
    print("Testing set stats :", len(test_dataset))
    
    return train_loader, test_loader

def CUB2011_split(args):
    manualSeed = args.manualSeed
    args       = args.dataloader
    
    train_transform = transforms.Compose([transforms.RandomResizedCrop(224, ratio=(1, 1.3)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),])
    test_transform  = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),])
    
    train_dataset = Cub2011(root=args.data_path,
                            train=True,
                            transform=train_transform)
    test_dataset  = Cub2011(root=args.data_path,
                            train=True,
                            transform=test_transform)
    
    if os.path.isfile('./split_index/CUB2011/index_001.pkl'):
        print("load index for split")
        with open('./split_index/CUB2011/index_001.pkl', 'rb') as f:
            train_index, test_index = pickle.load(f)
    else:
        print("make index for split")
        target_list = train_dataset.data.target.to_list()
        img_id_list = [i for i in range(len(target_list))]
        train_index, test_index = train_test_split( img_id_list, test_size=0.5, stratify=target_list )
        save_index = ([train_index, test_index])
        with open('./split_index/CUB2011/index_001.pkl', 'wb') as f:
            pickle.dump(save_index, f)
        
    print("Training set stats:", len(train_index))
    print("Testing set stats :", len(test_index))
    
    train_dataset_split = Subset(train_dataset, train_index)
    test_dataset_split  = Subset(test_dataset,  test_index)
    
    train_loader = DataLoader(train_dataset_split,
                              batch_size=args.batch_size,
                              shuffle=args.train_shuffle,
                              num_workers=args.workers,
                              pin_memory=False,
                              drop_last=args.train_drop_last,
                              worker_init_fn=worker_initializer(manualSeed).worker_init_fn)
    test_loader  = DataLoader(test_dataset_split,
                              batch_size=args.batch_size,
                              shuffle=args.test_shuffle,
                              num_workers=args.workers,
                              pin_memory=False,
                              drop_last=args.test_drop_last,
                              worker_init_fn=worker_initializer(manualSeed).worker_init_fn)

    return train_loader, test_loader


# CIFAR-10
def CIFAR10(args):    
    manualSeed = args.manualSeed
    args       = args.dataloader
    
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    test_transform  = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])    
    
    train_dataset = torchvision.datasets.CIFAR10(args.data_path, 
                                                 train=True, 
                                                 download=False, 
                                                 transform=train_transform)
    test_dataset  = torchvision.datasets.CIFAR10(args.data_path, 
                                                 train=False, 
                                                 download=False,
                                                 transform=test_transform)
    
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=args.train_shuffle,
                              num_workers=args.workers,
                              pin_memory=False,
                              drop_last=args.train_drop_last,
                              worker_init_fn=worker_initializer(manualSeed).worker_init_fn)
    test_loader  = DataLoader(test_dataset,
                              batch_size=args.batch_size,
                              shuffle=args.test_shuffle,
                              num_workers=args.workers,
                              pin_memory=False,
                              drop_last=args.test_drop_last,
                              worker_init_fn=worker_initializer(manualSeed).worker_init_fn)
    
    return train_loader, test_loader

def CIFAR10_split(args):
    manualSeed = args.manualSeed
    args       = args.dataloader
    
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    test_transform  = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),]) 

    train_dataset = torchvision.datasets.CIFAR10(args.data_path, 
                                                 train=True, 
                                                 download=False, 
                                                 transform=train_transform)
    test_dataset  = torchvision.datasets.CIFAR10(args.data_path, 
                                                 train=True, 
                                                 download=False, 
                                                 transform=test_transform)

    if os.path.isfile('./split_index/CIFAR10/index_001.pkl'):
        print("load index for split")
        with open('./split_index/CIFAR10/index_001.pkl', 'rb') as f:
            train_index, test_index = pickle.load(f)
    else:
        print("make index for split")
        target_list = train_dataset.targets
        train_index, test_index = train_test_split(list(range(len(target_list))), test_size=0.2, stratify=target_list)
        save_index = ([train_index, test_index])
        with open('./split_index/CIFAR10/index_001.pkl', 'wb') as f:
            pickle.dump(save_index, f)
            
    print("Training set stats:", len(train_index))
    print("Testing set stats :", len(test_index))

    train_dataset_split = Subset(train_dataset, train_index)
    test_dataset_split  = Subset(test_dataset,  test_index)

    train_loader = DataLoader(train_dataset_split,
                              batch_size=args.batch_size,
                              shuffle=args.train_shuffle,
                              num_workers=args.workers,
                              pin_memory=False,
                              drop_last=args.train_drop_last,
                              worker_init_fn=worker_initializer(manualSeed).worker_init_fn)
    test_loader  = DataLoader(test_dataset_split,
                              batch_size=args.batch_size,
                              shuffle=args.test_shuffle,
                              num_workers=args.workers,
                              pin_memory=False,
                              drop_last=args.test_drop_last,
                              worker_init_fn=worker_initializer(manualSeed).worker_init_fn)

    return train_loader, test_loader


# CIFAR-100
def CIFAR100(args):
    manualSeed = args.manualSeed
    args       = args.dataloader
    
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    test_transform  = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])       

    train_dataset = torchvision.datasets.CIFAR100(args.data_path, 
                                                  train=True, 
                                                  download=False, 
                                                  transform=train_transform)
    test_dataset  = torchvision.datasets.CIFAR100(args.data_path, 
                                                  train=False, 
                                                  download=False, 
                                                  transform=test_transform)
    
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=args.train_shuffle,
                              num_workers=args.workers,
                              pin_memory=False,
                              drop_last=args.train_drop_last,
                              worker_init_fn=worker_initializer(manualSeed).worker_init_fn)
    test_loader  = DataLoader(test_dataset,
                              batch_size=args.batch_size,
                              shuffle=args.test_shuffle,
                              num_workers=args.workers,
                              pin_memory=False,
                              drop_last=args.test_drop_last,
                              worker_init_fn=worker_initializer(manualSeed).worker_init_fn)
    
    return train_loader, test_loader

def CIFAR100_split(args):
    manualSeed = args.manualSeed
    args       = args.dataloader
    
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    test_transform  = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),]) 

    # 同じ学習用データを２つ読み込み
    train_dataset = torchvision.datasets.CIFAR100(args.data_path, 
                                                  train=True, 
                                                  download=False, 
                                                  transform=train_transform)
    test_dataset  = torchvision.datasets.CIFAR100(args.data_path, 
                                                  train=True, 
                                                  download=False, 
                                                  transform=test_transform)

    if os.path.isfile('./split_index/CIFAR100/index_001.pkl'):
        print("load index for split")
        with open('./split_index/CIFAR100/index_001.pkl', 'rb') as f:
            train_index, test_index = pickle.load(f)
    else:
        print("make index for split")
        target_list = train_dataset.targets
        train_index, test_index = train_test_split(list(range(len(target_list))), test_size=0.2, stratify=target_list)
        save_index = ([train_index, test_index])
        with open('./split_index/CIFAR100/index_001.pkl', 'wb') as f:
            pickle.dump(save_index, f)

    print("Training set stats:", len(train_index))
    print("Testing set stats :", len(test_index))

    train_dataset_split = Subset(train_dataset, train_index)
    test_dataset_split  = Subset(test_dataset,  test_index)

    train_loader = DataLoader(train_dataset_split,
                              batch_size=args.batch_size,
                              shuffle=args.train_shuffle,
                              num_workers=args.workers,
                              pin_memory=False,
                              drop_last=args.train_drop_last,
                              worker_init_fn=worker_initializer(manualSeed).worker_init_fn)
    test_loader  = DataLoader(test_dataset_split,
                              batch_size=args.batch_size,
                              shuffle=args.test_shuffle,
                              num_workers=args.workers,
                              pin_memory=False,
                              drop_last=args.test_drop_last,
                              worker_init_fn=worker_initializer(manualSeed).worker_init_fn)

    return train_loader, test_loader