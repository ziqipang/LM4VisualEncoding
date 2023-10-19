# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json
import numpy as np
import tarfile
from PIL import Image

import torch
from torch.utils import data
from torchvision import datasets, transforms
from torchvision.datasets import VOCSegmentation
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder

class ImageTarDataset(data.Dataset):
    def __init__(self, tar_file, return_labels=False, transform=transforms.ToTensor()):
        '''
        return_labels:
        Whether to return labels with the samples
        transform:
        A function/transform that takes in an PIL image and returns a transformed version. E.g, transforms.RandomCrop
        '''
        self.tar_file = tar_file
        self.tar_handle = None
        categories_set = set()
        self.tar_members = []
        self.categories = {}
        self.categories_to_examples = {}
        with tarfile.open(tar_file, 'r:') as tar:
            for index, tar_member in enumerate(tar.getmembers()):
                if tar_member.name.count('/') != 2:
                    continue
                category = self._get_category_from_filename(tar_member.name)
                categories_set.add(category)
                self.tar_members.append(tar_member)
                cte = self.categories_to_examples.get(category, [])
                cte.append(index)
                self.categories_to_examples[category] = cte
        categories_set = sorted(categories_set)
        for index, category in enumerate(categories_set):
            self.categories[category] = index
        self.num_examples = len(self.tar_members)
        self.indices = np.arange(self.num_examples)
        self.num = self.__len__()
        print("Loaded the dataset from {}. It contains {} samples.".format(tar_file, self.num))
        self.return_labels = return_labels
        self.transform = transform
  
    def _get_category_from_filename(self, filename):
        begin = filename.find('/')
        begin += 1
        end = filename.find('/', begin)
        return filename[begin:end]
  
    def __len__(self):
        return self.num_examples
  
    def __getitem__(self, index):
        index = self.indices[index]
        if self.tar_handle is None:
            self.tar_handle = tarfile.open(self.tar_file, 'r:')
    
        sample = self.tar_handle.extractfile(self.tar_members[index])
        image = Image.open(sample).convert('RGB')
        image = self.transform(image)
    
        if self.return_labels:
            category = self.categories[self._get_category_from_filename(
                self.tar_members[index].name)]
            return image, category
        return image


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        if args.data_type == 'tar':
            root = os.path.join(args.data_path, 'train.tar' if is_train else 'val.tar')
            dataset = ImageTarDataset(root, return_labels=True, transform=transform)
        elif args.data_type == 'folder':
            root = os.path.join(args.data_path, 'train' if is_train else 'val')
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

