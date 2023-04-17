import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import os
import torchvision
from torchvision import transforms


def make_dataset(image_list, label_list):
    len_ = len(image_list)
    images = [(image_list[i].strip(),  label_list[i]) for i in range(len_)]
    return images


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.resize([224, 224]).convert('RGB')


def default_loader(path):
    return pil_loader(path)


class MyData(Dataset):
    def __init__(self, train=True, transform=None, crop_size = 224, loader=default_loader):

        self._train = train
        self._loader = loader
        self.crop_size = crop_size
        self.loader = loader
        if self._train:
            # img
            train_image_list_path = "./FFdata/5帧/all_train.txt"
            train_image_list = open(train_image_list_path).readlines()
            # img labels
            train_label_list_path = "./FFdata/5帧/all_train_label.txt"
            train_label_list = np.loadtxt(train_label_list_path)

            self.data_list = make_dataset(train_image_list, train_label_list)

        else:
            # img
            test_image_list_path = "./FFdata/5帧/all_test.txt"
            test_image_list = open(test_image_list_path).readlines()

            # img labels
            test_label_list_path = "./FFdata/5帧/all_test_label.txt"
            test_label_list = np.loadtxt(test_label_list_path)
            self.data_list = make_dataset(test_image_list, test_label_list)

    def __getitem__(self, index):
        imgpath, label = self.data_list[index]

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        root_path = './'

        img = self._loader(imgpath)
        img = transform(img)

        return img, label

    def __len__(self):
        return len(self.data_list)