import numpy as np
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import torchvision
from torchvision import transforms

root_path = './data'


def make_dataset(image_list, label_list):
    len_ = len(label_list)
    images = [(image_list[i].strip(), label_list[i]) for i in range(len_)]
    return images


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.resize([224, 224]).convert('RGB')


def default_loader(path):
    return pil_loader(path)


class MyData(Dataset):
    def __init__(self, train=True, loader=default_loader):
        self._train = train
        self._loader = loader
        if self._train:
            # all face pic
            train_image_list_path1 = os.path.join('./CelebData', "fake_train.txt")
            train_image_list1 = open(train_image_list_path1).readlines()

            train_image_list_path2 = os.path.join('./CelebData', "real_train.txt")
            train_image_list2 = open(train_image_list_path2).readlines()
            #
            train_image_list1.extend(train_image_list2)

            train_image_list = train_image_list1

            # face label
            train_label_list_path1 = os.path.join('./CelebData', "fake_train_label.txt")
            train_label_list1 = np.loadtxt(train_label_list_path1)

            train_label_list_path2 = os.path.join('./CelebData', "real_train_label.txt")
            train_label_list2 = np.loadtxt(train_label_list_path2)
            #
            train_label_list = np.concatenate((train_label_list1, train_label_list2))

            self.data_list = make_dataset(train_image_list, train_label_list)
        else:
            test_image_list_path1 = os.path.join('./CelebData', "fake_test.txt")
            test_image_list1 = open(test_image_list_path1).readlines()

            test_image_list_path2 = os.path.join('./CelebData', "real_test.txt")
            test_image_list2 = open(test_image_list_path2).readlines()

            test_image_list1.extend(test_image_list2)

            test_image_list = test_image_list1

            # face label
            test_label_list_path1 = os.path.join('./CelebData', "fake_test_label.txt")
            test_label_list1 = np.loadtxt(test_label_list_path1)

            test_label_list_path2 = os.path.join('./CelebData', "real_test_label.txt")
            test_label_list2 = np.loadtxt(test_label_list_path2)

            test_label_list = np.concatenate((test_label_list1, test_label_list2))

            # # img
            # test_image_list_path = os.path.join(root_path, "Deepfakes.txt")
            # test_image_list = open(test_image_list_path).readlines()
            #
            # # img labels
            # test_label_list_path = os.path.join(root_path, "Deepfakes_label.txt")
            # test_label_list = np.loadtxt(test_label_list_path)

            self.data_list = make_dataset(test_image_list, test_label_list)

    def __getitem__(self, index):
        img, label = self.data_list[index]

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        img = self._loader(os.path.join(root_path, img))
        img = transform(img)
        return img, label

    def __len__(self):
        return len(self.data_list)


if __name__ == "__main__":
    trainset = MyData(train=True)
    img, label = trainset.__getitem__(1)
    print(len(img))
    print(img.shape, label)




