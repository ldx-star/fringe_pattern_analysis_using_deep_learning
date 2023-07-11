import torch
import os
import cv2
import torchvision.transforms
import torchvision
from torch.utils.data import Dataset
import numpy as np


def read_patterns_img(img_dir, split_nums, k, is_train=True):
    fname = os.path.join(img_dir, "train" if is_train else "test")

    input_path = os.path.join(fname, "input")
    ground_path = os.path.join(fname, "ground")
    denominator_path = os.path.join(fname, "denominator")
    numerator_path = os.path.join(fname, "numerator")

    img_len = len(os.listdir(input_path))
    read_nums = int(img_len / split_nums)

    input_imgs = []
    ground_imgs = []
    denominator_imgs = []
    numerator_imgs = []

    for i in range(read_nums):
        read_NO = read_nums * k + i
        if (read_NO <= img_len):
            input_img = cv2.imread(os.path.join(input_path, f"{read_NO}.bmp"), 0)
            ground_img = cv2.imread(os.path.join(ground_path, f"{read_NO}.tiff"), cv2.IMREAD_UNCHANGED)
            numerator_img = cv2.imread(os.path.join(numerator_path, f"{read_NO}.tiff"), cv2.IMREAD_UNCHANGED)
            denominator_img = cv2.imread(os.path.join(denominator_path, f"{read_NO}.tiff"), cv2.IMREAD_UNCHANGED)

            input_imgs.append(input_img)
            ground_imgs.append(ground_img)
            denominator_imgs.append(denominator_img)
            numerator_imgs.append(numerator_img)

    return input_imgs, ground_imgs, numerator_imgs, denominator_imgs


class Dataset(Dataset):
    def __init__(self, is_train, img_dir, split_nums, k):
        input_imgs, ground_imgs, numerator_imgs, denominator_imgs = read_patterns_img(img_dir, split_nums, k, is_train)
        self.input_images = [self.to_tensor(input_img) for input_img in input_imgs]
        self.ground_images = [self.to_tensor(ground_img) for ground_img in ground_imgs]
        self.numerator_images = [self.to_tensor(numerator_img) for numerator_img in numerator_imgs]
        self.denominator_images = [self.to_tensor(denominator_img) for denominator_img in denominator_imgs]

    def normalize_image(self, img):
        img = img.astype(np.float16)
        img /= 255
        toTensor = torchvision.transforms.ToTensor()
        img = toTensor(img)

        return img

    def to_tensor(self, img):
        img = img.astype(np.float16)
        toTensor = torchvision.transforms.ToTensor()
        img = toTensor(img)
        return img

    def __getitem__(self, idx):
        input_img, ground_img, numerator_img, denominator_img = self.input_images[idx], self.ground_images[idx], \
            self.numerator_images[idx], self.denominator_images[idx]
        return input_img, ground_img, numerator_img, denominator_img

    def __len__(self):
        return len(self.input_images)
