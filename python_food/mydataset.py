import os
import json
from os.path import join

import numpy as np
# import scipy
# from scipy import io
# import scipy.misc
from PIL import Image
# import pandas as pd
# import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
# from torchvision.datasets import VisionDataset
# from torchvision.datasets.folder import default_loader
# from torchvision.datasets.utils import download_url, list_dir, check_integrity, extract_archive, verify_str_arg

# import imageio
import cv2
import pdb

class Nutrition(Dataset):
    def __init__(self, image_path, txt_dir, transform=None):

        file = open(txt_dir, 'r')
        lines = file.readlines()
        self.images = []
        self.labels = []
        self.total_calories = []
        self.total_mass = []
        self.total_fat = []
        self.total_carb = []
        self.total_protein = []
        # pdb.set_trace()
        for line in lines:
            image = line.split()[0]#以空格划分取第一个  # side_angles/dish_1550862840/frames_sampled5/camera_A_frame_010.jpeg
            label = line.strip().split()[1]  # 类别 1-
            calories = line.strip().split()[2]
            mass = line.strip().split()[3]
            fat = line.strip().split()[4]
            carb = line.strip().split()[5]
            protein = line.strip().split()[6]

            self.images += [os.path.join(image_path, image)]  # 每张图片路径
            self.labels += [str(label)]
            self.total_calories += [np.array(float(calories))]
            self.total_mass += [np.array(float(mass))]
            self.total_fat += [np.array(float(fat))]
            self.total_carb += [np.array(float(carb))]
            self.total_protein += [np.array(float(protein))]
        # pdb.set_trace()
        # self.transform_rgb = transform[0]

        self.transform = transform
       
    def __getitem__(self, index):
        # img = cv2.imread(self.images[index])  
        # try:
        #     # img = cv2.resize(img, (self.imsize, self.imsize))
        #     img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) # cv2转PIL
        # except:
        #     print("图片有误：",self.images[index])
        img = Image.open(self.images[index]).convert('RGB')
        if self.transform is not None:
            try:
                #lmj  RGB-D图像尺寸不同,按照不同比例缩放
                if 'realsense_overhead' in self.images[index]:
                    # pdb.set_trace()
                    self.transform.transforms[0].size = (267, 356)
                    # print(self.transform)
                img = self.transform(img)
            except:
                # print('trans_img', img)
                print('trans_img有误')
        return img, self.labels[index], self.total_calories[index], self.total_mass[index], self.total_fat[index], self.total_carb[index], self.total_protein[index]

    def __len__(self):
        return len(self.images)



