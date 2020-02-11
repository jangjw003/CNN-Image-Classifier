import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


path_to_data = "/home/jw/code/ml/projects/dog_cat/dataset/trainnig_set"

label = ['cats', 'dogs']


# Size of images
IMGSIZE = 28

training_set = []


# getting training data
def training_data_generator():
    for i in label:
        class_num = label.index(i)
        path_to_class = os.path.join(path_to_data, i)
        for img in os.listdir(path_to_class):          
            array = cv2.imread(os.path.join(path_to_class, img), cv2.IMREAD_GRAYSCALE)
            resized = cv2.resize(array, (IMGSIZE, IMGSIZE))
            training_set.append([resized, class_num])

# Neural Netowrk
class net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=4)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=4)


        # (IMGSIZE - Kernel_size + 2 x padding)/Stride + 1 
        self.l1 = nn.Linear(in_features= 12*16, out_channels= 120)
        self.l2 = nn.Linear(in_features= 120, out_channels= 60)
        self.l3 = nn.Linear(in_features= 60, out_features= 10)
        self.out = nn.Linear(in_features=10, out_features=2)

    def forward(x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, stride=2, kernel_size=3)


        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, stride=2, kernel_size=3)

        x = x.reshape(-1, 12*16)

        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.out(x)

        return x

'''
# INGSIZE per Layer

conv1: 28 - 4 + 1 = 25
maxpool2d (1) : (25 - 3)/2 + 1 = 12

conv2: 12 - 4 + 1 = 9
maxpool2d (2) : (9 -3)/2 + 1 = 4  
'''

training_data_generator()

X = []
y = []

for array, label in training_set:
    X.append(torch.as_tensor(array))
    y.append(torch.as_tensor(label))

EPOCH = 10
batch_size = 32


for epoch in range(EPOCH):
    

    for i in range(batch_size):




