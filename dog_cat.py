import os
import cv2
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


path_to_data = "/home/jw/code/ml/projects/dog_cat/dataset/training_set"
path_to_data2 = "/home/jw/code/ml/projectsdog_cat/dataset2/dataset/training_set"

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
        self.l1 = nn.Linear(in_features= 12*16, out_features= 120)
        self.l2 = nn.Linear(in_features= 120, out_features= 60)
        self.l3 = nn.Linear(in_features= 60, out_features= 10)
        self.out = nn.Linear(in_features=10, out_features=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, stride=2, kernel_size=3)


        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, stride=2, kernel_size=3)

        x = x.reshape(-1, 12*16)

        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.out(x)

        return x               #F.softmax(x, dim=1)

'''
# INGSIZE per Layer
conv1: 28 - 4 + 1 = 25
maxpool2d (1) : (25 - 3)/2 + 1 = 12
conv2: 12 - 4 + 1 = 9
maxpool2d (2) : (9 -3)/2 + 1 = 4  
'''

training_data_generator()
random.shuffle(training_set)


X = []
y = []

for array, label in training_set:
    X.append(array)
    y.append(label)

X = torch.FloatTensor(X)
y = torch.tensor(y)

EPOCH = 70
batch_size = 32
learining_rate = 0.001

model = net()

optimizer = optim.Adam(model.parameters(), lr=learining_rate)

#print(torch.is_tensor(X))

model.train()

for epoch in range(EPOCH):
    for i in range(0, len(training_set), batch_size):
        b_x = X[i:i+batch_size].view(-1, 1, IMGSIZE, IMGSIZE) # (N, C, W, H)
        b_y = y[i:i+batch_size]
        optimizer.zero_grad()

        prediction = model.forward(b_x)
        loss = F.cross_entropy(prediction, b_y)
        loss.backward(loss)
        optimizer.step()
    print(f"Current EPOCH: {epoch+1}, with loss: {loss}")

torch.save(model, 'model.pth')

