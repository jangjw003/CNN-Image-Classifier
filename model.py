import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os


pickle_file = open('x.pickle', 'rb')
X = pickle.load(pickle_file)
pickle_file.close()

pickle_file = open('y.pickle', 'rb')
y = pickle.load(pickle_file)
pickle_file.close()


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

EPOCH = 70
batch_size = 32
learining_rate = 0.001
IMGSIZE = 28

model = net()

optimizer = optim.Adam(model.parameters(), lr=learining_rate)

#print(torch.is_tensor(X))

model.train()


if __name__ == '__main__':
	for epoch in range(EPOCH):
		for i in range(0, len(X), batch_size):
			b_x = X[i:i+batch_size].view(-1, 1, IMGSIZE, IMGSIZE) # (N, C, W, H)
			b_y = y[i:i+batch_size]
			optimizer.zero_grad()

			prediction = model.forward(b_x)
			loss = F.cross_entropy(prediction, b_y)
			loss.backward(loss)
			optimizer.step()
		
		print(f"Current EPOCH: {epoch+1}, with loss: {loss}")

	if not os.path.exists('model.pth'):
		torch.save(model.state_dict(), 'model.pth')
	else:
		print("Model is already saved")
