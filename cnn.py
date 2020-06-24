import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import os
import warnings

# Convolutional Neural Network Model
class CNN(nn.Module):
	def __init__(self):
		super().__init__()
		self.fit_done = False

		self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=4)
		self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=4)

		# (IMGSIZE - Kernel_size + 2 x padding)/Stride + 1
		self.l1 = nn.Linear(in_features=12*16, out_features=120)
		self.l2 = nn.Linear(in_features=120, out_features=64) 
		self.l3 = nn.Linear(in_features=64, out_features=10) 
		self.out = nn.Linear(in_features=10, out_features=2)
	
	def forward(self, x):
		assert torch.is_tensor(x), "x must be a tensor"
		x = F.relu(self.conv1(x))
		x = F.max_pool2d(x, stride=2, kernel_size=3)
		
		x = F.relu(self.conv2(x))
		x = F.max_pool2d(x, stride=2, kernel_size=3)

		x = x.view(-1, 12*16)

		x = F.relu(self.l1(x))
		x = F.relu(self.l2(x))
		x = F.relu(self.l3(x))
		x = self.out(x)
		
		return x 

	def fit(self, x_data, y_data, EPOCH, batch_size, learning_rate):
		print("Starting Fit")
		# putting the model to 'train' mode
		self.train()
		optimizer = optim.Adam(self.parameters(), lr=learning_rate)
		for epoch in range(EPOCH):
			for i in range(0, len(x_data), batch_size):
				# 28 is the size of the image; width, height both 28 pixels
				b_x = x_data[i:i+batch_size].view(-1, 1, 28, 28)
				b_y = y_data[i:i+batch_size]
				optimizer.zero_grad()
				prediction = self.forward(b_x)

				loss = F.cross_entropy(prediction, b_y)
				loss.backward(loss)
				optimizer.step()
			print(f"EPOCH:{epoch+1} \t LOSS:{loss}")
		print("Done Trainning")

		self.fit_done = True

	def predict(self, x):
		assert self.fit_done, "You must train your model before using it for prediction"
		output = self.forward(x)
		return output

	def save(self):
		if not os.path.exists('model.pth'):
			torch.save(model.state_dict(), 'model.pth')
		else:
			print("Model is already saved")

def load_data():
	# load training data
	pickle_in = open("X.pickle", "rb")
	X = torch.FloatTensor(pickle.load(pickle_in))
	pickle_in.close()

	pickle_in = open("y.pickle", "rb")
	y = torch.tensor(pickle.load(pickle_in))
	pickle_in.close()

	return [X, y]


if __name__ == "__main__":
	# loading in training data
	X, y = load_data()
	model = CNN()

	# defining parameters
	batch_size = 32
	lr = 0.001
	EPOCH = 60

	# putting the model in 'train' mode
	model.fit(X, y, EPOCH, batch_size, lr)
	model.save()






