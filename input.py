import torch
import cv2
import numpy as np
from model import net

def predictor(array, model):
	array = torch.FloatTensor(array)
	array = array.view(-1, 1, 28, 28)
	predicted = model.forward(array)
	label = ['cat', 'dog']
	p = label[predicted.argmax()]
	return p


def main():
	# Importing CNN	
	cnn = net()
	vals = torch.load('model.pth', map_location=lambda storage, loc: storage)	
	cnn.load_state_dict(vals)
	cnn.eval()
	
	# Getting Path to the Image
	path = input("Path to Image: ")
	array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	n_array = cv2.resize(array, (28, 28))
	prediction = predictor(n_array, cnn)
	print(f"Predicted : {prediction}")


if __name__ == "__main__":
	main()


