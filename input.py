import torch
import cv2
import numpy as np
from dog_cat import net

def predictor(array):
	array = torch.FloatTensor(array)
	array = array.view(-1, 1, 28, 28)
	predicted = model.forward(array)
	label = ['cats', 'dogs']
	p = label[predicted.argmax()]
	return p


def main():
	path = input("Path to Image: ")
	array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	n_array = cv2.resize(array, (28, 28))
	prediction = predictor(n_array)
	print(f"Predicted : {prediction}")


if __name__ == "__main__":
	model = torch.load('model.pth')
	model.eval()
	main()


