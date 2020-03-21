import os
import cv2
import random
import torch
import pickle

# getting training data
def training_data_generator(IMGSIZE, path_to_data):
	training_set = [] 
	label = ['cats', 'dogs']
	for i in label:
		class_num = label.index(i)
		path_to_class = os.path.join(path_to_data, i)
    	
		for img in os.listdir(path_to_class):
			array = cv2.imread(os.path.join(path_to_class, img), cv2.IMREAD_GRAYSCALE)
			resized = cv2.resize(array, (IMGSIZE, IMGSIZE))
			training_set.append([resized, class_num])
			#print(class_num)
    
	random.shuffle(training_set)
	return training_set


def finalizer(dataset):
	X = []
	y = []

	for array, label in dataset:
		X.append(array)
		y.append(label)

	X = torch.FloatTensor(X)
	y = torch.tensor(y)
	print(X)
	return X, y

def pickler(x, y):
	pickle_file = open('x.pickle', "wb")	
	pickle.dump(x, pickle_file)
	pickle_file.close()

	pickle_file = open('y.pickle', "wb")
	pickle.dump(y, pickle_file)
	pickle_file.close()
    
	print("Pickles dumped")


def main():
	path_to_data = "/home/jw/code/ml/projects/dog_cat/dataset/training_set"	
	#size of IMG
	size_of_img = 28
	data = training_data_generator(size_of_img, path_to_data)
	X, y = finalizer(data)
	pickler(X, y)

if __name__ == "__main__":
	main()


