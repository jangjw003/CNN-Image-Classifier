import os
import cv2
import torch
from cnn import CNN

def predictor(array, model):
    prediction = model.predict(array)
    label = ['dog', 'cat']
    class_of_image = label[prediction.argmax()]
    print(f"OUTPUT: Image of {class_of_image}")

def image_to_array(filepath):
    array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    n_array = cv2.resize(array, (28, 28))
    x_predict = torch.FloatTensor(n_array).view(-1, 1, 28, 28)
    return x_predict

if __name__ == "__main__":
    # loading cnn model
    model = CNN()
    vals = torch.load('model.pth', map_location=lambda storage, loc: storage)
    model.load_state_dict(vals)
    model.eval()

    # grabbing image
    path = input("Input path to Image: ")
    array =image_to_array(path)
    predictor(array, model)