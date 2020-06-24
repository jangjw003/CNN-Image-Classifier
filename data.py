import numpy as np
import cv2
import os
import pickle
import random

def pickler(X_d, y_d):
    pickle_file_x = open('X.pickle', 'wb')
    pickle.dump(X_d, pickle_file_x)
    pickle_file_x.close()
    
    pickle_file_y = open('y.pickle', 'wb')
    pickle.dump(y_d, pickle_file_y)
    pickle_file_y.close()
    
    print("Pickle Files Created...")

def training_data_generation(IMGSIZE, path_to_folder):
    labels = ['dogs', 'cats']
    training_set = []
    for label in labels:
        class_num = labels.index(label)
        full_path = os.path.join(path_to_folder, label)
        for img in os.listdir(full_path):
            array = cv2.imread(os.path.join(full_path, img), cv2.IMREAD_GRAYSCALE)
            n_array = cv2.resize(array, (IMGSIZE, IMGSIZE))
            training_set.append([n_array, class_num])
        
    print("Training data generated")
    
    random.shuffle(training_set)

    return training_set


if __name__ == "__main__":
    #data_location = "C:\Users\jangj\code\projects\dog_cat\data"
    data_location = r"C:\Users\jangj\code\projects\dog_cat\dataset\training_set"
    # 28 is the size of the image (width=28, height=28)
    IMGSIZE = 28

    data = training_data_generation(IMGSIZE, data_location)
    
    # separating the data into X and y
    X = []
    y = []
    for array, class_num in data:
        X.append(array)
        y.append(class_num)

    pickler(X, y)
    




