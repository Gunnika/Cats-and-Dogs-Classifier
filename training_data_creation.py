import numpy as np #for array operations
import matplotlib.pyplot as plt #for plotting graphs
import os #to iterate through directories, joining operation
import cv2 #to do image operations
import random
import pickle

DATADIR = "/home/gunnika/Downloads/kagglecatsanddogs/PetImages"
CATEGORIES = ['Dog', 'Cat']


IMG_SIZE = 50
training_data = []

def create_training_data():
    for category in CATEGORIES:
        path=os.path.join(DATADIR, category) 
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                #the resized image array:
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()

#shuffle the data
random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)
    
#we can't pass a list of features to the neural netwrok, should be a numpy array
#-1 means it can be any no.
# 1 for gray scale
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE,1)

#to save the model and tweak later
pickle_out= open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out= open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)