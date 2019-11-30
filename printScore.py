import torch
import os
import pickle


file = open("scores", "rb")
data = pickle.load(file)
file.close()

for key in data.keys():
    print("The key is ",key)
    print("The score and result are ", data[key])
