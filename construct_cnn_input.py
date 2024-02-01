# import pandas as pd
import matplotlib
matplotlib.use('Agg')
import numpy as np
from PIL import Image, ImageOps
# import tensorflow as tf
# from tensorflow.keras import preprocessing
import csv
import matplotlib.pyplot as plt
# import torch
# from torch.autograd import Variable
# from torchvision import models
from tqdm import tqdm
from io import BytesIO
import pickle
import gzip


file_path = "/ais/hal9000/yuzhang/construct_cnn_input/TS_Freq_array.csv"

# Initialize arrays
y_labels = []
x_TS = []

# Read the CSV file line by line
with open(file_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    
    next(csv_reader)
    
    # Extract the first column and construct rows as arrays
    for row in csv_reader:
        y_labels.append(row[0])
        x_TS.append(row[1:])


x_TS = np.array(x_TS, dtype = float)
print(x_TS.max()) #upper limit of plot
print(x_TS.min()) #lower limit of plot

print(len(x_TS))

# Create an array of x values
x_values = np.arange(45,170.5,0.5)
elements_to_remove = [90.0, 90.5]
x_values = x_values[~np.isin(x_values, elements_to_remove)]
print(x_values)


# Construct graphs and stored pixel matrices
Graph_Pixel = []

# Normalize
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

for i in tqdm(range(len(x_TS))):
    virtual_file = BytesIO()
    TS_array = np.array(x_TS[i])
    plt.plot(x_values, TS_array, linestyle='-')
    plt.xlim(45, 170)
    plt.ylim(-122, -17)
    plt.grid(True)
    plt.savefig(virtual_file, format = "png")
    plt.clf()
    plt.close()
    virtual_file.seek(0)
    image = Image.open(virtual_file)
    gray_image = image.convert('L')
    gray_image = gray_image.resize((224, 224), Image.LANCZOS)
    im_as_arr = np.float32(gray_image) # Q: will the axis labels affect the values?
    # im_as_arr = np.expand_dims(im_as_arr, axis=2)
    # im_as_arr = im_as_arr.transpose(2, 1, 0)
    # for channel, _ in enumerate(im_as_arr):
    #     im_as_arr[channel] /= 255
    #     im_as_arr[channel] -= mean[channel]
    #     im_as_arr[channel] /= std[channel]

    im_as_arr /= 255
    im_as_arr -= mean[0]
    im_as_arr /= std[0]
    Graph_Pixel.append(im_as_arr)


# np.savez('/1D CNN/Graph_Pixel.npz', *Graph_Pixel)
print(len(Graph_Pixel))

output = "/ais/hal9000/yuzhang/construct_cnn_input/graph_pixel_matrices_new.p"
# pickle.dump(Graph_Pixel, open(output, "wb"), protocol=2)


with gzip.open(output, 'wb') as file:
    pickle.dump(Graph_Pixel, file, protocol=pickle.HIGHEST_PROTOCOL)