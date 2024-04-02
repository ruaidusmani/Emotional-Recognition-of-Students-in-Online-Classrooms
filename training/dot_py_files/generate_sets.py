# %%

import numpy as np
import cv2
import os
import sys
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
import torchvision.transforms as transforms
import torchvision.datasets as datasets



# %%

#Takes images in concat_data and flatten them into a 1D array
def flatten_images():
    images_dict = {'focused': [],
                    'happy': [],
                    'neutral': [],
                    'surprised': []}
    image_name_dict = {'focused': [],
                    'happy': [],
                    'neutral': [],
                    'surprised': []}
    
    for category in ['focused', 'happy', 'neutral', 'surprised']:
        #Get the list of images in the directory
        images = os.listdir("../normalized-greyscale/%s" % category)
        #Create an empty list to store the flattened images
        flattened_images = []
        #Iterate through the images
        
        for image in images:
            #check extension of the image
            if image.split(".")[-1] == "jpg":
                #Read the image
                img = cv2.imread("../normalized-greyscale/%s/%s" %(category, image), 0)
                #Flatten the image
                # img = img.flatten()
                #Add the flattened image to the list
                images_dict[category].append(img)
                image_name_dict[category].append(category + "_" + image)


    #Return the list of flattened images
    return images_dict, image_name_dict

# %%
# augmentation functions

def flip(image):
    return cv2.flip(image, 1) # flipping around y-axis

def rotate(image):
    angle = np.random.uniform(-5,5)
    M = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1)
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

def brighten(image):
    grey = False
    if len(image.shape) == 2:
        grey = True
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    factor = np.random.uniform(0.5, 1.5)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = np.clip(hsv[:,:,2] * factor, 0, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) 
    return bgr if not grey else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def change_contrast(image):
    factor = np.random.uniform(0.5, 1.5)
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)

# %%
def augment(images, labels):
    augmented_images = []
    augmented_labels = []

    for i in range(len(images)):
        image = images[i]
        label = labels[i]

        # save original image
        augmented_images.append(image)
        augmented_labels.append(label)

        # flipping
        augmented_images.append(flip(image))
        augmented_labels.append(label)

        # rotating
        augmented_images.append(rotate(image))
        augmented_labels.append(label)

        # brightening
        augmented_images.append(brighten(image))
        augmented_labels.append(label)

        # changing contrast
        augmented_images.append(change_contrast(image))
        augmented_labels.append(label)

    return augmented_images, augmented_labels

# %%
# add images of each category into an array


images_dict, image_name_dict = flatten_images() 

# tokensize labels
labels = {'focused': 0,
          'happy': 1,
          'neutral': 2,
          'surprised': 3}


# concatencate all the data with respective labels
x = []
y = []
for key in images_dict:
    for image in images_dict[key]:
        x.append(image)
        y.append(labels[key])
x_name = []
y_name = []
for key in image_name_dict:
    for image_name in image_name_dict[key]:
        x_name.append(image_name)
        y_name.append(labels[key])
        
# split into training and valid/testing
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size =0.30, random_state=42, stratify=y)
x_train_name, x_temp_name, y_train_name, y_temp_name = train_test_split(x_name, y_name, test_size =0.30, random_state=42, stratify=y_name)

# split x_temp and y_temp into validation and testing\
x_valid, x_test, y_valid, y_test = train_test_split(x_temp, y_temp, test_size =0.50, random_state=42, stratify=y_temp)
x_valid_name, x_test_name, y_valid_name, y_test_name = train_test_split(x_temp_name, y_temp_name, test_size =0.50, random_state=42, stratify=y_temp_name)

# augment the data
x_train, y_train = augment(x_train, y_train)
x_valid, y_valid = augment(x_valid, y_valid)
x_test, y_test = augment(x_test, y_test)


# %%
# Hyperparameters and settings
batch_size = 64
test_batch_size = 64
input_size = 1 # because there is only one channel 
output_size = 4
num_epochs = 10
learning_rate = 0.001



# %%
def find_image(arr_in):
    for directory in ['focused', 'happy', 'neutral', 'surprised']:
                images = os.listdir("../concat_data/%s" % directory)
                #Iterate through the images
                for image in images:
                    #check extension of the image
                    if image.split(".")[-1] == "jpg":
                        #Read the image
                        img = cv2.imread("../concat_data/%s/%s" %(directory, image), 0)
                        #Flatten the image
                        # img = img.flatten()
                        #Add the flattened image to the list
                        if np.array_equal(img, arr_in):
                            # 
                            print(image)
                            print(directory)
                            return image, directory
                            
         
    return None
def find_respective_images(tensor_in):
    # finds the image in the one of the directories
    # tensor_in: tensor
    # return: string
    dict_out = {'focused': [], 'happy': [], 'neutral': [], 'surprised': []}

    #Get the list of images in the directory
    for element in tensor_in:
        for element_name in element:
            numpy_tensor = element_name.numpy()
            image, directory = find_image(numpy_tensor)
            dict_out[directory].append(image)

            
    
    # print(tensor_in.numpy())

# %%

#Create data that can be fed into pytorch

#getting device type
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#training
images_tensor = torch.tensor(x_train, dtype=torch.float32, device=device)
images_tensor = images_tensor.unsqueeze(1)
labels_tensor = torch.tensor(y_train, dtype=torch.long, device=device)

# validation
images_valid_tensor = torch.tensor(x_valid, dtype=torch.float32, device=device)
images_valid_tensor = images_valid_tensor.unsqueeze(1)
labels_valid_tensor = torch.tensor(y_valid, dtype=torch.long, device=device)

# testing
images_testing_tensor = torch.tensor(x_test, dtype=torch.float32, device=device)
images_testing_tensor = images_testing_tensor.unsqueeze(1)
labels_testing_tensor = torch.tensor(y_test, dtype=torch.long, device=device)

batch_size = 2048

# Create a TensorDataset
# training
dataset_train = td.TensorDataset(images_tensor, labels_tensor)
data_loader = td.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)


# validation
dataset_valid = td.TensorDataset(images_valid_tensor, labels_valid_tensor)
valid_loader = td.DataLoader(dataset_valid, batch_size=batch_size, shuffle=True)


# testing
dataset_test = td.TensorDataset(images_testing_tensor, labels_testing_tensor)
test_loader = td.DataLoader(dataset_test, batch_size=batch_size, shuffle=True)


# # save each dataset
# torch.save(dataset_train, 'dataset_train.pt')
# torch.save(dataset_valid, 'dataset_valid.pt')
# torch.save(dataset_test, 'dataset_test.pt')

#save loaders
torch.save(data_loader, 'data_loader.pt')
torch.save(valid_loader, 'valid_loader.pt')
torch.save(test_loader, 'test_loader.pt')


# %%
def verify_unique_sets():
    for name in x_test_name:
        if (name in x_valid_name) or (name in x_train_name):
            print(name)
            print("error")
    for name in x_valid_name:
        if (name in x_test_name) or (name in x_train_name):
            print(name)
            print("error")
    for name in x_train_name:
        if (name in x_valid_name) or (name in x_test_name):
            print(name)
            print("error")


