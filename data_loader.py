import os
import random
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

# Dataset Parameters
DATASET_PATH = 'dataset/jpegs'

LABEL = {
    'org' : 0,
    'high' : 1,
    'low' : 2,
    'tonal' : 3,
    'denoise' : 4,
}

def get_Ctr():
    C_tr = read_images(os.path.join(DATASET_PATH, 'train/ctr', True)) #(512x512) images
    return C_tr

def get_Mtr():
    M_tr = read_images(os.path.join(DATASET_PATH, 'train/mtr')) #arbitrarily sized images
    return M_tr

def get_Cval():
    C_val = read_images(os.path.join(DATASET_PATH, 'val/ctr', True)) #(512x512) images
    return C_val

def get_Mval():
    M_val = read_images(os.path.join(DATASET_PATH, 'val/mtr')) #arbitrarily sized images
    return M_val

def get_Ctest():
    C_test = read_images(os.path.join(DATASET_PATH, 'test/ctr', True)) #(512x512) images
    return C_test

def get_Mtest():
    M_test = read_images(os.path.join(DATASET_PATH, 'test/mtr')) #arbitrarily sized images
    return M_test

def read_images(dataset_path, batch_size=None, shuffle=False):
    """
    Load Images stored in class-wise folders and return batchwise lists
    """
    imagepaths, labels = list(), list()

    # List the directory
    classes = sorted(os.walk(dataset_path).__next__()[1])

    # List each sub-directory (the classes)
    for c in classes:
        c_dir = os.path.join(dataset_path, c)
        walk = os.walk(c_dir).__next__()
        for sample in walk[2]:
            # Only keeps jpeg images
            if sample.endswith('.jpg') or sample.endswith('.jpeg'):
                imagepaths.append(os.path.join(c_dir, sample))
                labels.append(LABEL[c])

    # Shuffle image dataset
    c = list(zip(imagepaths, labels))
    if shuffle:
        random.shuffle(c)
    imagepaths, labels = zip(*c)

    data = [imagepaths, labels]

    # if batch_size is given return batches of data
    if batch_size:
        imagepaths = [ list(imagepaths[i:i + batch_size])
            for i in range(0, len(imagepaths), batch_size)
        ]
        labels = [ list(labels[i:i + batch_size])
            for i in range(0, len(labels), batch_size)
        ]

        data =  [list(a) for a in zip(imagepaths, labels)]

    return data

class Dataset(Dataset):
    """a class to let torch manage the entire dataset for training the CNN in phase 1  
    Arguments:
        image_data : collection of strings representing the image paths and labels tuples
        transform : the transform to be applied to the images
    """    
    def __init__(self, image_data, transform = None):
        self.image_paths, self.image_labels = image_data
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index])
        img = img.convert('RGB')
        if self.transform is not None:    
            img = self.transform(img)
        label = torch.from_numpy(np.asarray(self.image_labels[index]).reshape([1,1]))
        #print (img)
        return img, label

    def __len__(self):
        return len(self.image_paths)

class MLPDataset(Dataset):
    """a class to let torch manage the entire dataset for training MLP in phase 3  
    Arguments:
        moments : a tuple of (mom, labels) where:
            mom -> tensor of (Nx4096) consisting of moments of all images
            labels -> tensor of (Nx1) listing label of each corresponding image 
    """
    def __init__(self, moments):
        self.mom, self.labels = moments

    def __getitem__(self, index):
        return self.mom[index], self.labels[index]

    def __len__(self):
        return len(self.mom)


#TESTING PART

# transformations = transforms.Compose([transforms.RandomCrop(32),
#                                     transforms.ToTensor()])
# dataset = Dataset(get_Ctr() ,transform=transformations)
# train_loader = DataLoader(dataset, batch_size = 2, shuffle = True, num_workers = 4)
# print (len(train_loader))

# print ('length of dataset is ', len(dataset))
# inputs = []
# labels = []
# for i,(a,b) in enumerate(train_loader, 0):
#     inputs = a
#     labels = b
# print ('size of inputs is ', inputs.shape)
# for data in inputs:
#     #print ('i is ', i)
#     print ('size of data is ', data.shape)
#     print ('data is ', data)
#     #print ('label is ', label)
#     print ("="*30)
# print('All LABELS are ', labels)    

# a = torch.randn(10,4096)
# b = labels
# c = (a, b)
# dataset = MLPDataset(c)
# train_loader = DataLoader(dataset, batch_size = 2, shuffle = True, num_workers = 4)

# for inputs,labels in train_loader:
#     #inputs, labels = data
#     #print ('i=', i)
#     print('input is ', inputs)
#     print('size of input is ', inputs.shape)
#     print('label is ', labels)
#     print('size of labels is', labels.shape)
#     print('='*30)
