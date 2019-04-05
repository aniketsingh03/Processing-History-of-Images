import torch.optim as optim
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from data_loader import *
from network import *
import numpy as np
import time
import torchvision
import torch
import pickle

def load_checkpoints(model, PATH):
    """load existing model pretrained to some epochs
    """
    if os.path.isfile(PATH):
        print("=> loading checkpoint '{}'".format(PATH))
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (accuracy {})"
                  .format(PATH, checkpoint['accuracy']))
    else:
        print("=> no checkpoint found at '{}'".format(PATH))

    return model

def loadSavedMoments(filename):
    """load already existing moments from disk
    """
    if os.path.isfile(filename):
        fileObject = open(filename, 'rb')
        moments_list, labels_list, _ = pickle.load(fileObject)
    else:
        moments_list, labels_list = [], []
    
    return (moments_list, labels_list)

def convertListsToTensors(moments, labels):
    num_samples = len(labels)
    ret = (torch.cat(moments, dim=0).view(num_samples, -1), torch.cat(labels, dim=0))

    return ret

#TESTING
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print (device)

#path for the best model
BEST_MODEL_PATH = 'best_model_MLP.pth'
net = MLPNet()
net = load_checkpoints(net, BEST_MODEL_PATH)

#move model to cuda
net = net.to(device)

saved_test_moments_filename = "test_moments"
test_moments, test_labels = loadSavedMoments(saved_test_moments_filename)
test_moments, test_labels = convertListsToTensors(test_moments, test_labels)
M_test = (test_moments, test_labels)

test_dataset = MLPDataset(M_test)
test_loader = DataLoader(test_dataset, batch_size = 100, shuffle = True, num_workers = 0)

#TESTING
correct_test = 0
total_test = 0
test_accuracy = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        labels = labels.flatten()
        inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
        outputs = net(inputs)
        #print ("------------------TESTING OUTPUTS------------------", outputs.size())
        _, predicted = torch.max(outputs.data, 1)
        #print ("-----------------PREDICTED SIZE-------------------", predicted.size())
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()
test_accuracy = 100 * correct_test / total_test               
print("Test Accuracy of the network is : {:.3f}".format(test_accuracy))