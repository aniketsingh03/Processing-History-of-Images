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
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(PATH, checkpoint['epoch']))
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
BEST_MODEL_PATH = 'checkpoints_MLP.pth'
net = MLPNet()
net = load_checkpoints(net, BEST_MODEL_PATH)

#move model to cuda
net = net.to(device)

saved_test_moments_filename = "test_moments"
test_moments, test_labels = loadSavedMoments(saved_test_moments_filename)
test_moments, test_labels = convertListsToTensors(test_moments, test_labels)
M_test = (test_moments, test_labels)

test_dataset = MLPDataset(M_test)
test_loader = DataLoader(test_dataset, batch_size = 1000, shuffle = True, num_workers = 0)

#TESTING
correct_test = 0
total_test = 0
test_accuracy = 0
org_correct = 0
org_total = 0

high_correct = 0
high_total = 0 

low_correct = 0
low_total = 0

tonal_correct = 0
tonal_total = 0

denoise_correct = 0
denoise_total = 0

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

        org_total += (labels==0).sum().item()
        high_total += (labels==1).sum().item()
        low_total += (labels==2).sum().item()
        tonal_total += (labels==3).sum().item()
        denoise_total += (labels==4).sum().item()

        org_correct += torch.min(predicted==0, labels==0).sum().item()
        high_correct += torch.min(predicted==1, labels==1).sum().item()
        low_correct += torch.min(predicted==2, labels==2).sum().item()
        tonal_correct += torch.min(predicted==3, labels==3).sum().item()
        denoise_correct += torch.min(predicted==4, labels==4).sum().item()

test_accuracy = 100 * correct_test / total_test
print("Test Accuracy of the network is : {:.3f}".format(test_accuracy))

# print ("The number of original images are ", org_total)
# print ("The number of images correctly detected are ", org_correct)
print ("Accuracy for original images is {:.2f}".format(100 * org_correct / org_total))
print ("Accuracy for high pass filtering is {:.2f}".format(100 * high_correct / high_total))
print ("Accuracy for low pass filtering is {:.2f}".format(100 * low_correct / low_total))
print ("Accuracy for tonal adjustment is {:.2f}".format(100 * tonal_correct / tonal_total))
print ("Accuracy for denoising operation is {:.2f}".format(100 * denoise_correct /denoise_total))