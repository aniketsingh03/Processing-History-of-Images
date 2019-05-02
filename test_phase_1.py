import torch.nn as nn
from torch.autograd import Variable
from data_loader import *
from network import *
import torch
import sys

def load_checkpoints(model, PATH):
    if os.path.isfile(PATH):
        print("=> loading checkpoint '{}'".format(PATH))
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (accuracy {})"
                  .format(PATH, checkpoint['accuracy']))
    else:
        print("=> no checkpoint found at '{}'".format(PATH))

    return model

#MAIN
QF = sys.argv[1]

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print (device)

#path for the best model
PATH = QF+'/best_model_phase_1.pth'

net_phase_1 = Net()
net_phase_1 = load_checkpoints(net_phase_1, PATH)
#move model to cuda
net_phase_1 = net_phase_1.to(device)
#net_phase_1.eval()

transformations = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
batch_size = 40

Ctest_dataset = Dataset(get_Ctest(QF) ,transform=transformations)
Ctest_loader = DataLoader(Ctest_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)

correct = 0
total = 0
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
    for inp, lab in Ctest_loader:
        lab = lab.flatten()
        inputs = inp.cuda(device)
        labels = lab.cuda(device)

        inputs, labels = Variable(inputs), Variable(labels)
        #print ("INPUTS ARE ", inputs)
        #print ("LABELS ARE ", labels)
        outputs = net_phase_1(inputs)
        #print ("OUTPUTS ARE ", outputs)
        
        #print ("------------------TESTING OUTPUTS------------------", outputs.size())
        
        _, predicted = torch.max(outputs.data, 1)
        
        #print ("-----------------PREDICTED SIZE-------------------", predicted.size())
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

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


    print ("correct values ", correct)
    print ("total values ", total)
    print('Accuracy of the network in the first phase is : %d %%' % (
        100 * correct / total))

    print ("Accuracy for original images is {:.2f}".format(100 * org_correct / org_total))
    print ("Accuracy for high pass filtering is {:.2f}".format(100 * high_correct / high_total))
    print ("Accuracy for low pass filtering is {:.2f}".format(100 * low_correct / low_total))
    print ("Accuracy for tonal adjustment is {:.2f}".format(100 * tonal_correct / tonal_total))
    print ("Accuracy for denoising operation is {:.2f}".format(100 * denoise_correct /denoise_total))

    torch.cuda.empty_cache()
