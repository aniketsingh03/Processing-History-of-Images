import torch.nn as nn
from torch.autograd import Variable
from data_loader import *
from network import *
import torch

def load_checkpoints(model, PATH):
    if os.path.isfile(PATH):
        print("=> loading checkpoint '{}'".format(PATH))
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(PATH, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(PATH))

    return model

#TESTING
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print (device)

#path to save each training epoch
PATH = 'checkpoints.pth'

net_phase_1 = Net()
net_phase_1 = load_checkpoints(net_phase_1, PATH)
#move model to cuda
net_phase_1 = net_phase_1.to(device)
#net_phase_1.eval()

transformations = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
batch_size = 40

Ctest_dataset = Dataset(get_Ctest() ,transform=transformations)
Ctest_loader = DataLoader(Ctest_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)

correct = 0
total = 0
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

    print ("correct values ", correct)
    print ("total values ", total)
    print('Accuracy of the network in the first phase is : %d %%' % (
        100 * correct / total))
    torch.cuda.empty_cache()