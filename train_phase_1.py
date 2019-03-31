import torch.optim as optim
import torch.nn as nn
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from torch.autograd import Variable
from data_loader import *
from network import *
import numpy as np
import time
import torchvision
import torch
import sys

def createLoss(net):
    """create loss for the CNN
    """
    loss = nn.CrossEntropyLoss()
    
    return loss

def createOptimizer(net, learning_rate=0.001):
    """create optimizer for the CNN
    """
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)

    return optimizer

def load_checkpoints(model, optimizer, PATH):
    """load existing model pretrained to some epochs
    """
    start_epoch = 0
    if os.path.isfile(PATH):
        print("=> loading checkpoint '{}'".format(PATH))
        checkpoint = torch.load(PATH)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(PATH, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(PATH))

    return model, optimizer, start_epoch

def getBestModelAccuracy(model, optimizer, PATH):
    """load model with highest accuracy
    """
    accuracy = 0
    if os.path.isfile(PATH):
        print("=> loading best model accuracy'{}'".format(PATH))
        best_model = torch.load(PATH)
        accuracy = best_model['accuracy']
        #model.load_state_dict(best_model['state_dict'])
        #optimizer.load_state_dict(best_model['optimizer'])
        print("=> loaded best model '{}' with accuracy {}"
                  .format(PATH, accuracy))
    else:
        print("=> no best model found at '{}'".format(PATH))

    return accuracy

def trainNet(net, batch_size, optimizer, start_epoch, n_epochs, learning_rate):
    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS FOR PHASE 1 =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)

    transformations = torchvision.transforms.Compose([torchvision.transforms.RandomRotation((90,90)), torchvision.transforms.ToTensor()])
    
    #generate datasets by using a wrapper class
    
    #TODO currently the images in the datasets are of dimensions (batch_size x (3x512x512)) whereas in the paper
    #it's mentioned as (batch_size x (512x512x3)), check for correctness
    Ctr_dataset = Dataset(get_Ctr() ,transform=transformations)
    Ctr_loader = DataLoader(Ctr_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)

    Cval_dataset = Dataset(get_Cval() ,transform=transformations)
    Cval_loader = DataLoader(Cval_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)

    #Create our loss and optimizer functions
    loss = createLoss(net)

    training_start_time = time.time()

    n_batches = len(Ctr_loader)
    #print ("The number of batches are ", n_batches)

    #Train the moment generator part with C_tr (phase 1)
    #Loop for n_epochs
    for epoch in range(start_epoch, n_epochs):
        running_loss = 0.0
        print_every = n_batches // 100
        print ("PRINT AFTER EVERY {} batches ".format(print_every))
        start_time = time.time()
        total_train_loss = 0

        for i, data in enumerate(Ctr_loader, 0):
            #data represents a single mini-batch
            #Get inputs
            inp, lab = data
            #print ("before", labels.size())
            lab = lab.flatten()
            
            #print (inp)
            #print ("labels are ", lab)
            #print ("THE SIZE OF INPUTS IS ", inp.size())
            #print ("THE SIZE OF LABELS IS ", lab.size())
            
            #Wrap them in a Variable object
            inputs = inp.cuda(device)
            labels = lab.cuda(device)
            inputs, labels = Variable(inputs), Variable(labels)
            #print("INPUTS ARE ", inputs)
            #print("LABELS ARE ", labels)
            
            #Set the parameter gradients to zero
            optimizer.zero_grad()
            
            #Forward pass, backward pass, optimize for phase 1
            outputs = net.forward(inputs)
            #print("OUTPUTS ARE ", outputs)
            #print ("size of outputs after forward propagation is", outputs.size())
            #print ("size of labels is ", labels.size())
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()
            
            #print ("Loss for batch {} is {:f}".format(i, loss_size.item()))
            #Print statistics
            running_loss += loss_size.item()
            total_train_loss += loss_size.item()
            
            #Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                print ("BATCH NUMBER ", i)
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                        epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))
                #Reset running loss and time
                running_loss = 0.0
                start_time = time.time()
                torch.cuda.empty_cache()
        
        #save every epoch
        state = { 'epoch': epoch + 1, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(), }
        torch.save(state, PATH)                

        torch.cuda.empty_cache()
        
        #At the end of the epoch, do a pass on the validation set
        total_val_loss = 0
        for inp, lab in Cval_loader:
            lab = lab.flatten()
            #print ("-------------------INPUTS SIZE-----------------", inp.size())
            #print ("-------------------LABELS SIZE-----------------", lab.size())
            #Wrap tensors in Variables
            inputs = inp.cuda(device)
            labels = lab.cuda(device)
            inputs, labels = Variable(inputs), Variable(labels)
            
            #Forward pass
            val_outputs = net(inputs)
            #print("-------------------OUTPUT------------------", val_outputs)
            val_loss_size = loss(val_outputs, labels)
            total_val_loss += val_loss_size.item()
            
        print("Validation loss = {:.2f}".format(total_val_loss / len(Cval_loader)))
        torch.cuda.empty_cache()
        
        #Extracting accuracy of best model till now
        dummy_model = Net()
        dummy_optimizer = createOptimizer(dummy_model, learning_rate)
        previous_best_accuracy = getBestModelAccuracy(dummy_model, dummy_optimizer, BEST_MODEL_PATH)
        
        test_transformations = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])

        Ctest_dataset = Dataset(get_Ctest() ,transform=test_transformations)
        Ctest_loader = DataLoader(Ctest_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)
        #TESTING
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
                outputs = net(inputs)
                #print ("OUTPUTS ARE ", outputs)
                
                #print ("------------------TESTING OUTPUTS------------------", outputs.size())
                
                _, predicted = torch.max(outputs.data, 1)
                
                #print ("-----------------PREDICTED SIZE-------------------", predicted.size())
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print ("correct values ", correct)
            print ("total values ", total)
            current_accuracy = 100 * correct / total
            print('Current accuracy for testing phase is : %d %%' % (
                current_accuracy))
            #Saving best model till now
            if current_accuracy>previous_best_accuracy:
                state = { 'accuracy': current_accuracy, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(), }
                torch.save(state, BEST_MODEL_PATH)

            torch.cuda.empty_cache()

    print("Training for phase 1 finished, took {:.2f}s".format(time.time() - training_start_time))

#MAIN
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print (device)

#path to save each training epoch
PATH = 'checkpoints.pth'
BEST_MODEL_PATH = 'best_model_phase_1.pth'

#each of M's and C's are a tuple of list of image and labels ie ([list_of_images], [list_of_labels])
#All the C denominations are (512x512) and all the M denominations are of arbitrary size
#C_tr and M_tr consist of multiple mini batches

net_phase_1 = Net()
batch_size_phase_1 = 40
learning_rate_phase_1 = 0.01
optimizer = createOptimizer(net_phase_1, learning_rate_phase_1)

net_phase_1, optimizer, start_epoch = load_checkpoints(net_phase_1, optimizer, PATH)

#move model and optimizer to cuda
net_phase_1 = net_phase_1.to(device)
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device)

#TODO change n_epochs to larger value later on
trainNet(net_phase_1, batch_size=batch_size_phase_1, optimizer=optimizer, start_epoch=start_epoch, n_epochs=100000, learning_rate=learning_rate_phase_1)
