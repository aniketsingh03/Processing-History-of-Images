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
    optimizer = optim.Adam([{'params': net.parameters()}], lr=learning_rate)

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
    print ("The number of batches are ", n_batches)

    #Train the moment generator part with C_tr (phase 1)
    #Loop for n_epochs
    for epoch in range(start_epoch, n_epochs):
        running_loss = 0.0
        print_every = n_batches // 100
        print ("PRINT EVERY IS ", print_every)
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
            
            #Set the parameter gradients to zero
            optimizer.zero_grad()
            
            #Forward pass, backward pass, optimize for phase 1
            outputs = net.forward(inputs)
            #print ("size of outputs after forward propagation is", outputs.size())
            #print ("size of labels is ", labels.size())
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()
            
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
            #Wrap tensors in Variables
            lab = lab.flatten()
            #print ("-------------------INPUTS SIZE-----------------", inp.size())
            #print ("-------------------LABELS SIZE-----------------", lab.size())
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

    print("Training for phase 1 finished, took {:.2f}s".format(time.time() - training_start_time))

def extractTrainMoments(net):
    #TODO add conversion to tensor part as below while writing training phase 3 part
    
    num_samples = len(output_labels)
    #print ("-----------------SIZE OF OUTPUT IMAGE-----------------", len(output_image))
    #print ("-----------------SIZE OF OUTPUT LABELS-----------------", len(output_labels))
    ret = (torch.cat(output_image, dim=0).view(num_samples, -1).to(device), torch.cat(output_labels, dim=0).to(device))
    #print ("-----------FINAL IMAGE OUTPUT SIZE----------", ret[0].size())
    #print ("-----------FINAL LABEL OUTPUT SIZE----------", ret[1].size())

    print ("-----------------FINISHED EXTRACTING MOMENTS FOR TRAIN SET--------------------------")
    return ret

def extractValMoments(net):
    """
    Extracting moments by using the network trained in phase 1
    and the inputs as M_val(Phase 2). 4096 moments are extracted
    for each image.
    """

    #TODO currently the images in the datasets are of dimensions (batch_size x (3x1000x1000)) whereas in the paper
    #it's mentioned as (batch_size x (1000x1000x3)), check for correctness
    Mval_dataset = get_Mval()
    output_image = []
    output_labels = []
    attempts = 0
    #print ("SET TO PASS: ", Mval_dataset)
    image_paths = Mval_dataset[0]
    #print ("IMAGE PATHS SIZE: ", len(image_paths))
    image_labels = Mval_dataset[1]
    #print ("IMAGE LABELS SIZE: ", len(image_labels))

    for i in range(len(image_labels)):
        attempts+=1
        print("Attempt number ", attempts)
        
        image, label = obtainDataAsTensors(image_paths[i], image_labels[i])
        #print ("SIZE OF IMAGE: ", image.size())
        #print ("SIZE OF LABEL: ", label.size())
        image = image.unsqueeze(0)
        #print ("size of image is ", image.size())
        #Wrap them in a Variable object
        img = Variable(image.to(device))
        
        #Forward pass to extract moments for phase 2(this will be done one image at a time)
        single_moment = net.forward(img, phase = 1)
        #print ("-------------SIZE OF SINGLE MOMENT-----------", single_moment.size())
        output_image.append(single_moment.data[0])
        output_labels.append(label)

    num_samples = len(output_labels)
    #print ("-----------------SIZE OF OUTPUT IMAGE-----------------", len(output_image))
    #print ("-----------------SIZE OF OUTPUT LABELS-----------------", len(output_labels))
    ret = (torch.cat(output_image, dim=0).view(num_samples, -1).to(device), torch.cat(output_labels, dim=0).to(device))
    #print ("-----------FINAL IMAGE OUTPUT SIZE----------", ret[0].size())
    #print ("-----------FINAL LABEL OUTPUT SIZE----------", ret[1].size())

    print ("-----------------FINISHED EXTRACTING MOMENTS FOR VALIDATION SET--------------------------")
    return ret

def extractTestMoments(net):
    """
    Extracting moments by using the network trained in phase 1
    and the inputs as M_test(Phase 2). 4096 moments are extracted
    for each image.
    """

    #TODO currently the images in the datasets are of dimensions (batch_size x (3x1000x1000)) whereas in the paper
    #it's mentioned as (batch_size x (1000x1000x3)), check for correctness
    Mtest_dataset = get_Mtest()
    attempts = 0
    output_image = []
    output_labels = []
    #print ("SET TO PASS: ", Mtest_dataset)
    image_paths = Mtest_dataset[0]
    #print ("IMAGE PATHS SIZE: ", len(image_paths))
    image_labels = Mtest_dataset[1]
    #print ("IMAGE LABELS SIZE: ", len(image_labels))
    
    for i in range(len(image_labels)):
        attempts+=1
        print("Attempt number ", attempts)
        
        image, label = obtainDataAsTensors(image_paths[i], image_labels[i])
        #print ("SIZE OF IMAGE: ", image.size())
        #print ("SIZE OF LABEL: ", label.size())
        image = image.unsqueeze(0)
        #print ("size of image is ", image.size())
        #Wrap them in a Variable object
        img = Variable(image.to(device))
        
        #Forward pass to extract moments for phase 2(this will be done one image at a time)
        single_moment = net.forward(img, phase = 1)
        #print ("-------------SIZE OF SINGLE MOMENT-----------", single_moment.size())
        output_image.append(single_moment.data[0])
        output_labels.append(label)

    num_samples = len(output_labels)
    #print ("-----------------SIZE OF OUTPUT IMAGE-----------------", len(output_image))
    #print ("-----------------SIZE OF OUTPUT LABELS-----------------", len(output_labels))
    ret = (torch.cat(output_image, dim=0).view(num_samples, -1).to(device), torch.cat(output_labels, dim=0).to(device))
    #print ("-----------FINAL IMAGE OUTPUT SIZE----------", ret[0].size())
    #print ("-----------FINAL LABEL OUTPUT SIZE----------", ret[1].size())

    print ("-----------------FINISHED EXTRACTING MOMENTS FOR TEST SET--------------------------")
    return ret


#TODO resolve issues related to loss and labels here also
def train_MLP_net(net, batch_size, n_epochs, learning_rate, M_tr, M_val, M_test):
    """train the MLP with extracted moments in phase 2
    x:(Nx4096) matrix where N->number of images of random size
    """
    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS FOR PHASE 3 =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)

    #DEBUG STATEMENTS
    print ("TRAINING SET MOMENTS SIZE: ", M_tr[0].size())
    print ("TRAINING SET LABELS SIZE: ", M_tr[1].size())
    
    train_dataset = MLPDataset(M_tr)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 0)
    
    val_dataset = MLPDataset(M_val)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True, num_workers = 0)
    
    test_dataset = MLPDataset(M_test)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True, num_workers = 0)
    
    #Create our loss and optimizer functions
    loss, optimizer = createLossAndOptimizer(net, learning_rate)
    #Time for printing
    training_start_time = time.time()
    
    n_batches = len(train_loader)
    #Train the moment generator part with C_tr (phase 1)
    #Loop for n_epochs
    for epoch in range(n_epochs):
        running_loss = 0.0
        print_every = n_batches // 100
        start_time = time.time()
        total_train_loss = 0
        #print ("size of data loader is ", len(train_loader))
        for i, data in enumerate(train_loader, 0):
            #data represents a single mini-batch
            #Get inputs
            inputs, labels = data
            
            #print ("labels before flattening ", labels.size())
            labels = labels.flatten()

            #Wrap them in a cudaVariable object
            inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
            
            #Set the parameter gradients to zero
            optimizer.zero_grad()
            
            #Forward pass, backward pass, optimize for phase 1
            outputs = net.forward(inputs)
            loss_size = loss(outputs, labels)
            #print ("-----------------LOSS SIZE-----------------", loss_size)
            loss_size.backward()
            optimizer.step()
            
            #Print statistics
            running_loss += loss_size.item()
            total_train_loss += loss_size.item()
            
            #Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                        epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))
                #Reset running loss and time
                running_loss = 0.0
                start_time = time.time()
            
        #At the end of the epoch, do a pass on the validation set
        total_val_loss = 0
        for inputs, labels in val_loader:
            labels = labels.flatten()
            #DEBUG STATEMENTS
            #print ("-------------------INPUTS SIZE-----------------", inputs.size())
            #print ("-------------------LABELS SIZE-----------------", labels.size())
            #Wrap tensors in Variables
            inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
            
            #Forward pass
            val_outputs = net(inputs)
            #print ("-------------------OUTPUTS-----------------", val_outputs)
            val_loss_size = loss(val_outputs, labels)
            total_val_loss += val_loss_size.item()
            
        print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))
    
    #TESTING
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            labels = labels.flatten()
            inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
            outputs = net(inputs)
            #print ("------------------TESTING OUTPUTS------------------", outputs.size())

            _, predicted = torch.max(outputs.data, 1)
            #print ("-----------------PREDICTED SIZE-------------------", predicted.size())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network in the third phase is : %d %%' % (
        100 * correct / total))

    print("Training for phase 3 finished, took {:.2f}s".format(time.time() - training_start_time))

#MAIN
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print (device)

#path to save each training epoch
PATH = 'checkpoints.pth'

#each of M's and C's are a tuple of list of image and labels ie ([list_of_images], [list_of_labels])
#All the C denominations are (512x512) and all the M denominations are of arbitrary size
#C_tr and M_tr consist of multiple mini batches

net_phase_1 = Net()
batch_size_phase_1 = 40
learning_rate_phase_1 = 0.001
optimizer = createOptimizer(net_phase_1, learning_rate_phase_1)

net_phase_1, optimizer, start_epoch = load_checkpoints(net_phase_1, optimizer, PATH)

#move model and optimizer to cuda
net_phase_1 = net_phase_1.to(device)
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device)

#TODO change n_epochs to larger value later on

trainNet(net_phase_1, batch_size=batch_size_phase_1, optimizer=optimizer, start_epoch=start_epoch, n_epochs=3, learning_rate=learning_rate_phase_1)

#PHASE 2
#each of the extracted moments are a tuple of (moments, labels)
extracted_moments_Mtr = extractTrainMoments(net_phase_1)
extracted_moments_Mval = extractValMoments(net_phase_1)
extracted_moments_Mtest  = extractTestMoments(net_phase_1)

#PHASE 3
#TODO change batch_size and learning rate for testing purposes
batch_size_phase_3 = 100
learning_rate_phase_3 = 0.0001
net_phase_2 = MLPNet().cuda(device)
train_MLP_net(net_phase_2, batch_size=batch_size_phase_3, n_epochs=20, learning_rate=learning_rate_phase_3, 
M_tr=extracted_moments_Mtr,M_val=extracted_moments_Mval,M_test=extracted_moments_Mtest)
