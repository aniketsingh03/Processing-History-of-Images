import torch.optim as optim
import torch.nn as nn
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from data_loader import *
from network import *
import numpy as np
import time

def createLossAndOptimizer(net, learning_rate=0.001):
    """create loss and optimizer for the CNN
    """
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    return (loss, optimizer)

def trainNet(net, batch_size, n_epochs, learning_rate):
    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS FOR PHASE 1 =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)

    transformations = transforms.Compose([transforms.RandomRotation(90),transforms.ToTensor()])
    
    #generate datasets by using a wrapper class
    Ctr_dataset = Dataset(get_Ctr() ,transform=transformations)
    Ctr_loader = DataLoader(Ctr_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)

    Cval_dataset = Dataset(get_Cval() ,transform=transformations)
    Cval_loader = DataLoader(Cval_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)

    Ctest_dataset = Dataset(get_Ctest() ,transform=transformations)
    Ctest_loader = DataLoader(Ctest_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)

    #Create our loss and optimizer functions
    loss, optimizer = createLossAndOptimizer(net, learning_rate)
    #Time for printing
    training_start_time = time.time()
    
    n_batches = len(Ctr_loader)
    #Train the moment generator part with C_tr (phase 1)
    #Loop for n_epochs
    for epoch in range(n_epochs):
        running_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0

        for i, data in enumerate(Ctr_loader, 0):
            #data represents a single mini-batch
            #Get inputs
            inputs, labels = data

            #Wrap them in a Variable object
            inputs, labels = Variable(inputs), Variable(labels)
            
            #Set the parameter gradients to zero
            optimizer.zero_grad()
            
            #Forward pass, backward pass, optimize for phase 1
            outputs = net.forward(inputs)
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()
            
            #Print statistics
            running_loss += loss_size.data[0]
            total_train_loss += loss_size.data[0]
            
            #Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                        epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))
                #Reset running loss and time
                running_loss = 0.0
                start_time = time.time()
            
        #At the end of the epoch, do a pass on the validation set
        total_val_loss = 0
        for inputs, labels in Cval_loader:
            #Wrap tensors in Variables
            inputs, labels = Variable(inputs), Variable(labels)
            
            #Forward pass
            val_outputs = net(inputs)
            val_loss_size = loss(val_outputs, labels)
            total_val_loss += val_loss_size.data[0]
            
        print("Validation loss = {:.2f}".format(total_val_loss / len(Cval_loader)))
        
    print("Training for phase 1 finished, took {:.2f}s".format(time.time() - training_start_time))

def extractMoments(net):
    """
    Extracting moments by using the network trained in phase 1
    and the inputs as M_tr(Phase 2). 4096 moments are extracted
    for each image.
    """
    #for training set
    transformations = transforms.Compose([transforms.ToTensor()])
    
    #generate datasets by using a wrapper class
    Mtr_dataset = Dataset(get_Mtr() ,transform=transformations)
    Mtr_loader = DataLoader(Mtr_dataset, batch_size = len(Mtr_dataset), shuffle = False, num_workers = 4)

    Mval_dataset = Dataset(get_Mval() ,transform=transformations)
    Mval_loader = DataLoader(Mval_dataset, batch_size = len(Mval_dataset), shuffle = False, num_workers = 4)

    Mtest_dataset = Dataset(get_Mtest() ,transform=transformations)
    Mtest_loader = DataLoader(Mtest_dataset, batch_size = len(Mtest_dataset), shuffle = False, num_workers = 4)

    M_tr_final_results = [] 
    M_val_final_results = []
    M_test_final_results = []
    set_to_pass = []

    for i in range(3):
        output = []
        if (i==0):
            set_to_pass = Mtr_loader
        elif (i==1):
            set_to_pass = Mval_loader
        else:
            set_to_pass = Mtest_loader
        
        inputs = []
        labels = []
        for i,(a,b) in enumerate(set_to_pass, 0):
            inputs = a
            labels = b

        for im in inputs:
            #Wrap them in a Variable object
            img = Variable(im)
            #Forward pass to extract moments for phase 2(this will be done one image at a time)
            single_moment = net.forward(img, phase = 1)
            output.append(single_moment)

        if (i==0):
            M_tr_final_results = (torch.tensor(output), labels)
        elif (i==1):
            M_val_final_results = (torch.tensor(output), labels)
        else:
            M_test_final_results = (torch.tensor(output), labels)

    return (M_tr_final_results, M_val_final_results, M_test_final_results)

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

    train_dataset = MLPDataset(M_tr)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)
    
    val_dataset = MLPDataset(M_val)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)
    
    test_dataset = MLPDataset(M_test)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)
    
    #Create our loss and optimizer functions
    loss, optimizer = createLossAndOptimizer(net, learning_rate)
    #Time for printing
    training_start_time = time.time()
    
    n_batches = len(train_dataset)/batch_size

    #Train the moment generator part with C_tr (phase 1)
    #Loop for n_epochs
    for epoch in range(n_epochs):
        running_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0

        for i, data in enumerate(train_loader, 0):
            #data represents a single mini-batch
            #Get inputs
            inputs, labels = data

            #Wrap them in a Variable object
            inputs, labels = Variable(inputs), Variable(labels)
            
            #Set the parameter gradients to zero
            optimizer.zero_grad()
            
            #Forward pass, backward pass, optimize for phase 1
            outputs = net.forward(inputs)
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()
            
            #Print statistics
            running_loss += loss_size.data[0]
            total_train_loss += loss_size.data[0]
            
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
            #Wrap tensors in Variables
            inputs, labels = Variable(inputs), Variable(labels)
            
            #Forward pass
            val_outputs = net(inputs)
            val_loss_size = loss(val_outputs, labels)
            total_val_loss += val_loss_size.data[0]
            
        print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))
        
    print("Training for phase 3 finished, took {:.2f}s".format(time.time() - training_start_time))

#MAIN
#Get training data from the data_loader class
batch_size_phase_1 = 40
learning_rate_phase_1 = 0.01

#each of M's and C's are a tuple of list of image and labels ie ([list_of_images], [list_of_labels])
#All the C denominations are (512x512) and all the M denominations are of arbitrary size
#C_tr and M_tr consist of multiple mini batches

#PHASE 1
net_phase_1 = Net()
trainNet(net_phase_1, batch_size=batch_size_phase_1, n_epochs=5, learning_rate=learning_rate_phase_1)

#PHASE 2
#each of the extracted moments are a tuple of (moments, labels)
extracted_moments_Mtr, extracted_moments_Mval, extracted_moments_Mtest  = extractMoments(net_phase_1)

#PHASE 3
batch_size_phase_3 = 1000
learning_rate_phase_3 = 0.01
net_phase_2 = MLPNet()
train_MLP_net(net_phase_2, batch_size_phase_3, 5, learning_rate_phase_3, 
extracted_moments_Mtr, extracted_moments_Mval, extracted_moments_Mtest)
