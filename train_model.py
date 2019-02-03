import torch.optim as optim
import torch.nn as nn
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from data_loader import *
from network import *

def createLossAndOptimizer(net, learning_rate=0.001):
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    
    return (loss, optimizer)

def trainNet(net, batch_size, n_epochs, learning_rate):  
    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)
    
    #Get training data from the data_loader class
    train_loader = get_data(batch_size)
    
    #each of M's and C's are a tuple of list of image and labels ie ([list_of_images], [list_of_labels])
    #All the C denominations are (512x512) and all the M denominations are of arbitrary size
    #C_tr and M_tr consist of multiple mini batches
    M_tr = train_loader[0]
    C_tr = train_loader[1]
    
    M_val = train_loader[2] 
    C_val = train_loader[3] 
    
    M_test = train_loader[4] 
    C_test = train_loader[5]
    
    n_batches = len(C_tr)
    
    #Create our loss and optimizer functions
    loss, optimizer = createLossAndOptimizer(net, learning_rate)
    
    #Time for printing
    training_start_time = time.time()
    
    #TRAIN THE MOMENT GENERATOR PART WITH C_tr (phase 1)
    #Loop for n_epochs
    for epoch in range(n_epochs):
        running_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0
        
        for i, data in enumerate(C_tr, 0):
            #data represents a single mini-batch
            #Get inputs
            inputs, labels = data
            
            #Wrap them in a Variable object
            inputs, labels = Variable(inputs), Variable(labels)
            
            #Set the parameter gradients to zero
            optimizer.zero_grad()
            
            #Forward pass, backward pass, optimize
            outputs = net(inputs)
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
        for inputs, labels in C_val:
            
            #Wrap tensors in Variables
            inputs, labels = Variable(inputs), Variable(labels)
            
            #Forward pass
            val_outputs = net(inputs)
            val_loss_size = loss(val_outputs, labels)
            total_val_loss += val_loss_size.data[0]
            
        print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))
        
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
