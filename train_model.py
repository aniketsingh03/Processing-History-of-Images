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

def get_images_from_paths(image_paths):
    loadedImages = []
    for path in image_paths:
        img = Image.open(path)
        loadedImages.append(img)
    
    return loadedImages    

def trainNet(net, batch_size, n_epochs, learning_rate, C_tr):  
    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)
    
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
            input_paths, labels = data
            inputs = get_images_from_paths(input_paths)
            
            for img in inputs:
                img.show()

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
        for input_paths, labels in C_val:
            inputs = get_images_from_paths(input_paths)
            #Wrap tensors in Variables
            inputs, labels = Variable(inputs), Variable(labels)
            
            #Forward pass
            val_outputs = net(inputs)
            val_loss_size = loss(val_outputs, labels)
            total_val_loss += val_loss_size.data[0]
            
        print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))
        
    print("Training for phase 1 finished, took {:.2f}s".format(time.time() - training_start_time))
    return net

def extractMoments(net, M_tr, M_val, M_test):
    """
    Extracting moments by using the network trained in phase 1
    and the inputs as M_tr(Phase 2). 4096 moments are extracted
    for each image.
    """
    #for training set
    M_tr_final_results = []
    outputs = []
    for i, data in enumerate(M_tr, 0):
        #data represents a single mini-batch
        #Get inputs
        input_paths, labels = data
        inputs = get_images_from_paths(input_paths)
        
        for img in inputs:
            img.show()
        #Wrap them in a Variable object
        inputs, labels = Variable(inputs), Variable(labels)
        
        #Forward pass to extract moments for phase 2(this will be done one image at a time)
        for im in inputs:
            outputs.append(net.forward(im))
        M_tr_final_results.append(outputs)
        
    #for validation set    
    M_val_final_results = []
    outputs = []
    for i, data in enumerate(M_val, 0):
        #data represents a single mini-batch
        #Get inputs
        input_paths, labels = data
        inputs = get_images_from_paths(input_paths)
        
        for img in inputs:
            img.show()
        #Wrap them in a Variable object
        inputs, labels = Variable(inputs), Variable(labels)
        
        #Forward pass to extract moments for phase 2(this will be done one image at a time)
        for im in inputs:
            outputs.append(net.forward(im))
        M_val_final_results.append(outputs)

    #for test set
    M_test_final_results = []
    outputs = []
    for i, data in enumerate(M_test, 0):
        #data represents a single mini-batch
        #Get inputs
        input_paths, labels = data
        inputs = get_images_from_paths(input_paths)
        
        for img in inputs:
            img.show()
        #Wrap them in a Variable object
        inputs, labels = Variable(inputs), Variable(labels)
        
        #Forward pass to extract moments for phase 2(this will be done one image at a time)
        for im in inputs:
            outputs.append(net.forward(im))
        M_test_final_results.append(outputs)                
    
    return (M_tr_final_results, M_val_final_results, M_test_final_results)

#Get training data from the data_loader class
train_loader = get_data(batch_size)

#each of M's and C's are a tuple of list of image and labels ie ([list_of_images], [list_of_labels])
#All the C denominations are (512x512) and all the M denominations are of arbitrary size
#C_tr and M_tr consist of multiple mini batches

M_tr = train_loader[0] #used to train MLPNet
C_tr = train_loader[1] #used to train Net

M_val = train_loader[2] 
C_val = train_loader[3] 

M_test = train_loader[4] 
C_test = train_loader[5]

#PHASE 1
net_phase_1 = Net()
trainNet(net_phase_1, batch_size=40, n_epochs=5, learning_rate=0.01, C_tr)

#PHASE 2
extracted_moments = extractMoments(net, M_tr, M_val, M_test)
