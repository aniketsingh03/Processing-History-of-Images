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

    transformations = torchvision.transforms.Compose([torchvision.transforms.RandomRotation((90,90)), torchvision.transforms.ToTensor()])
    
    #generate datasets by using a wrapper class
    
    #TODO currently the images in the datasets are of dimensions (batch_size x (3x512x512)) whereas in the paper
    #it's mentioned as (batch_size x (512x512x3)), check for correctness
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
            #print (inputs)
            #print ("THE SIZE OF INPUTS IS ", inputs.size())
            labels = labels.squeeze()
            #print ("labels are ", labels)

            #Wrap them in a Variable object
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
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                        epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))
                #Reset running loss and time
                running_loss = 0.0
                start_time = time.time()
            
        #At the end of the epoch, do a pass on the validation set
        total_val_loss = 0
        for inputs, labels in Cval_loader:
            #Wrap tensors in Variables
            labels = labels.squeeze()
            inputs, labels = Variable(inputs), Variable(labels)
            
            #Forward pass
            val_outputs = net(inputs)
            val_loss_size = loss(val_outputs, labels)
            total_val_loss += val_loss_size.item()
            
        print("Validation loss = {:.2f}".format(total_val_loss / len(Cval_loader)))
        
    print("Training for phase 1 finished, took {:.2f}s".format(time.time() - training_start_time))

def obtainDataAsTensors(im_path, im_label):
    """obtain images and labels in the form of torch tensors for phase 2 manipulation
    """
    transformations = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    img = Image.open(im_path)
    img = img.convert('RGB')
    img = transformations(img)

    label = torch.from_numpy(np.asarray(im_label).reshape([1,1]))

    return (img, label)

def extractTrainMoments(net):
    """
    Extracting moments by using the network trained in phase 1
    and the inputs as M_tr(Phase 2). 4096 moments are extracted
    for each image.
    """

    #TODO currently the images in the datasets are of dimensions (batch_size x (3x1000x1000)) whereas in the paper
    #it's mentioned as (batch_size x (1000x1000x3)), check for correctness
    Mtr_dataset = get_Mtr()
    output_image = []
    output_labels = []
    attempts = 0
    #print ("SET TO PASS: ", Mtr_dataset)
    image_paths = Mtr_dataset[0]
    #print ("IMAGE PATHS SIZE: ", len(image_paths))
    image_labels = Mtr_dataset[1]
    #print ("IMAGE LABELS SIZE: ", len(image_labels))

    for i in range(len(image_labels)):
        attempts+=1
        print("Attempt number ", attempts)
        
        image, label = obtainDataAsTensors(image_paths[i], image_labels[i])
        image = image.unsqueeze(0)
        #print ("size of image is ", image.size())
        #Wrap them in a Variable object
        img = Variable(image)
        
        #Forward pass to extract moments for phase 2(this will be done one image at a time)
        single_moment = net.forward(img, phase = 1)
        output_image.append(single_moment)
        output_labels.append(label)
    
    print ("-----------------FINISHED EXTRACTING MOMENTS FOR TRAIN SET--------------------------")
    return (torch.tensor(output_image), torch.tensor(output_labels))

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
        image = image.unsqueeze(0)
        #print ("size of image is ", image.size())
        #Wrap them in a Variable object
        img = Variable(image)
        
        #Forward pass to extract moments for phase 2(this will be done one image at a time)
        single_moment = net.forward(img, phase = 1)
        output_image.append(single_moment)
        output_labels.append(label)

    print ("-----------------FINISHED EXTRACTING MOMENTS FOR VALIDATION SET--------------------------")
    return (torch.tensor(output_image), torch.tensor(output_labels))        

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
        image = image.unsqueeze(0)
        #print ("size of image is ", image.size())
        #Wrap them in a Variable object
        img = Variable(image)
        
        #Forward pass to extract moments for phase 2(this will be done one image at a time)
        single_moment = net.forward(img, phase = 1)
        output_image.append(single_moment)
        output_labels.append(label)

    print ("-----------------FINISHED EXTRACTING MOMENTS FOR TEST SET--------------------------")
    return (torch.tensor(output_image), torch.tensor(output_labels))


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
#TODO change batch_size and learning rate for testing purposes
batch_size_phase_1 = 3
learning_rate_phase_1 = 0.01

#each of M's and C's are a tuple of list of image and labels ie ([list_of_images], [list_of_labels])
#All the C denominations are (512x512) and all the M denominations are of arbitrary size
#C_tr and M_tr consist of multiple mini batches

#PHASE 1
net_phase_1 = Net()
#TODO change n_epochs to larger value later on
trainNet(net_phase_1, batch_size=batch_size_phase_1, n_epochs=3, learning_rate=learning_rate_phase_1)

#PHASE 2
#each of the extracted moments are a tuple of (moments, labels)
extracted_moments_Mtr = extractTrainMoments(net_phase_1)
extracted_moments_Mval = extractValMoments(net_phase_1)
extracted_moments_Mtest  = extractTestMoments(net_phase_1)

#PHASE 3
#TODO change batch_size and learning rate for testing purposes
batch_size_phase_3 = 1000
learning_rate_phase_3 = 0.01
net_phase_2 = MLPNet()
train_MLP_net(net_phase_2, batch_size_phase_3, 5, learning_rate_phase_3, 
extracted_moments_Mtr, extracted_moments_Mval, extracted_moments_Mtest)
