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
            #print ("before", labels.size())
            labels = labels.flatten()
            
            #print (inputs)
            #print ("labels are ", labels)
            #print ("THE SIZE OF INPUTS IS ", inputs.size())
            #print ("THE SIZE OF LABELS IS ", labels.size())
            
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
            labels = labels.flatten()
            #print ("-------------------INPUTS SIZE-----------------", inputs.size())
            #print ("-------------------LABELS SIZE-----------------", labels.size())
            inputs, labels = Variable(inputs), Variable(labels)
            
            #Forward pass
            val_outputs = net(inputs)
            #print("-------------------OUTPUT------------------", val_outputs)
            val_loss_size = loss(val_outputs, labels)
            total_val_loss += val_loss_size.item()
            
        print("Validation loss = {:.2f}".format(total_val_loss / len(Cval_loader)))
        
    print("Training for phase 1 finished, took {:.2f}s".format(time.time() - training_start_time))

    #TESTING
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in Ctest_loader:
            labels = labels.flatten()
            inputs, labels = Variable(inputs), Variable(labels)
            outputs = net(inputs)
            #print ("------------------TESTING OUTPUTS------------------", outputs.size())

            _, predicted = torch.max(outputs.data, 1)
            #print ("-----------------PREDICTED SIZE-------------------", predicted.size())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print ("correct values ", correct)
    print ("total values ", total)
    print('Accuracy of the network in the first phase is : %d %%' % (
        100 * correct / total))

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
        #print ("SIZE OF IMAGE: ", image.size())
        #print ("SIZE OF LABEL: ", label.size())
        image = image.unsqueeze(0)
        #print ("size of image is ", image.size())
        
        #Wrap them in a Variable object
        img = Variable(image)
        
        #Forward pass to extract moments for phase 2(this will be done one image at a time)
        single_moment = net.forward(img, phase = 1)
        #print ("-------------SIZE OF SINGLE MOMENT-----------", single_moment.size())
        output_image.append(single_moment.data[0])
        output_labels.append(label)
    
    num_samples = len(output_labels)
    #print ("-----------------SIZE OF OUTPUT IMAGE-----------------", len(output_image))
    #print ("-----------------SIZE OF OUTPUT LABELS-----------------", len(output_labels))
    ret = (torch.cat(output_image, dim=0).view(num_samples, -1), torch.cat(output_labels, dim=0))
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
        img = Variable(image)
        
        #Forward pass to extract moments for phase 2(this will be done one image at a time)
        single_moment = net.forward(img, phase = 1)
        #print ("-------------SIZE OF SINGLE MOMENT-----------", single_moment.size())
        output_image.append(single_moment.data[0])
        output_labels.append(label)

    num_samples = len(output_labels)
    #print ("-----------------SIZE OF OUTPUT IMAGE-----------------", len(output_image))
    #print ("-----------------SIZE OF OUTPUT LABELS-----------------", len(output_labels))
    ret = (torch.cat(output_image, dim=0).view(num_samples, -1), torch.cat(output_labels, dim=0))
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
        img = Variable(image)
        
        #Forward pass to extract moments for phase 2(this will be done one image at a time)
        single_moment = net.forward(img, phase = 1)
        #print ("-------------SIZE OF SINGLE MOMENT-----------", single_moment.size())
        output_image.append(single_moment.data[0])
        output_labels.append(label)

    num_samples = len(output_labels)
    #print ("-----------------SIZE OF OUTPUT IMAGE-----------------", len(output_image))
    #print ("-----------------SIZE OF OUTPUT LABELS-----------------", len(output_labels))
    ret = (torch.cat(output_image, dim=0).view(num_samples, -1), torch.cat(output_labels, dim=0))
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
    #print ("TRAINING SET MOMENTS SIZE: ", M_tr[0].size())
    #print ("TRAINING SET LABELS SIZE: ", M_tr[1].size())
    
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
            
            #print ("labels before flattening ", labels.size())
            labels = labels.flatten()

            #Wrap them in a Variable object
            inputs, labels = Variable(inputs), Variable(labels)
            
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
            inputs, labels = Variable(inputs), Variable(labels)
            
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
            inputs, labels = Variable(inputs), Variable(labels)
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
#Get training data from the data_loader class
#TODO change batch_size and learning rate for testing purposes
batch_size_phase_1 = 1
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
batch_size_phase_3 = 3
learning_rate_phase_3 = 0.01
net_phase_2 = MLPNet()
train_MLP_net(net_phase_2, batch_size=batch_size_phase_3, n_epochs=3, learning_rate=learning_rate_phase_3, 
M_tr=extracted_moments_Mtr,M_val=extracted_moments_Mval,M_test=extracted_moments_Mtest)
