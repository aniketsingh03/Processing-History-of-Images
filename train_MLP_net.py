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
import pickle

def createLoss(net):
	"""create loss for the CNN
	"""
	loss = nn.CrossEntropyLoss()
	
	return loss

def createOptimizer(net, learning_rate=0.01):
	"""create optimizer for the CNN
	"""
	optimizer = optim.SGD(net.parameters(), lr=learning_rate)

	return optimizer

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

	return (model, optimizer, start_epoch)

def train_MLP_net(net, batch_size, optimizer, start_epoch, n_epochs, learning_rate, M_tr, M_val):
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
	
	#Create our loss and optimizer functions
	loss = createLoss(net)
	#Time for printing
	training_start_time = time.time()
	
	n_batches = len(train_loader)
	#Train the moment generator part with C_tr (phase 1)
	#Loop for n_epochs
	for epoch in range(start_epoch, n_epochs):
		running_loss = 0.0
		print_every = n_batches // 100
		print ("PRINT AFTER EVERY {} batches ".format(print_every))
		start_time = time.time()
		total_train_loss = 0
		#print ("size of data loader is ", len(train_loader))
		for i, data in enumerate(train_loader, 0):
			#data represents a single mini-batch
			#Get inputs
			inp, lab = data
			
			#print ("labels before flattening ", labels.size())
			lab = lab.flatten()

			#Wrap them in a cudaVariable object
			inputs = inp.cuda(device)
			labels = lab.cuda(device)
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
			#print ("Loss for batch {} is {:f}".format(i, loss_size.item()))
			running_loss += loss_size.item()
			total_train_loss += loss_size.item()
			
			#Print every 10th batch of an epoch
			if (i + 1) % (print_every + 1) == 0:
				print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
						epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))
				#Reset running loss and time
				running_loss = 0.0
				start_time = time.time()
		
		#save every epoch
		state = { 'epoch': epoch + 1, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(), }
		torch.save(state, PATH)

		#Extracting accuracy of best model till now
		dummy_model = MLPNet()
		dummy_optimizer = createOptimizer(dummy_model, learning_rate)
		previous_best_accuracy = getBestModelAccuracy(dummy_model, dummy_optimizer, BEST_MODEL_PATH)

		#At the end of the epoch, do a pass on the validation set
		total_val_loss = 0
		val_loss = 0
		total_val = 0
		correct_val = 0
		current_accuracy = 0
		for inp, lab in val_loader:
			lab = lab.flatten()
			#DEBUG STATEMENTS
			#print ("-------------------INPUTS SIZE-----------------", inputs.size())
			#print ("-------------------LABELS SIZE-----------------", labels.size())
			#Wrap tensors in Variables
			inputs = inp.cuda(device)
			labels = lab.cuda(device)
			inputs, labels = Variable(inputs), Variable(labels)
			
			#Forward pass
			val_outputs = net(inputs)
			#print ("-------------------OUTPUTS-----------------", val_outputs)
			val_loss_size = loss(val_outputs, labels)
			total_val_loss += val_loss_size.item()
			_, predicted = torch.max(val_outputs.data, 1)
			#print ("-----------------PREDICTED SIZE-------------------", predicted.size())
			total_val += labels.size(0)
			correct_val += (predicted == labels).sum().item()
		
		val_loss = total_val_loss / len(val_loader)
		current_accuracy = 100 * correct_val / total_val
		print('Current accuracy for validation phase is : %d %%' % (
			current_accuracy))
		print("Validation loss = {:.2f}".format(val_loss))

		with open('stats.log', 'a') as logfile:
			logfile.write("Epoch = {:d}, Average train Loss = {:.2f}, Validation Loss = {:.2f}, Validation Accuracy = {:.3f} \n".format(epoch, total_train_loss/n_batches, val_loss, current_accuracy))
		
		#Saving best model till now
		if current_accuracy>previous_best_accuracy:
			state = { 'accuracy': current_accuracy, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(), }
			torch.save(state, BEST_MODEL_PATH)

		torch.cuda.empty_cache()
		
	print("Training for phase 3 finished, took {:.2f}s".format(time.time() - training_start_time))

#Extract moments saved in disk
saved_train_moments_filename = "train_moments"
saved_val_moments_filename = "val_moments"

train_moments, train_labels = loadSavedMoments(saved_train_moments_filename)
train_moments, train_labels = convertListsToTensors(train_moments, train_labels)

val_moments, val_labels = loadSavedMoments(saved_val_moments_filename)
val_moments, val_labels = convertListsToTensors(val_moments, val_labels)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print (device)

#path to save each training epoch
PATH = 'checkpoints_MLP.pth'
BEST_MODEL_PATH = 'best_model_MLP.pth'
net_phase_2 = MLPNet()
batch_size_phase_2 = 100
learning_rate_phase_2 = 0.001
optimizer = createOptimizer(net_phase_2, learning_rate_phase_2)

net_phase_2, optimizer, start_epoch = load_checkpoints(net_phase_2, optimizer, PATH)

#move model and optimizer to cuda
net_phase_2 = net_phase_2.to(device)
for state in optimizer.state.values():
	for k, v in state.items():
		if isinstance(v, torch.Tensor):
			state[k] = v.to(device)

M_tr = (train_moments, train_labels)
M_val = (val_moments, val_labels)
train_MLP_net(net=net_phase_2, batch_size=batch_size_phase_2, optimizer=optimizer, start_epoch=start_epoch, n_epochs=1000, learning_rate=learning_rate_phase_2, M_tr=M_tr, M_val=M_val)