import pickle
import torch
from data_loader import *
from network import *
import sys
from torch.autograd import Variable

def loadCheckpoints(model, PATH):
	"""load pretrained model from disk
	"""
	if os.path.isfile(PATH):
		print("=> loading checkpoint '{}'".format(PATH))
		checkpoint = torch.load(PATH)
		model.load_state_dict(checkpoint['state_dict'])
		print("=> loaded checkpoint '{}' (epoch {})"
				  .format(PATH, checkpoint['epoch']))
	else:
		print("=> no checkpoint found at '{}'".format(PATH))

	return model

def loadSavedMoments(filename):
	"""load already existing moments from disk
	"""
	if os.path.isfile(filename):
		fileObject = open(filename, 'rb')
		moments_list, labels_list, number_of_images_processed = pickle.load(fileObject)
	else:
		moments_list, labels_list = [], []
		number_of_images_processed = 0
	
	return (moments_list, labels_list, number_of_images_processed)		

def storeMoments(filename, data):
	"""store newly extracted moments on disk
	"""
	fileObject = open(filename, 'wb')
	pickle.dump(data, fileObject)
	print ('Data successfully written on disk')

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
	Mtest_dataset = get_Mtest()
	print ("THE SIZE OF MTR DATASET IS ", sys.getsizeof(Mtest_dataset))
	output_image, output_labels, num_images_already_processed = loadSavedMoments(saved_moments_filename)
	num_of_images_processed = num_images_already_processed
	#print ("SET TO PASS: ", Mtest_dataset)
	image_paths = Mtest_dataset[0]
	print ("THE NUMBER OF IMAGES FOR TRAINING ARE ", len(image_paths))
	print ("THE SIZE OF IMAGE PATHS IS ", sys.getsizeof(image_paths))
	#print ("IMAGE PATHS SIZE: ", len(image_paths))
	image_labels = Mtest_dataset[1]
	print ("THE SIZE OF IMAGE LABELS IS ", sys.getsizeof(image_labels))
	#print ("IMAGE LABELS SIZE: ", len(image_labels))

	for i in range(num_images_already_processed, len(image_labels)):
		image, label = obtainDataAsTensors(image_paths[i], image_labels[i])
		num_of_images_processed+=1
		if (image.size(1)<2048 or image.size(2)<2048):
			print("Image number ", num_of_images_processed)
			print ("SIZE OF IMAGE: ", image.size())
			image = image.unsqueeze(0)
			print ("SIZE OF IMAGE TENSOR ON MEMORY IS ", image.element_size() * image.nelement())        
			#Wrap them in a Variable object
			img = image.cuda(device)
			img = Variable(img)
			#Forward pass to extract moments for phase 2(this will be done one image at a time)
			single_moment = net(img, phase = 1)
			print ("SIZE OF OUTPUT FROM MODEL ON DISK IS ", single_moment.element_size() * single_moment.nelement())
			#print ("-------------SIZE OF SINGLE MOMENT-----------", single_moment.size())
			output_image.append(single_moment.data[0])
			output_labels.append(label)

			if (num_of_images_processed%200==0):
				storeMoments(saved_moments_filename, (output_image, output_labels, num_of_images_processed))
			torch.cuda.empty_cache()

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print (device)

#path to save each training epoch
#CHANGE THIS PATH TO BEST MODEL'S PATH TO GET THE BEST MODEL
saved_model_filename = 'checkpoints.pth'
saved_moments_filename = 'test_moments'

net_phase_1 = Net()
net_phase_1 = loadCheckpoints(net_phase_1, saved_model_filename)
#move model to cuda
net_phase_1 = net_phase_1.to(device)
#net_phase_1.eval()

extractTrainMoments(net_phase_1)