import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.numberOfPoolingMethods = 2
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1) #in_channels,out_channels,kernel_size respectively
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv7 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv8 = nn.Conv2d(512, 1024, 3, padding=1)
        
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(512)
        self.bn8 = nn.BatchNorm2d(1024)
        
        self.pool = nn.AvgPool2d(2, stride = 2)
        self.avgPoolLastlayer = nn.AvgPool2d(8, stride = 8)
        self.maxPoolLastLayer = nn.MaxPool2d(8, stride = 8)
        
        self.dropoutLastLayer = nn.Dropout()
        
        num_fc1 = 1024*self.numberOfPoolingMethods
        self.fc1 = nn.Linear(num_fc1, num_fc1)
        self.fc2 = nn.Linear(num_fc1, 5)
        
    def forward(self, x, process):
        x = self.pool(F.relu(self.bn2(self.conv2(self.conv1(x)))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = self.pool(F.relu(self.bn6(self.conv6(x))))
        x = self.pool(F.relu(self.bn7(self.conv7(x))))
        
        x = F.relu(self.bn8(self.conv8(x)))
        
        #total number of pooling methods must be 'numberOfPoolingMethods'
        #calculating average pool
        a = self.avgPoolLastlayer(x)
        #calculating max pool
        b = self.maxPoolLastLayer(x)
        #TODO: according to the paper two more poolings have to be done namely minimum and variance.
        
        #concatenating these to form (numberOfPoolingMethods,1024) feature martrix brfore feeding to fully connected layer  
        #The value of numberOfPoolingMethods depends on the number of matrices concatenated here
        x = torch.cat((a,b), 0)
        print (x.size()) #should be (numberOfPoolingMethods,1024) as of now ie without min pooling and variance pooling layer
        
        x = x.view(-1, 1024)
        print (x.size())
        
        x = self.dropoutLastLayer()
        return x
    
    #feature_matrix is a matrix consisting of (1024*numberOfPoolingMethods) dimensions/features fed into fully connected layers
    def fullyConnected(self, feature_matrix):
        feature_matrix = F.relu(self.fc1(feature_matrix))
        feature_matrix = self.fc2(feature_matrix)
        softmax_probs = self.Softmax(feature_matrix)
        
        return softmax_probs
        