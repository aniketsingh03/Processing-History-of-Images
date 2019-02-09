import torch.nn as nn
import torch.nn.functional as F
import torch
from custom_pooling import *

#A combo of the below models is used in the final network used for testing

#Model for phase 1 of the process
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.numberOfPoolingMethods = 3
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1) #in_channels,out_channels,kernel_size respectively
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv7 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv8 = nn.Conv2d(512, 1024, 3, padding=1)
        
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.conv4.weight)
        nn.init.xavier_uniform_(self.conv5.weight)
        nn.init.xavier_uniform_(self.conv6.weight)
        nn.init.xavier_uniform_(self.conv7.weight)
        nn.init.xavier_uniform_(self.conv8.weight)

        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(512)
        self.bn8 = nn.BatchNorm2d(1024)
        
        self.pool = nn.AvgPool2d(2, stride = 2)
        self.avgPoolLastlayer = nn.AdaptiveAvgPool2d(1)
        self.maxPoolLastLayer = nn.AdaptiveMaxPool2d(1)
        
        self.dropoutLastLayer = nn.Dropout()
        
        #IP layers
        #TODO: check whether we need mlp here or just fully connected layers will do
        num_fc1 = 1024*self.numberOfPoolingMethods
        self.fc1 = nn.Linear(num_fc1, num_fc1)
        self.fc2 = nn.Linear(num_fc1, 5)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
    
    def forward(self, x, phase = 0):
        """training the CNN for phase 1 ie for images of fixed
        dimensions (say 512x512) and extracting moments for phase 2
        var *phase* is used to determine whether it is phase 1 or 2
        """
        x = self.pool(F.relu(self.bn2(self.conv2(self.conv1(x)))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = self.pool(F.relu(self.bn6(self.conv6(x))))
        x = self.pool(F.relu(self.bn7(self.conv7(x))))
        
        x = F.relu(self.bn8(self.conv8(x)))
        
        #total number of pooling methods must be 'numberOfPoolingMethods'
        a = self.avgPoolLastlayer(x).view(-1, 1024)
        b = self.maxPoolLastLayer(x).view(-1, 1024)
        dim_previous_layer = x.size(2)
        min_pool_2D = AdaptiveMinPool2D(dim_previous_layer)
        variance_pool_2D = AdaptiveVariancePool2D(dim_previous_layer)
        c = min_pool_2D.forward(x).view(-1, 1024)
        d = variance_pool_2D.forward(x).view(-1, 1024)

        x = torch.cat((a,b,c,d), dim = 1).view(-1, 4096)
        print (x.size()) #should be (Nx4096) for 4096 features in each image
        
        #just return the extracted moments of a batch in case of phase 2(val of phase=1)
        if (phase == 1):
            return x
        x = self.dropoutLastLayer()
        
        #TODO check whether we need a ReLU here or not
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.Softmax(x)
        
        return x

#Model for phase 3 of the process
class MLPNet(nn.Module):
    def __init__(self):
       super(MLPNet, self).__init__()
       self.fc1 = nn.Linear(4096, 4096)
       self.fc2 = nn.Linear(4096, 5)
       nn.init.normal_(self.fc1.weight, std=0.01)
       nn.init.normal_(self.fc2.weight, std=0.01)

    def forward(self, x):
       """2 layer MLP for training the model for images of 
       random dimensions.
       """
       #TODO: check whether ReLU is needed here or not
       x = F.relu(self.fc1(x))
       x = self.fc2(x)
       x = self.Softmax(x)
       
       print(x.size())
       #x consists of the class probabilities for a batch
       return x
