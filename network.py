import torch.nn as nn
import torch.nn.functional as F
import torch

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
        
        nn.init.xavier_uniform(conv1.weight)
        nn.init.xavier_uniform(conv2.weight)
        nn.init.xavier_uniform(conv3.weight)
        nn.init.xavier_uniform(conv4.weight)
        nn.init.xavier_uniform(conv5.weight)
        nn.init.xavier_uniform(conv6.weight)
        nn.init.xavier_uniform(conv7.weight)
        nn.init.xavier_uniform(conv8.weight)

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
        
    def variancePooling(self, x):
        num_features = x.size()[0] #assuming (featx512x512) tensor
        a = torch.var(x[0,:,:]).view(1, 1)
        for i in range(1, num_features):
            a = torch.cat((a, torch.var(x[i,:,:]).view(1, 1)))
        a = a.view(-1,num_features)    
        return a            
    
    def minPooling(self, x):
        num_features = x.size()[0]
        a = torch.min(x[0,:,:]).view(1, 1)
        for i in range(1, num_features):
            a = torch.cat((a, torch.min(x[i,:,:]).view(1, 1)))
        a = a.view(-1, num_features)
        return a
    
    def forward(self, x, phase = 0):
        x = self.pool(F.relu(self.bn2(self.conv2(self.conv1(x)))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = self.pool(F.relu(self.bn6(self.conv6(x))))
        x = self.pool(F.relu(self.bn7(self.conv7(x))))
        
        x = F.relu(self.bn8(self.conv8(x)))
        
        #total number of pooling methods must be 'numberOfPoolingMethods'
        a = self.avgPoolLastlayer(x).view(-1,1024)
        b = self.maxPoolLastLayer(x).view(-1,1024)
        c = self.variancePooling(x)
        d = self.minPooling(x)

        x = torch.cat((a,b,c,d), 0).view(4096)
        print (x.size()) 
        
        x = x.view(-1, 1024)
        print (x.size()) #should be (numberOfPoolingMethods,1024)
        
        x = self.dropoutLastLayer()
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.Softmax(x)
        
        return x
