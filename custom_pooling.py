import torch

class AdaptiveMinPool2D(torch.nn.Module):
    def __init__(self, kernel_size):
        super(AdaptiveMinPool2D, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        N, C, H, W = [x.size(i) for i in range(4)]
        #print ("dimension of x in min pooling is ",x.size())
        x = x.view(N, C, int(H/self.kernel_size), W*self.kernel_size)
        #print ("without zero")
        #print (x.min(dim=3)[0].size(),x.min(dim=3)[1].size())
        #print ("with zero")
        #print (x.min(dim=3)[0].size())
        return x.min(dim=3)[0].view(N, C)

class AdaptiveVariancePool2D(torch.nn.Module):
    def __init__(self, kernel_size):
        super(AdaptiveVariancePool2D, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        N, C, H, W = [x.size(i) for i in range(4)]
        #print ("dimension of x in variance pooling is ",x.size())
        x = x.view(N, C, int(H/self.kernel_size), W*self.kernel_size)
        #print (x.var(dim=3).size())
        return x.var(dim=3).view(N, C)         
