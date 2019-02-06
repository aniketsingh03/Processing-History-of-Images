import torch

class AdaptiveMinPool2D(torch.nn.Module):
    def __init__(self, kernel_size):
        super(AdaptiveMinPool2D, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        N, C, H, W = [x.size(i) for i in range(4)]
        x = x.view(N, C, int(H/self.kernel_size), W*self.kernel_size)
        return x.min(dim=3)[0].view(N, C, -1)

class AdaptiveVariancePool2D(torch.nn.Module):
    def __init__(self, kernel_size):
        super(AdaptiveVariancePool2D, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        N, C, H, W = [x.size(i) for i in range(4)]
        x = x.view(N, C, int(H/self.kernel_size), W*self.kernel_size)
        return x.var(dim=3)[0].view(N, C, -1)         
