# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# ベースモデル
class Base(nn.Module):
    """ 
    input
    --------------------
    c_input (int) : the size of ECFP
    batch_size (int)
    dropout_rate (float)
    kernel_size (turple)
    num_kernel (int)
    pool_size (turple)
    pool_stride (int)
    H_0 (int) : the size of fully connected layer
    H_1 (int) : the size of fully connected layer
    H_2 (int) : the size of fully connected layer
    H_3 (int) : the size of fully connected layer
    out_size (int) : the size of output layer
    """

    # 
    def __init__(self, c_input=1024, batch_size=64, dropout_rate=0.5,
                kernel_size=(4,9), num_kernel=8, pool_size=(1,3), pool_stride=3,
                H_0=512, H_1=1024, H_2=256, H_3=64, out_size=2):
        super(Base, self).__init__()
        # hyper parameters
        self.dropout_rate = dropout_rate
        self.batch_size  = batch_size
        self.c_input     = c_input
        # convolution layer
        self.kernel_size = kernel_size
        self.num_kernel  = num_kernel
        self.pool_size   = pool_size
        self.pool_stride = pool_stride
        # fully connected layer
        self.H_0 = H_0
        self.H_1 = H_1
        self.H_2 = H_2
        self.H_3 = H_3
        self.out_size = out_size

        self.conv = nn.Conv2d(1,self.num_kernel,self.kernel_size)
        self.pool = nn.MaxPool2d(self.pool_size,stride=self.pool_stride)
        self.fc1 = nn.Linear(self.c_input,self.H_0)
        self.fc2 = nn.Linear(self.H_1,self.H_2)
        self.fc3 = nn.Linear(self.H_2,self.H_3)
        self.fc4 = nn.Linear(self.H_3,self.out_size)
        self.dropout = nn.Dropout(self.dropout_rate)

    #
    def forward(self, x1, x2):
        # RNA feature extraction module
        x1 = F.relu(self.conv(x1))
        x1 = self.pool(x1)
        x1 = x1.view(self.batch_size,1,self.H_0)
        # Compound feature extraction module
        x2 = F.relu(self.fc1(x2))
        # Interaction prediction module
        x = torch.cat([x1,x2], dim=2)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        return x.view(self.batch_size,self.out_size)