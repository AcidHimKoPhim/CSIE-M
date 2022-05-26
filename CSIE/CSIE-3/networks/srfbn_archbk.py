import torch
import torch.nn as nn
from .blocks import ConvBlock, DeconvBlock, MeanShift

class ConvGroup(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,stride=1,padding=1,act_type='prelu', bias=True):
        super(ConvGroup, self).__init__()
        
        act_type = act_type.lower()
        layer = None
        self.ac1 = nn.PReLU(num_parameters=1, init=0.2)
        self.ac2 = nn.PReLU(num_parameters=1, init=0.2)
        if act_type == 'relu':
            self.ac1 = nn.ReLU(inplace=True)
            self.ac2 = nn.ReLU(inplace=True)       
            
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        
    def forward(self, x):
        cv1 = self.ac1(self.conv1(x))
        return self.ac2(self.conv2(cv1))

class FeedbackBlock(nn.Module):
    def __init__(self, num_features, num_groups, act_type, norm_type):
        super(FeedbackBlock, self).__init__()
        
        stride = 1
        padding = 2
        kernel_size = 5

        self.num_groups = num_groups
        
        #Compress the feedback from last step and the input
        self.compress_in = ConvBlock(2*num_features, num_features,
                                     kernel_size=1,
                                     act_type=act_type, norm_type=norm_type)

        self.upBlocks = nn.ModuleList()
        self.downBlocks = nn.ModuleList()
        self.uptranBlocks = nn.ModuleList()
        self.downtranBlocks = nn.ModuleList() 

        for idx in range(self.num_groups):
            self.upBlocks.append(ConvBlock(num_features, num_features,
                                             kernel_size=kernel_size, stride=stride, padding=padding,
                                             act_type=act_type, norm_type=norm_type))
            self.downBlocks.append(ConvBlock(num_features, num_features,
                                             kernel_size=kernel_size, stride=stride, padding=padding,
                                             act_type=act_type, norm_type=norm_type, valid_padding=False))
            if idx > 0:
                self.uptranBlocks.append(ConvBlock(num_features*(idx+1), num_features,
                                                   kernel_size=1, stride=1,
                                                   act_type=act_type, norm_type=norm_type))
                self.downtranBlocks.append(ConvBlock(num_features*(idx+1), num_features,
                                                     kernel_size=1, stride=1,
                                                     act_type=act_type, norm_type=norm_type))

        self.compress_out = ConvBlock(num_groups*num_features, num_features,
                                      kernel_size=1,
                                      act_type=act_type, norm_type=norm_type)

        self.should_reset = True
        self.last_hidden = None

    def forward(self, x):
        if self.should_reset:
            self.last_hidden = torch.zeros(x.size()).cuda()
            self.last_hidden.copy_(x)
            self.should_reset = False
            
            
        x = torch.cat((x, self.last_hidden), dim=1)
        x = self.compress_in(x)

        lr_features = []
        hr_features = []
        lr_features.append(x)

        #forward the recurrent block
        for idx in range(self.num_groups):
            LD_L = torch.cat(tuple(lr_features), 1)    # when idx == 0, lr_features == [x]
            if idx > 0:
                LD_L = self.uptranBlocks[idx-1](LD_L)
            LD_H = self.upBlocks[idx](LD_L)

            hr_features.append(LD_H)

            LD_H = torch.cat(tuple(hr_features), 1)
            if idx > 0:
                LD_H = self.downtranBlocks[idx-1](LD_H)
            LD_L = self.downBlocks[idx](LD_H)

            lr_features.append(LD_L)

        del hr_features
        output = torch.cat(tuple(lr_features[1:]), 1)   # leave out input x, i.e. lr_features[0]
        output = self.compress_out(output)

        self.last_hidden = output

        return output

    def reset_state(self):
        self.should_reset = True

class SRFBN(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, num_steps, num_groups, act_type = 'prelu', norm_type = None):
        super(SRFBN, self).__init__()
        
        self.num_steps = num_steps
        self.num_features = num_features

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        
        #feature extraction block
        #branch 1
        self.block1 = ConvGroup(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        #branch 2
        self.block2 = ConvGroup(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.block3 = ConvGroup(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)

        # basic block
        self.rrblock = FeedbackBlock(num_features, num_groups, act_type, norm_type)

        self.block4 = ConvGroup(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        
        # reconstruction block
        self.conv_out = ConvBlock(num_features, out_channels,
                                  kernel_size=3,padding=1,
                                  act_type=None, norm_type=norm_type)
        
    def forward(self, x):
        self._reset_state()
        
        inter_res = x  #no 
        x = self.conv1(x)
        x = self.relu1(x)
        
        #branch 1
        x1 = self.block1(x)
        
        #branch 2
        x2 = self.block2(x)
        
        #cat branch 1 and 2       
        x3 = self.block3(torch.cat((x1,x2),1))
        
        outs = []
        for _ in range(self.num_steps):
            h = self.rrblock(x3)
            fbout = self.conv_out(h) #no deconv block
            h = torch.add(inter_res, fbout)
            outs.append(h)  
        return outs # return output of every timesteps

    def _reset_state(self):
        self.rrblock.reset_state()