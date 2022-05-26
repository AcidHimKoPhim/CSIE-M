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


class DenseSkipCntBlock(nn.Module):
    def __init__(self, num_features, act_type, norm_type):
        super(DenseSkipCntBlock, self).__init__()
        stride = 1
        padding = 1
        kernel_size = 3
        self.conv1 = ConvBlock(num_features+1*num_features, num_features, kernel_size, stride, padding=padding, norm_type=norm_type, act_type=act_type)
        self.conv2 = ConvBlock(num_features+2*num_features, num_features, kernel_size, stride, padding=padding, norm_type=norm_type, act_type=act_type)
        self.conv3 = ConvBlock(num_features+3*num_features, num_features, kernel_size, stride, padding=padding, norm_type=norm_type, act_type=act_type)
        self.conv4 = ConvBlock(num_features+4*num_features, num_features, kernel_size, stride, padding=padding, norm_type=norm_type, act_type=act_type)
        self.conv5 = ConvBlock(num_features+5*num_features, num_features, kernel_size, stride, padding=padding, norm_type=norm_type, act_type=act_type)
        self.conv6 = ConvBlock(num_features+6*num_features, num_features, kernel_size, stride, padding=padding, norm_type=norm_type, act_type=act_type)
        self.conv7 = ConvBlock(num_features+7*num_features, num_features, kernel_size, stride, padding=padding, norm_type=norm_type, act_type=act_type)
        self.conv8 = ConvBlock(num_features+8*num_features, num_features, kernel_size, stride, padding=padding, norm_type=norm_type, act_type=act_type)
                                
        self.compress_out = ConvBlock(num_features, num_features,
                                      kernel_size=1,  padding=padding, 
                                      act_type=act_type, norm_type=norm_type)

        self.should_reset = True
        self.last_hidden = None

    def forward(self, x):
        if self.should_reset:
            self.last_hidden = torch.zeros(x.size()).cuda()
            self.last_hidden.copy_(x)
            self.should_reset = False

        x = torch.cat((x, self.last_hidden), dim=1)
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1)) 
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x6 = self.conv6(torch.cat((x, x1, x2, x3, x4, x5), 1))
        x7 = self.conv7(torch.cat((x, x1, x2, x3, x4, x5, x6), 1))
        x8 = self.conv8(torch.cat((x, x1, x2, x3, x4, x5, x6, x7), 1))
        output = self.compress_out(x8)

        self.last_hidden = output

        return output

    def reset_state(self):
        self.should_reset = True


class SKIPRR(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, num_steps, act_type = 'prelu', norm_type = None):
        super(SKIPRR, self).__init__()

        stride = 1
        padding = 1
        kernel_size = 3
    
        self.num_steps = num_steps
        self.num_features = num_features

        # RGB mean for DIV2K
        #rgb_mean = (0.4488, 0.4371, 0.4040)
        #rgb_std = (1.0, 1.0, 1.0)
        #self.sub_mean = MeanShift(rgb_mean, rgb_std)

        # LR feature extraction block
        self.conv_in1 = ConvGroup(in_channels, num_features,
                                 kernel_size=3, padding=padding,
                                 act_type=act_type, norm_type=norm_type)
        self.conv_in2 = ConvGroup(in_channels, num_features,
                                 kernel_size=3, padding=padding,
                                 act_type=act_type, norm_type=norm_type)
        self.conv_in3 = ConvGroup(in_channels, num_features,
                                 kernel_size=3, padding=padding,
                                 act_type=act_type, norm_type=norm_type)
        self.conv_merge = ConvGroup(num_features*3, num_features,
                                 kernel_size=3, padding=padding,
                                 act_type=act_type, norm_type=norm_type)

        # basic block
        self.block = DenseSkipCntBlock(num_features, act_type, norm_type)

        self.conv_out = ConvBlock(num_features, out_channels,
                                  kernel_size=3, padding=padding,
                                  act_type=None, norm_type=norm_type)


    def forward(self, x01, x02,x03):
        self._reset_state()

        inter_res = x01

        x11 = self.conv_in1(x01)
        
        x12 = self.conv_in2(x02)
        
        x13 = self.conv_in3(x03)
        
        x2 = self.conv_merge(torch.cat((x11, x12,x13), 1))
        outs = []
        for _ in range(self.num_steps):
            x3 = self.block(x2)
            
            x4 = self.conv_out(x3)
            
            h = torch.add(inter_res, x4)
            outs.append(h)

        return outs # return output of every timesteps

    def _reset_state(self):
        self.block.reset_state()

'''        
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


class DenseSkipCntBlock(nn.Module):
    def __init__(self, num_features, act_type, norm_type):
        super(DenseSkipCntBlock, self).__init__()
        stride = 1
        padding = 1
        kernel_size = 3
        self.conv1 = ConvBlock(num_features+1*num_features, num_features, kernel_size, stride, padding=padding, norm_type=norm_type, act_type=act_type)
        self.conv2 = ConvBlock(num_features+2*num_features, num_features, kernel_size, stride, padding=padding, norm_type=norm_type, act_type=act_type)
        self.conv3 = ConvBlock(num_features+3*num_features, num_features, kernel_size, stride, padding=padding, norm_type=norm_type, act_type=act_type)
        self.conv4 = ConvBlock(num_features+4*num_features, num_features, kernel_size, stride, padding=padding, norm_type=norm_type, act_type=act_type)
        self.conv5 = ConvBlock(num_features+5*num_features, num_features, kernel_size, stride, padding=padding, norm_type=norm_type, act_type=act_type)
        self.conv6 = ConvBlock(num_features+6*num_features, num_features, kernel_size, stride, padding=padding, norm_type=norm_type, act_type=act_type)
        self.conv7 = ConvBlock(num_features+7*num_features, num_features, kernel_size, stride, padding=padding, norm_type=norm_type, act_type=act_type)
        self.conv8 = ConvBlock(num_features+8*num_features, num_features, kernel_size, stride, padding=padding, norm_type=norm_type, act_type=act_type)
                                
        self.compress_out = ConvBlock(num_features, num_features,
                                      kernel_size=1,  padding=padding, 
                                      act_type=act_type, norm_type=norm_type)

        self.should_reset = True
        self.last_hidden = None

    def forward(self, x):
        if self.should_reset:
            self.last_hidden = torch.zeros(x.size()).cuda()
            self.last_hidden.copy_(x)
            self.should_reset = False

        x = torch.cat((x, self.last_hidden), dim=1)
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1)) 
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x6 = self.conv6(torch.cat((x, x1, x2, x3, x4, x5), 1))
        x7 = self.conv7(torch.cat((x, x1, x2, x3, x4, x5, x6), 1))
        x8 = self.conv8(torch.cat((x, x1, x2, x3, x4, x5, x6, x7), 1))
        output = self.compress_out(x8)

        self.last_hidden = output

        return output

    def reset_state(self):
        self.should_reset = True


class SKIPRR(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, num_steps, act_type = 'prelu', norm_type = None):
        super(SKIPRR, self).__init__()

        stride = 1
        padding = 1
        kernel_size = 3
    
        self.num_steps = num_steps
        self.num_features = num_features

        # RGB mean for DIV2K
        #rgb_mean = (0.4488, 0.4371, 0.4040)
        #rgb_std = (1.0, 1.0, 1.0)
        #self.sub_mean = MeanShift(rgb_mean, rgb_std)

        # LR feature extraction block
        self.conv_in1 = ConvBlock(in_channels, num_features,
                                 kernel_size=3, padding=padding,
                                 act_type=act_type, norm_type=norm_type)
        self.conv_in2 = ConvBlock(in_channels, num_features,
                                 kernel_size=3, padding=padding,
                                 act_type=act_type, norm_type=norm_type)
        self.conv_in3 = ConvBlock(in_channels, num_features,
                                 kernel_size=3, padding=padding,
                                 act_type=act_type, norm_type=norm_type)
        self.conv_merge = ConvBlock(num_features*3, num_features,
                                 kernel_size=3, padding=padding,
                                 act_type=act_type, norm_type=norm_type)

        # basic block
        self.block = DenseSkipCntBlock(num_features, act_type, norm_type)
        #self.block = FeedbackBlock(num_features, act_type, norm_type)

        self.conv_out1 = ConvBlock(num_features, num_features,
                                  kernel_size=3, padding=padding,
                                  act_type=act_type, norm_type=norm_type)

        self.conv_out2 = ConvBlock(num_features, out_channels,
                                  kernel_size=3, padding=padding,
                                  act_type=None, norm_type=norm_type)

        #self.add_mean = MeanShift(rgb_mean, rgb_std, 1)

    def forward(self, x01, x02,x03):
        self._reset_state()

        #x = self.sub_mean(x)
        inter_res = x01

        x11 = self.conv_in1(x01)
        
        x12 = self.conv_in2(x02)
        
        x13 = self.conv_in3(x03)
        
        x2 = self.conv_merge(torch.cat((x11, x12,x13), 1))
        outs = []
        for _ in range(self.num_steps):
            x3 = self.block(x2)
            
            x4 = self.conv_out1(x3)
            x5 = self.conv_out2(x4)
            
            h = torch.add(inter_res, x5)
            #h = self.add_mean(h)
            outs.append(h)

        return outs # return output of every timesteps

    def _reset_state(self):
        self.block.reset_state()
'''