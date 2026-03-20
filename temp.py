import math
import torch.nn as nn
def _calcPoolOut(in_size,kernal_size,stride,padding,dilation):
        
        if(padding is int):

            padding=(padding,padding)
            
        
        out_size=[0,0]
        out_size[0]=(in_size[0]+2*padding[0]-dilation[0]*(kernal_size[0]-1)-1)/stride[0]+1
        out_size[1]=(in_size[1]+2*padding[1]-dilation[1]*(kernal_size[1]-1)-1)/stride[1]+1
        out_size[0]=math.floor(out_size[0])
        out_size[1]=math.floor(out_size[1])
        return out_size

def _calcPoolOutObj(in_size,pool_layer):

        kernal_size=pool_layer.kernel_size
        padding=pool_layer.padding
        dilation=pool_layer.dilation
        stride=pool_layer.stride
        
        if(type(pool_layer.padding) == int):
            padding=(pool_layer.padding,pool_layer.padding)
        
        if(type(pool_layer.dilation) == int):
            dilation=(pool_layer.dilation,pool_layer.dilation)
        
        if(type(pool_layer.kernel_size) == int):
            kernal_size=(pool_layer.kernel_size,pool_layer.kernel_size)
        
        if(type(pool_layer.stride) == int):
            stride=(pool_layer.stride,pool_layer.stride)



        return _calcPoolOut(in_size,kernal_size,stride,padding,dilation)
maxpool=nn.MaxPool2d(kernel_size=(8,12),stride=(8,12))
print(_calcPoolOutObj((30,40),maxpool))
    
