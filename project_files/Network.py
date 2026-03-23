import torch
import torch.nn as nn
import torchrl as rl
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers import LazyMemmapStorage
from tensordict import TensorDict
import math
#import tensordict

import numpy as np
class Network(nn.Module):
    def __init__(self,inChannels,outDims,input_size):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=inChannels,out_channels=32,kernel_size=5,stride=1)
        conv1Out=self._calcConvOutObj(input_size,self.conv1)
        self.maxPool=nn.MaxPool2d(kernel_size=3,stride=2)
        pool1Out=self._calcPoolOutObj(conv1Out,self.maxPool)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=1,kernel_size=5,stride=1)
        conv2Out=self._calcConvOutObj(pool1Out,self.conv2)
        self.relu=nn.ReLU()
        self.flatten=nn.Flatten()
        self.linear1=nn.Linear(int(conv2Out[0]*conv2Out[1]),128)
        self.linear2=nn.Linear(128,outDims)

    def forward(self,state):
        x=torch.from_numpy(np.expand_dims(state.sensorDat,axis=0))
        #temporary model for debugging
        x=self.conv1(x)
        x=self.relu(x)
        x=self.maxPool(x)
        x=self.conv2(x)
        #TODO add pooling layer
        x=self.flatten(x)
        x=self.linear1(x)
        x=self.relu(x)
        x=self.linear2(x)
        return x
    
    def _calcPoolOutObj(self,in_size,pool_layer):

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



        return self._calcPoolOut(in_size,kernal_size,stride,padding,dilation)
    
    def _calcPoolOut(self,in_size,kernal_size,stride,padding,dilation):
        
        if(padding is int):

            padding=(padding,padding)
            
        
        out_size=[0,0]
        out_size[0]=(in_size[0]+2*padding[0]-dilation[0]*(kernal_size[0]-1)-1)/stride[0]+1
        out_size[1]=(in_size[1]+2*padding[1]-dilation[1]*(kernal_size[1]-1)-1)/stride[1]+1
        out_size[0]=math.floor(out_size[0])
        out_size[1]=math.floor(out_size[1])
        return out_size

    
    #wrapper for meathod below
    def _calcConvOutObj(self,in_size,conv_layer):
        return self._calcConvOut(in_size,conv_layer.kernel_size,conv_layer.stride,conv_layer.padding)
    
    #internal meathod to clalculate output dims of a conv2d layer
    def _calcConvOut(self,in_size,kernel_size,stride,padding):
        out_size=[0,0]
        #i know duplicate code is not good practice but here it just feels silly to add a third maethod
        out_size[0]=(in_size[0]-kernel_size[0]+2*padding[0])/stride[0] +1
        out_size[1]=(in_size[1]-kernel_size[1]+2*padding[0])/stride[0] +1

        if(out_size[0] %1 !=0 or out_size[1] %1 !=0):
            print("[WARNING] convolution output (size: "+str(out_size)+") has one or more non integer dimentions")

        return out_size


class NetworkPair(nn.Module):
    def __init__(self,inputChannels,outDims,input_size):
        super().__init__()

        self.online=Network(inputChannels,outDims,input_size)
        self.target = Network(inputChannels, outDims,input_size)

        self.target.load_state_dict(self.online.state_dict()) #copy online paramaters to target network

        #disable learning for trget network
        for p in self.target.parameters():
            p.requires_grad =False

    def forward(self,state,use_target=False):
        if(use_target):
            return self.target(state)
        else:
            return self.online(state)






class drone_brain():


    def __init__(self,explore_chance,explore_decay,explore_min,lr,input_size):
        self.explore_chance=explore_chance
        self.explore_decay=explore_decay
        self.explore_min=explore_min

        self.memory=TensorDictReplayBuffer(storage= LazyMemmapStorage(100000,device=torch.device("cpu")))
        self.recall_batchSize=32

        self.net=NetworkPair(inputChannels=1,outDims=6,input_size=input_size)
        self.optimizer = torch.optim.Adam(params=self.net.online.parameters(),lr=lr)
        self.loss_fn=torch.nn.SmoothL1Loss()

        self.burn = 1e4 # num of expiriences before training
        self.learn_interval = 3 #update paramaters every x expiriences
        self.sync_interval = 14 # sync every x expiriences

        self.save_interval =5e5

        self.curr_step=0


    def act(self,state):

        #explore
        if(np.random.rand() < self.explore_chance):
            random_move=np.random.rand(3,1)
            random_rotate=np.random.rand(3,1)
            ret=np.concatenate(random_move,random_rotate)

        #exploite
        else:
            ret=self.net(state)


        #exploration rate decay
        self.explore_chance *= self.explore_decay
        self.explore_chance=min(self.explore_min,self.explore_chance)

        self.curr_step+=1

        return ret


    def cache(self,state,next_state,action,reward,is_done):

        state=torch.tensor(state)
        next_state=torch.tensor(next_state)
        action=torch.tensor([action])
        reward=torch.tensor([reward])
        is_done=torch.tensor([is_done])

        self.memory.add(TensorDict({"state":state,"next_sate":next_state,"action":action,"reward":reward,"is_done":is_done}),batch_size=[])

    def learn(self,final_state):
        reward=self.rewardFunc(final_state)
        


    def rewardFunc(self,next_state):
        # +1 to reward if reached the end, -1 to reward if crashed
        # main part, sd = goal distance from start, cd = current distance to goal. reward calulation: (sd-cd)/sd=r
        # so when cd -> 0, r -> 1
        return (1 if next_state.is_sucesssful else 0) + (-1 if next_state.has_crashed else 0)+((next_state.goalDistFromStart-next_state.goalDist)/next_state.goalDistFromStart)

    def update_online(self,td_estimate,td_target):
        loss=self.loss_fn(td_estimate,td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        path="somthing"
        torch.save(
            dict(online = self.net.online.state_dict(),target=self.net.target.state_dict(),explore_chance=self.explore_chance),
            path
        )
        print("backup created")

    

    def classDecode(self,chosenAction): #chosen action: number in range 1-13
        map ={1:np.array([0, 1, 0, 0, 0, 0]),2:np.array([0, -1, 0, 0, 0, 0]),3:np.array([0, 0, 0, 10, 0, 0]),4:np.array([0, 0, 0, -10, 0, 0]),5:np.array([0, 0, 0, 0, 0, 1]),6:np.array([0, 0, 0, 0, 0, -1]),7:np.array([0, 0, 0, 0, 10, 0]),8:np.array([0, 0, 0, 0, -10, 0]),9:np.array([0, 0, 1, 0, 0, 0]),10:np.array([0, 0, -1, 0, 0, 0]),11:np.array([-1, 0, 0, 0, 0, 0]),12:np.array([1, 0, 0, 0, 0, 0]),13:np.array([0, 0, 0, 0, 0, 0])}
        #     w                              s                               left                           right                             e                              q                               up                              down                             space                          shift                            a                                 d                               no action
        return map[chosenAction]

