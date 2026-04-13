import torch
import torch.nn as nn
import torch.nn.functional as F
import torchrl as rl
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers import LazyMemmapStorage
from tensordict import TensorDict
import math
#import tensordict

import numpy as np
#the network
class Network(nn.Module):
    def __init__(self,inChannels,outDims,input_size):
        super().__init__()
        self.relu=nn.ReLU()
        self.flatten=nn.Flatten()
        self.avgpool=nn.AvgPool2d(kernel_size=60,stride=60)
        poolout=self._calcAvgPoolOutObj(input_size,self.avgpool)
        self.linear1=nn.Linear(int(poolout[0]*poolout[1]),32)
        self.linear2=nn.Linear(32,outDims)


    def forward(self,state):
        x_in=torch.from_numpy(np.expand_dims(state.sensorDat,axis=0))
        x=self.avgpool(x_in)
        x=self.flatten(x)
        x=self.linear1(x)
        x=self.relu(x)
        x=self.linear2(x)
        x=self.relu(x)
        return x
    #algorithem for preforming sanity cheacks and tests
    def _manualEncoding(self,inp):
        inp=inp.squeeze()
        agmax=np.unravel_index(inp.argmax(), inp.shape)
        print(agmax,inp)
        if(agmax[0]==1 and agmax[1]==1):
            return torch.tensor(np.array([1,0,0,0,0],np.float64))
        ret=np.zeros(5)
        if(agmax[0]==2):
            ret[2]=1
        if(agmax[0]==0):
            ret[1]=1
        if(agmax[1]==2):
            ret[4]=1
        if(agmax[1]==0):
            ret[3]=1
        return torch.tensor(ret)
    #helper functions
    def _calcAvgPoolOutObj(self,in_size,pool_layer):

        kernal_size=pool_layer.kernel_size
        padding=pool_layer.padding
        dilation=(0,0)
        stride=pool_layer.stride
        
        if(type(pool_layer.padding) == int):
            padding=(pool_layer.padding,pool_layer.padding)
    
        
        if(type(pool_layer.kernel_size) == int):
            kernal_size=(pool_layer.kernel_size,pool_layer.kernel_size)
        
        if(type(pool_layer.stride) == int):
            stride=(pool_layer.stride,pool_layer.stride)



        return self._calcAvgPoolOut(in_size,kernal_size,stride,padding)
    def _calcAvgPoolOut(self,in_size,kernal_size,stride,padding):
        out_size=[0,0]
        if(padding is int):

            padding=(padding,padding)
        
        out_size[0]=(in_size[0]+2*padding[0]-kernal_size[0])/stride[0]+1
        out_size[1]=(in_size[1]+2*padding[1]-kernal_size[1])/stride[1]+1
        out_size[0]=math.floor(out_size[0])
        out_size[1]=math.floor(out_size[1])
        return out_size
    #wrapper for below function
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
    
    
    #internal meathod for calcuating output dims of a pooling layer
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

#main RL class
class drone_brain():


    def __init__(self,explore_factor,explore_decay,explore_min,lr,input_size,far_dist,near_dist):
        self.far_dist=far_dist
        self.near_dist=near_dist
        
        self.action_num=5
        self.net=Network(inChannels=1,outDims=self.action_num,input_size=input_size)
        self.optimizer = torch.optim.Adam(params=self.net.parameters(),lr=lr)

        self.rewards=[]
        self.action_probs=[]
        self.lifetime_rewards=[]

        self.explore_factor=explore_factor
        self.explore_decay=explore_decay
        self.explore_min=explore_min

        self.curr_step=0

    #predict the best move for a given state
    def act(self,state,is_zero_epoch):

        probs_raw=self.net.forward(state)
        #print(probs)
        probs=probs_raw.squeeze()


        explore_modifier=torch.full(size=[self.action_num],fill_value=self.explore_factor)
        keep_idx=torch.argmax(probs)
        explore_modifier[keep_idx]=1

        zero_modifier=torch.zeros(self.action_num)
        zero_modifier[torch.isclose(probs,torch.zeros(self.action_num),rtol=0.01)]=0.1
        zero_modifier[torch.isclose(probs,torch.zeros(self.action_num),rtol=0.01)==False]=0
        probs=probs+zero_modifier
        #print(probs)

        probs=(probs)*explore_modifier
        #print(probs)
        #probs=probs+torch.full(size=[self.action_num],fill_value=0.1)
        if is_zero_epoch:
            probs=(probs+1)*torch.Tensor([1,0,0,0,0])

        if(probs.sum()==0): #fix for if the network put out all zeros
            probs+=0.1
        action_raw=torch.tensor([0,0,0,0,0])
        try:
            action_raw=torch.multinomial(probs,1,replacement=True)
        except Exception as e:
            print("we've erred!") #for debuging, print probs if error in action selection
            print(probs)
            print("exception: "+str(e))
        action_formatted=F.one_hot(action_raw,num_classes=self.action_num)
        #action=self.classDecode(action_raw)
        #action=self.classDecodeExpiramntal(action_raw)
        #action=self.classDecodeExpiramntal2(action_raw)
        action=self.classDecodeLRFlipped(action_raw)
        

        self.curr_step+=1
        self.action_probs.append(torch.sum(action_formatted*probs))

        return action,probs_raw
    
    
    #log reward for last action, to be run after proccing the results of act()
    def analize_stateV1(self,state):
        reward=self.rewardFuncV1(state)
        self.rewards.append(reward)
        return reward

    def analize_stateV2(self,state):
        reward=self.rewardFuncV2(state,los_area_size=8)
        self.rewards.append(reward)
        return reward
    
    def analize_stateV3(self,state):
        reward=self.rewardFuncV3(state,los_area_size=8)
        self.rewards.append(reward)
        return reward
    
    def analize_stateV4(self,state):
        reward=self.rewardFuncV4(state,los_area_size=4)
        self.rewards.append(reward)
        return reward
    def analize_stateV5(self,state):
        reward=self.rewardFuncV5(state,los_area_size=4)
        self.rewards.append(reward)
        return reward
    def analize_stateV6(self,state):
        reward=self.rewardFuncV6(state,los_area_size=8)
        self.rewards.append(reward)
        return reward
    
    #reward function
    def rewardFunc_old(self,next_state):
        # +1 to reward if reached the end, -1 to reward if crashed
        # main part, sd = goal distance from start, cd = current distance to goal. reward calulation: (sd-cd)/sd=r
        # so when cd -> 0, r -> 1
        return (1 if next_state.is_sucesssful else 0) + (-5 if next_state.has_crashed else 0)+((next_state.goalDistFromStart-next_state.goalDist)/next_state.goalDistFromStart)
    
    def rewardFuncV1(self,next_state):
        # +1 to reward if reached the end, -1 to reward if crashed
        # main part, sd = goal distance from start, cd = current distance to goal. reward calulation: (sd-cd)/sd=r
        # so when cd -> 0, r -> 1
        # for some stupid reason, my code insisstes on minimising the reward. therefore, the reward function is now negeted
        return-( (1 if next_state.is_sucesssful else 0) + (-1 if next_state.has_crashed else 0)+((next_state.goalDistFromStart-next_state.goalDist)/next_state.goalDistFromStart)*2)
    
    def rewardFuncV2(self,next_state,los_area_size=4):
        # +1 to reward if reached the end, -1 to reward if crashed
        # main part, sd = goal distance from start, cd = current distance to goal. reward calulation: (sd-cd)/sd=r
        # so when cd -> 0, r -> 1
        #factor in the mean value of an area in the middle of the screen
        dat=next_state.sensorDat
        x_mid=dat.shape[1]//2
        x_start=x_mid-(los_area_size//2-1)
        x_end=x_mid+1+(los_area_size//2-1)
        y_mid=dat.shape[0]//2
        y_start=x_mid-(los_area_size//2-1)
        y_end=x_mid+1+(los_area_size//2-1)
        los=dat[y_start:y_end,x_start:x_end]

        # for some stupid reason, my code insisstes on minimising the reward. therefore, the reward function is now negeted
        return-( (1 if next_state.is_sucesssful else 0) + (-5 if next_state.has_crashed else 0)+((next_state.goalDistFromStart-next_state.goalDist)/next_state.goalDistFromStart)*0.5+(np.mean(los)*2))
    
    def rewardFuncV3(self,next_state,los_area_size=4):
        # +1 to reward if reached the end, -1 to reward if crashed
        # main part, sd = goal distance from start, cd = current distance to goal. reward calulation: (sd-cd)/sd=r
        # so when cd -> 0, r -> 1
        #factor in the mean value of an area in the middle of the screen
        dat=next_state.sensorDat
        x_mid=dat.shape[1]//2
        x_start=x_mid-(los_area_size//2-1)
        x_end=x_mid+1+(los_area_size//2-1)
        y_mid=dat.shape[0]//2
        y_start=x_mid-(los_area_size//2-1)
        y_end=x_mid+1+(los_area_size//2-1)
        los=dat[y_start:y_end,x_start:x_end]

        #los=los-0.1
        #los[los<0]=0


        # for some stupid reason, my code insisstes on minimising the reward. therefore, the reward function is now negeted
        return ( (1 if next_state.is_sucesssful else 0) + (-2 if next_state.has_crashed else 0)+((next_state.goalDistFromStart-next_state.goalDist)/next_state.goalDistFromStart)*0.16+(np.mean(los)*1))
    
    def rewardFuncV4(self,next_state,los_area_size=4):
        # +1 to reward if reached the end, -1 to reward if crashed
        # main part, sd = goal distance from start, cd = current distance to goal. reward calulation: (sd-cd)/sd=r
        # so when cd -> 0, r -> 1
        #factor in the mean value of an area in the middle of the screen
        dat=next_state.sensorDat
        x_mid=dat.shape[1]//2
        x_start=x_mid-(los_area_size//2-1)
        x_end=x_mid+1+(los_area_size//2-1)
        y_mid=dat.shape[0]//2
        y_start=x_mid-(los_area_size//2-1)
        y_end=x_mid+1+(los_area_size//2-1)
        los=dat[y_start:y_end,x_start:x_end]


        pseudo_max=np.max(dat)
        pseudo_max=pseudo_max-pseudo_max*0.1
        poses=np.argwhere(dat>pseudo_max)
        #account for if poses is empty (becouse it for some reason can still be)
        if poses[:,0].shape[0]!=0:
            y_thingy=np.mean(poses[:,0])
        else:
            y_thingy=0.5
        
        if poses[:,1].shape[0]!=0:
            x_thingy=np.mean(poses[:,1])
        else:
            x_thingy=0.5
        y_thingy=-abs(y_thingy-y_mid)
        x_thingy=-abs(x_thingy-x_mid)
        y_thingy=y_thingy/dat.shape[0]
        x_thingy=x_thingy/dat.shape[1]
        #los=los-0.1
        #los[los<0]=0
        dist=math.sqrt(y_thingy**2+x_thingy**2)
        print(dist)

        # for some stupid reason, my code insisstes on minimising the reward. therefore, the reward function is now negeted
        return -( (1 if next_state.is_sucesssful else 0) + (-2 if next_state.has_crashed else 0)+((next_state.goalDistFromStart-next_state.goalDist)/next_state.goalDistFromStart)*0.5-(dist)*2)
    def rewardFuncV5(self,next_state,los_area_size=4):
        # +1 to reward if reached the end, -1 to reward if crashed
        # main part, sd = goal distance from start, cd = current distance to goal. reward calulation: (sd-cd)/sd=r
        # so when cd -> 0, r -> 1
        #factor in the mean value of an area in the middle of the screen
        dat=next_state.sensorDat
        x_mid=dat.shape[1]//2
        x_start=x_mid-(los_area_size//2-1)
        x_end=x_mid+1+(los_area_size//2-1)
        y_mid=dat.shape[0]//2
        y_start=x_mid-(los_area_size//2-1)
        y_end=x_mid+1+(los_area_size//2-1)
        los=dat[y_start:y_end,x_start:x_end]


        pseudo_max=np.max(dat)
        pseudo_max=pseudo_max-pseudo_max*0.1
        poses=np.argwhere(dat>pseudo_max)
        #account for if poses is empty (becouse it for some reason can still be)
        if poses[:,0].shape[0]!=0:
            y_thingy=np.mean(poses[:,0])
        else:
            y_thingy=0.5
        
        if poses[:,1].shape[0]!=0:
            x_thingy=np.mean(poses[:,1])
        else:
            x_thingy=0.5
        y_thingy=abs(y_thingy-y_mid)
        x_thingy=abs(x_thingy-x_mid)
        y_thingy=y_thingy/dat.shape[0]
        x_thingy=x_thingy/dat.shape[1]
        #los=los-0.1
        #los[los<0]=0
        dist=math.sqrt((y_thingy**2)+(x_thingy**2))
        print(str(dist)+"|"+str(x_thingy*dat.shape[1])+"|"+str(y_thingy*dat.shape[0]))

        # for some stupid reason, my code insisstes on minimising the reward. therefore, the reward function is now negeted
        return (((next_state.goalDistFromStart-next_state.goalDist)/next_state.goalDistFromStart)*0.5-(dist))
    
    def rewardFuncV6(self,next_state,los_area_size=4):
        # +1 to reward if reached the end, -1 to reward if crashed
        # main part, sd = goal distance from start, cd = current distance to goal. reward calulation: (sd-cd)/sd=r
        # so when cd -> 0, r -> 1
        #factor in the mean value of an area in the middle of the screen
        dat=next_state.sensorDat
        x_mid=dat.shape[1]//2
        x_start=x_mid-(los_area_size//2-1)
        x_end=x_mid+1+(los_area_size//2-1)
        y_mid=dat.shape[0]//2
        y_start=x_mid-(los_area_size//2-1)
        y_end=x_mid+1+(los_area_size//2-1)
        los=dat[y_start:y_end,x_start:x_end]

        edge_mean=np.sum(dat)-np.sum(los)
        edge_mean=edge_mean/(dat.shape[0]*dat.shape[1]-los.shape[0]*los.shape[1])

        pseudo_max=np.max(dat)
        pseudo_max=pseudo_max-pseudo_max*0.1
        poses=np.argwhere(dat>pseudo_max)
        #account for if poses is empty (becouse it for some reason can still be)
        if poses[:,0].shape[0]!=0:
            y_thingy=np.mean(poses[:,0])
        else:
            y_thingy=0.5
        
        if poses[:,1].shape[0]!=0:
            x_thingy=np.mean(poses[:,1])
        else:
            x_thingy=0.5
        y_thingy=abs(y_thingy-y_mid)
        x_thingy=abs(x_thingy-x_mid)
        #y_thingy=y_thingy/dat.shape[0]
        #x_thingy=x_thingy/dat.shape[1]
        #los=los-0.1
        #los[los<0]=0
        dist=math.sqrt((y_thingy**2)+(x_thingy**2))
        dist=dist/math.sqrt((dat.shape[0]**2)+(dat.shape[1]**2))

        pseudo_min=np.min(dat)
        poses=np.argwhere(dat<=pseudo_min)
        if poses[:,0].shape[0]!=0:
            tmp=np.abs(poses[:,0]-y_mid)
            y_thingy=np.min(tmp)
        else:
            y_thingy=0.5
        
        if poses[:,1].shape[0]!=0:
            tmp=np.abs(poses[:,1]-x_mid)
            x_thingy=np.min(tmp)
        else:
            x_thingy=0.5
        
        dist_min=math.sqrt((y_thingy**2)+(x_thingy**2))
        dist_min=dist_min/math.sqrt((dat.shape[0]**2)+(dat.shape[1]**2))


        d=np.min(los)
        #print(d)
        d=d*self.far_dist+self.near_dist
        if(d < 5 and d > 0):
            los=np.exp(-d/1.0)
        else:
            los=0
        


        #print(str(los)+"|"+str(d))

        
        return (((next_state.goalDistFromStart-next_state.goalDist)/next_state.goalDistFromStart)*0-(dist*0.75)+(dist_min*0.25)+(10 if next_state.is_sucesssful else 0) + (-6 if next_state.has_crashed else 0))
    
    #alias for optimizer.zero_grad()
    def zerograd(self):    
        self.optimizer.zero_grad()

    #backprop
    #no clue if any of this actually works but lets hope for the best

    def learn(self):

        self.optimizer.zero_grad()
        
        discounted=self.calc_discounted(0.1)
        loss=self.calcLoss(self.action_probs,discounted)
        loss.backward()
        self.optimizer.step()

        #log rewards for this epoch and clear memeory
        self.lifetime_rewards.append(self.rewards)
        self.rewards=[]
        self.action_probs=[]
        
        #decay explore
        self.explore_factor-=self.explore_decay
        self.explore_factor=max(self.explore_factor,self.explore_min)
        return loss.item()
        
    #log rewards and clear memory without preforming backprop
    def noLearn(self):
        self.lifetime_rewards.append(self.rewards)
        self.rewards=[]
        self.action_probs=[]
    

    
    #calculate loss
    def calcLoss(self,action_likleyhoods,rewards):
        loss=[]
        
        for probs,reward in zip(action_likleyhoods,rewards):
            loss.append(torch.log(probs)*reward)
        loss=torch.stack(loss)

        return -torch.mean(loss)


    #calculate discounted rewards
    def calc_discounted(self,gamma):
        ret =np.zeros(len(self.rewards))
        running_value=0
        for i in reversed(range(len(self.rewards))): #itirate over rewards in reversed order
            running_value=self.rewards[i]+gamma*running_value
            ret[i]=running_value #add discounted reward to ret in appropriate index
        return ret
        
    #saves the model
    def save(self,path,epoch):
        torch.save(
            dict(net=self.net.state_dict(),explore_chance=self.explore_factor,epochs_left=epoch),
            path
        )

    #load a model from file
    def load(self,path):
        dat=torch.load(path)
        self.explore_factor=dat["explore_chance"]
        self.net.load_state_dict(dat["net"])
        return dat["epochs_left"]
    
    #decodes network output 
    def classDecode(self,chosenAction): #chosen action: number in range 1-13
        map ={1:np.array([0, 1, 0, 0, 0, 0]),2:np.array([0, -1, 0, 0, 0, 0]),3:np.array([0, 0, 0, 10, 0, 0]),4:np.array([0, 0, 0, -10, 0, 0]),5:np.array([0, 0, 0, 0, 0, 1]),6:np.array([0, 0, 0, 0, 0, -1]),7:np.array([0, 0, 0, 0, 10, 0]),8:np.array([0, 0, 0, 0, -10, 0]),9:np.array([0, 0, 1, 0, 0, 0]),10:np.array([0, 0, -1, 0, 0, 0]),11:np.array([-1, 0, 0, 0, 0, 0]),12:np.array([1, 0, 0, 0, 0, 0]),13:np.array([0, 0, 0, 0, 0, 0])}
        #     w                              s                               left                           right                             e                              q                               up                              down                             space                          shift                            a                                 d                               no action
        return map[chosenAction.item()+1]
    
    #classDecode without rotation
    def classDecodeExpiramntal(self,chosenAction): #chosen action: number in range 1-13
        map ={1:np.array([0, 1, 0, 0, 0, 0]),2:np.array([0, -1, 0, 0, 0, 0]),3:np.array([0, 0, 1, 0, 0, 0]),4:np.array([0, 0, -1, 0, 0, 0]),5:np.array([-1, 0, 0, 0, 0, 0]),6:np.array([1, 0, 0, 0, 0, 0]),7:np.array([0, 0, 0, 0, 0, 0])}
       
        return map[chosenAction.item()+1]
    #variation of classDecode
    def classDecodeExpiramntal2(self,chosenAction): #chosen action: number in range 1-13 #no backwards movement
        map ={1:np.array([0, 1, 0, 0, 0, 0]),2:np.array([0, 0, 1, 0, 0, 0]),3:np.array([0, 0, -1, 0, 0, 0]),4:np.array([-1, 0, 0, 0, 0, 0]),5:np.array([1, 0, 0, 0, 0, 0]),6:np.array([0, 0, 0, 0, 0, 0])}
       
        return map[chosenAction.item()+1]
    #variation of classDecode
    def classDecodeLRFlipped(self,chosenAction): #chosen action: number in range 1-13 #no backwards movement
        map ={1:np.array([0, 0, 0, 0, 0, 0]),2:np.array([0, 0, 1, 0, 0, 0]),3:np.array([0, 0, -1, 0, 0, 0]),5:np.array([-1, 0, 0, 0, 0, 0]),4:np.array([1, 0, 0, 0, 0, 0]),6:np.array([0, 0, 0, 0, 0, 0])}
       
        return map[chosenAction.item()+1]


