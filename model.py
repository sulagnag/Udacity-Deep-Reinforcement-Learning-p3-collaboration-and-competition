#network
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def hidden_init(layer):
    """Initialise the weights of the linear layers from a unifrom distribution"""
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


def linear_layers(nS,hidden_dim1,hidden_dim2, l2_in,l3_out):
    """Generic layers for Actor and Critic since both have the same structure
    
    Params
    =======
    nS (int) : state size
    hidden_dim1 (int) : dimension of the first hidden/linear layer
    hidden_dim2 (int) : dimension of the second hidden/linear layer
    l2_input (int) : size of input to the second layer
    l3_out (int) : final output size (output of the third linear layer)
    """

    layer1 = nn.Linear(nS,hidden_dim1)
    layer2 = nn.Linear(l2_in,hidden_dim2)
    layer3 = nn.Linear(hidden_dim2,l3_out)
    layer5= nn.Dropout(p=0.5)
    layer6= nn.Dropout(p=0.5)
    
    return layer1, layer2,layer3,layer5,layer6
      

class Actor(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self,nS,nA,seed,hidden_dim1=400,hidden_dim2=300):
        super(Actor, self).__init__()
               
        self.fc1, self.fc2,self.fc3 ,self.drop1,self.drop2= linear_layers(nS,hidden_dim1,hidden_dim2,hidden_dim1,nA)
        
        self.reset_parameters(seed)
        
    def reset_parameters(self,seed):                
        """Weights initialization for each linear layer """
        torch.manual_seed(seed)
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)    
        
    def forward(self,state):
        """Build an actor (policy) network that maps states -> actions."""
        x=F.relu(self.fc1(state))
        x=self.drop2(x)
        x=F.relu(self.fc2(x))
        x=self.drop1(x)
        
        return torch.tanh(self.fc3(x))
    
class Critic(nn.Module):
    """Critic (Value) Model."""
    
    def __init__(self,nS,nA,seed,hidden_dim1=400,hidden_dim2=300):
        super(Critic, self).__init__()
    
        self.fc1, self.fc2,self.fc3,self.drop1,self.drop2 = linear_layers(nS,hidden_dim1,hidden_dim2,hidden_dim1+nA,1)
                
        self.reset_parameters(seed)
        
    def reset_parameters(self,seed):
        """Weights initialization for each linear layer """
        torch.manual_seed(seed)
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)    
        
    def forward(self,state,action):
        """Build an critic (value) network that maps (states,actions) pairs to Q-values."""
        
        
        x = F.relu(self.fc1(state))
        x = torch.cat((x, action),dim=1)      
        x = F.relu(self.fc2(x))
        x=self.drop2(x)
        return self.fc3(x)   
    


    