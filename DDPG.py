#ddpgagent
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import numpy as np
import random
from collections import namedtuple, deque
from model import Actor, Critic


from configparser import ConfigParser 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

agent_conf = ConfigParser() 
agent_conf.read('config.ini')

Buffer = int(agent_conf.get('agent','buffer'))
Batch_size = int(agent_conf.get('agent','batch_size'))
Tau = float(agent_conf.get('agent','tau'))
lr_actor =  float(agent_conf.get('agent','lr_actor'))
lr_critic =  float(agent_conf.get('agent','lr_critic'))
w_decay = float(agent_conf.get('agent','w_decay'))
update_every = int(agent_conf.get('agent','update_every'))
Gamma = float(agent_conf.get('agent','gamma'))



class AgentDDPG():
    """Interacts with and learns from the environment"""
        
    def __init__(self,state_size,action_size,gamma, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int) : number of agents
            gamma (float) : discount factor
            random_seed (int): random seed
        """
        self.actor_local = Actor(state_size,action_size,random_seed).to(device)
        self.actor_target = Actor(state_size,action_size,random_seed).to(device)
        self.actor_optim = optim.Adam(self.actor_local.parameters(),lr=lr_actor)
        
        self.critic_local=Critic(state_size,action_size,random_seed).to(device)
        self.critic_target=Critic(state_size,action_size,random_seed).to(device)
        self.critic_optim=optim.Adam(self.critic_local.parameters(),lr=lr_critic,weight_decay=w_decay)
        
        self.noise = OUNoise(action_size,random_seed)
        self.gamma=gamma
    
    def reset(self):
        self.noise.reset()  
        
    
    def get_action(self,state,noise=0.0):
        """Returns the action an agent takes as per the current policy given a state"""
        self.actor_local.eval()
        with torch.no_grad():
                    action = self.actor_local(state.view(1,-1)).cpu().data.numpy()
                     
        self.actor_local.train()
        action = np.squeeze(action)
        action += (self.noise.sample()*noise)
        return np.clip(action,-1,1)
    
    def get_trgetaction(self,states):
        """Returns the action an agent takes as per the old policy - used for evaluation"""
        self.actor_target.eval()
        action = self.actor_target(states.to(device)).cpu().data.numpy()
        self.actor_target.train()
        return np.clip(action,-1,1)
          
  
    def step(self,state,action,reward,next_state,done):
        """Calculates the TD error for a s,a,r,s' tuple and returns the absolute value of the error """
        err = 0.0
        
        action=torch.from_numpy(action).float().to(device)
        state=torch.from_numpy(state).float().to(device)
        next_state=torch.from_numpy(next_state).float().to(device)
        next_state=next_state.view(1,-1)
        
        next_action = self.actor_target(next_state)
        next_action=next_action.view(1,-1)
        
        self.critic_target.eval()
        self.critic_local.eval()
        pred = self.critic_target(next_state,next_action)
        Q_targets= reward + self.gamma * pred * (1 -done)
            
        with torch.no_grad():
            Q_expected = self.critic_local(state.view(1,-1),action.view(1,-1))
        error = abs(Q_targets - Q_expected).detach().data
        self.critic_local.train()
        self.critic_target.train()
        return error    
                           
        
     
    def learn(self,experiences,is_weights):
        """update the critic network using the loss calculated as 
           Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
            
         The critic loss is multiplies by the importance sampling factor to make sure all samples are considered during sampling and not the 
         ones that appear more frequently   
            
          update the actor network using the loss calculated as
          actor loss = - critic_target(state,action)
        """
               
        states, actions,rewards,next_states,dones = experiences
        next_actions = self.actor_target(next_states)
                
        pred = self.critic_target(next_states,next_actions)
        Q_targets= rewards + self.gamma * pred * (1 - dones)
        Q_expected = self.critic_local(states,actions)
        
        #critic loss
        errors = abs(Q_expected - Q_targets)
        
        self.critic_optim.zero_grad()
        critic_loss = (torch.FloatTensor(is_weights) * F.mse_loss(Q_expected, Q_targets)).mean()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optim.step()  
      
        # update actor
        actions_actor = self.actor_local(states)
        loss = -self.critic_local(states,actions_actor).mean()
        
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()
        
        return errors
        
    def update_networks(self,Tau):
        """update weights of actor critic local network
        
        """        
        self.soft_update(self.critic_local,self.critic_target,Tau)
        self.soft_update(self.actor_local,self.actor_target,Tau)
   
        
    def soft_update(self,local_model,target_model,Tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        
        for local_params, target_params in zip(local_model.parameters(),target_model.parameters()):
            target_params.data.copy_(local_params*Tau + (1-Tau)*target_params)
    

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size,seed, mu=0., theta=0.15, sigma=0.5):
        """Initialize parameters and noise process.
        
        Params:
        =======
             size (int or tuple): sample space 
             mu (float): mean
             theta (float):optimal parameter
             sigma (float) :variance
        """       
        
        self.size=size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()
        np.random.seed(seed)

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
                
        self.state = x + dx
        return self.state
    

 