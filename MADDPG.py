import torch
import numpy as np
import random
from configparser import ConfigParser
from collections import namedtuple, deque
from DDPG import AgentDDPG


m_conf = ConfigParser() 
m_conf.read('config.ini')
random_seed=int(m_conf.get('train','random_seed'))
Buffer = int(m_conf.get('agent','buffer'))
Batch_size = int(m_conf.get('agent','batch_size'))
Tau = float(m_conf.get('agent','tau'))
update_every = float(m_conf.get('agent','update_every'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(random_seed)
np.random.seed(random_seed)

def convert_to_tensor(anylist,isbool=False):
    """" convert a list of int/float to a flaot tensor """
    if isbool:
        return torch.from_numpy(np.vstack(anylist).astype(np.uint8)).float().to(device)
    else:
        return torch.from_numpy(np.vstack(anylist)).float().to(device)


class MADDPG():
    def __init__(self,state_size,action_size,num_agents,Gamma):
        
        self.agents = [AgentDDPG(state_size,action_size,Gamma,random_seed) for _ in range(num_agents)]
        
        self.num_agents=num_agents
        self.replaybuff = ReplayBuffer(Buffer,state_size,Batch_size,random_seed)
        self.batch_size = Batch_size
        self.gamma=Gamma
   
                
        
    def get_action(self,states,noise=0.0):
        """Returns action that an agent can take in a given state as per current policy."""
        states  = torch.from_numpy(states).float().to(device)     
        actions=[agent.get_action(state,noise) for agent,state in zip(self.agents,states)]
        
        return np.array(actions)
       
      
    def step(self,states,actions,rewards,next_states,dones,ts):
        """Save experience in replay memory prioritised by the TD error"""
        
        for state,action,reward,next_state,done in zip(states,actions,rewards,next_states,dones):
             self.replaybuff.add(state, action, reward, next_state, done)
        if  len(self.replaybuff) >Batch_size  and ts % update_every ==0:
                self.learn()        
            
        
    def learn(self):
        """Sample from the replay buffer and learn"""
        for _ in range(5):
            experiences = self.replaybuff.sample()
            for agent in self.agents:
                agent.learn(experiences)    
                   
        

        
               
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    
    def __init__(self,size,sA,batch,seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        
        self.dq=deque(maxlen=size)
        self.batch=batch
        self.action_size=sA
        self.experience=namedtuple("Experience",field_names=["state","action","reward","next_state","done"])
        np.random.seed(seed)
        
    def add(self,state,action,reward,next_state,done):
        """Add a new experience to memory."""
        experience=self.experience(state,action,reward,next_state,done)
        self.dq.append(experience)
        
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experience=random.sample(self.dq,k=self.batch)
        
        states = convert_to_tensor([e.state for e in experience if e is not None])
        actions = convert_to_tensor([e.action for e in experience if e is not None])
        rewards = convert_to_tensor([e.reward for e in experience if e is not None])
        next_states = convert_to_tensor([e.next_state for e in experience if e is not None])
        dones = convert_to_tensor([e.done for e in experience if e is not None],isbool=True)
        
        return states,actions,rewards,next_states,dones    
    
    def __len__(self):
        """returns the current length of the replay buffer"""
        return len(self.dq)    