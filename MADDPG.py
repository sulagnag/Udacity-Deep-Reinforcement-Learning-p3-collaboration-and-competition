import torch
import numpy as np
import random
from configparser import ConfigParser
from collections import namedtuple, deque
from DDPG import AgentDDPG

m_conf = ConfigParser() 
m_conf.read('config.ini')

Buffer = int(m_conf.get('agent','buffer'))
Batch_size = int(m_conf.get('agent','batch_size'))
Tau = float(m_conf.get('agent','tau'))
update_every = float(m_conf.get('agent','update_every'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def convert_to_tensor(anylist,isbool=False):
    """" convert a list of int/float to a flaot tensor """
    if isbool:
        return torch.from_numpy(np.vstack(anylist).astype(np.uint8)).float().to(device)
    else:
        return torch.from_numpy(np.vstack(anylist)).float().to(device)


class MADDPG():
    def __init__(self,state_size,action_size,num_agents,Gamma,random_seed):
        
        self.agents = [AgentDDPG(state_size,action_size,Gamma,random_seed) for _ in range(num_agents)]
        np.random.seed(random_seed)
        self.num_agents=num_agents
        self.replaybuff = ReplayBuffer(Buffer,action_size,Batch_size,random_seed)
        self.t=0
    
                
        
    def get_action(self,states,noise):
        """Returns action that an agent can take in a given state as per current policy."""
        states  = torch.from_numpy(states).float().to(device)     
        actions=[agent.get_action(state,noise) for agent,state in zip(self.agents,states)]
        
        return np.array(actions)
       
    
    def get_taction(self,states):
        """Returns action that an agent can take in a given state as per current policy."""
        
        tactions=[]
                     
        for agent,s in zip(self.agents,np.hsplit(states,2)):
                tactions.append(agent.get_trgetaction(s))
                
        actions= np.concatenate((tactions[0],tactions[1]),axis=1)
        return actions
    
      
    def step(self,states,actions,rewards,next_states,dones,ts):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        
        states=np.concatenate(states).flat.copy()
        actions=np.concatenate(actions).flat.copy()
        #rewards=np.concatenate(rewards).flat
        next_states=np.concatenate(next_states).flat.copy()
        #dones=np.concatenate(dones).flat
        
                 
        self.replaybuff.add(states,actions,rewards,next_states,dones)
        
        if (len(self.replaybuff) >= Batch_size):
                experiences = self.replaybuff.sample()
                states,actions,rewards,next_states,dones=experiences
                next_actions = self.get_taction(next_states)
                
                for j,agent in enumerate(self.agents):
                    agent.learn(experiences,next_actions,j)
                    agent.update_networks(Tau)
                    agent.reset()
                   
        
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
        self.experience=namedtuple("Experience",field_names=["states","actions","rewards","next_states","dones"])
        np.random.seed(seed)
        
    def add(self,state,action,reward,next_state,done):
        """Add a new experience to memory."""
        experience=self.experience(state,action,reward,next_state,done)
        self.dq.append(experience)
        
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experience=random.sample(self.dq,k=self.batch)
     
        states = convert_to_tensor([e.states for e in experience if e is not None])
        actions = convert_to_tensor([e.actions for e in experience if e is not None])
        rewards = convert_to_tensor([e.rewards for e in experience if e is not None])
        next_states = convert_to_tensor([e.next_states for e in experience if e is not None])
        dones = convert_to_tensor([e.dones for e in experience if e is not None],isbool=True)
        
        return states,actions,rewards,next_states,dones    
    
    def __len__(self):
        """returns the current length of the replay buffer"""
        return len(self.dq)


    