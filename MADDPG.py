import torch
import numpy as np
import random
from configparser import ConfigParser
from collections import namedtuple, deque
from DDPG import AgentDDPG
from SumTree import SumTree 

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
        self.replaybuff = ReplayBuffer(Buffer)
        self.batch_size = Batch_size
        self.gamma=Gamma
    
                
        
    def get_action(self,states,noise=0.0):
        """Returns action that an agent can take in a given state as per current policy."""
        states  = torch.from_numpy(states).float().to(device)     
        actions=[agent.get_action(state,noise) for agent,state in zip(self.agents,states)]
        
        return np.array(actions)
       
      
    def step(self,states,actions,rewards,next_states,dones):
        """Save experience in replay memory prioritised by the TD error"""
        
        for j, agents in enumerate(self.agents):
            error = self.agents[j].step(states[j],actions[j],rewards[j],next_states[j],dones[j])
            self.replaybuff.add(error, states[j], actions[j], rewards[j], next_states[j], dones[j])
            
        
    def learn(self):
        """Sample from the prioritised replay buffer and update the entries with the new TD errors"""
        for _ in range(5):
            for agent in self.agents:
                
                mini_batch, idxs, is_weights = self.replaybuff.sample(self.batch_size)
              
                states = convert_to_tensor([e.states for e in mini_batch if e != 0])
                actions = convert_to_tensor([e.actions for e in mini_batch if e !=0])
                rewards = convert_to_tensor([e.rewards for e in mini_batch if e !=0])
                next_states = convert_to_tensor([e.next_states for e in mini_batch if e !=0])
                dones = convert_to_tensor([e.dones for e in mini_batch if e !=0],isbool=True)       
        
                experiences = states, actions, rewards, next_states, dones 
                   
                errors= agent.learn(experiences,is_weights).detach().numpy()
            
                for i in range(len(idxs)):
                    idx = idxs[i]
                    self.replaybuff.update(idx, errors[i])
                agent.update_networks(Tau)
            

        
               
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001
    
    def __init__(self,capacity):
        """Initialize a ReplayBuffer object.
        Params
        ======
            capacity (int): maximum size of buffer
        """
        
        # Making the tree 
        self.tree = SumTree(capacity)
        self.capacity=capacity
        self.experience=namedtuple("Experience",field_names=["states","actions","rewards","next_states","dones"])
    
    def _get_priority(self, error):
        """retrieve the priority of the sample from the error. Add a small value e for 0 error"""
        return (np.abs(error) + self.e) ** self.a 
    
    def add(self,error,states,actions,rewards,next_states,dones):
        """Add s,a,r,s' into memory"""
        experience=self.experience(states,actions,rewards,next_states,dones)
        
        p = self._get_priority(error)
        self.tree.add(p, experience)
        
        
    def sample(self, n):
        """ Sample from the memory by dividing the probability range by the batch size and selecting samples from each segment randomly"""

        batch = []
        idxs = []
        segment = self.tree.total() / n
        
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            if data == 0:
                continue
            priorities.append(p)
            batch.append(data)
            
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        #print (self.tree.n_entries, priorities, self.tree.total() )
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        """ update the error for the sample in the sum tree"""
        p = self._get_priority(error)
        self.tree.update(idx, p)            

    