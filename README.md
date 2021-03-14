# Udacity-Deep-Reinforcement-Learning-p3-collaboration-and-competition
The goal of the project is to train the two players(agents) in the Tennis environment. The environment is based on Unity ML-agents.

![](images/tennis.png)


## The problem description

**The Environment:** In this environment, two agents control rackets to bounce a ball over a net. 

**Observation space:** The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

**The Rewards:** If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

**Task (Episodic/Continuous):** The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). 

**Solution:** The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.
After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores. This yields a single score for each episode.



## Getting Started - installation
  
1. Dependencies
To set up your python environment to run the code in this repository, follow the instructions below.

Create (and activate) a new environment with Python 3.6.

Linux or Mac:
conda create --name drlnd python=3.6
source activate drlnd
Windows:
conda create --name drlnd python=3.6 
activate drlnd
Follow the instructions in this repository to perform a minimal install of OpenAI gym.

Next, install the classic control environment group by following the instructions here.
Then, install the box2d environment group by following the instructions here.
Clone the repository (if you haven't already!), and navigate to the python/ folder. Then, install several dependencies.

git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
Create an IPython kernel for the drlnd environment.
python -m ipykernel install --user --name drlnd --display-name "drlnd"
Before running code in a notebook, change the kernel to match the drlnd environment by using the drop-down Kernel menu.
Kernel

2. Python packages
  a. numpy
  b. torchvision
  c. pandas
  d. matplotlib


3. For this project, you will not need to install Unity - this is because we have already built the environment for you, and you can download it from one of the links below. You need only select the environment that matches your operating system:

* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

## Execution
Clone the repository onto your system.
The configuration for the environement, the agent and the DDPG parameters are mentioned in the config file.
Execute the .ipynb file

