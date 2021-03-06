# Udacity-Deep-Reinforcement-Learning-p3-collaboration-and-competition - Tennis environemnt

## The solution
This tennis environment can be seen as an extension of the reacher environment where 20 agents were trained all together. In this case we have to train 2 agents.
A good place to start is therefore the DDPG algorithm with 2 agents. 

### Initial experiments
First model
1. Two DDPG agents with the critics only getting their respective agents obervations.
2. Shared Replay buffer from which each critic sample experiences.
3. Actor Critic model with 2 hidden layers 400,300 dim each.
4. local and target models for both actor and critic with soft updates
5. Ornstein-Uhlenbeck noise with mu=0, theta=.15 and sigma=0.5

Second model
MADDPG algorithm
1. 2 DDPG agents with critics receving the full observation set
2. Shared replay buffer from which each critic sample experiences.
3. Actor Critic model with 2 hidden layers 400,300 dim each with drop out layers
4. Local and target models for both actor and critic with soft updates
5. Ornstein-Uhlenbeck noise with mu=0, theta=.15 and sigma=0.5

Wit the both the above models, I couldn't train the model succesfully. The avg score over 100 consecutive episodes would increase to max 0.01 and then decrease. The cycle would just continue for greater number of episodes.

### final implementation
Intuitively, it felt that the agents needed to learn from the good actions (those rewarded positively) more frequently, and PER (prioritised replay buffer) is just perfect for that.

### Prioritised Replay buffer
The original replay buffer allows us to perform uniform sampling with no preference for better samples. Prioritised replay buffer allows us to assign priorities based on the amount of learning that is possible from a sample. The sample with higher learning scope (TD error in this case)gets a higher priority.

In order to order the samples as per priority and keep updating the tree with new priorities, we use **Sum Trees**. I used the sumtree.py from rlcode.
A sideeffect of this priority based samples is the bias added towards these samples. The samples with lower priority(Error) may not be sampled for a long time. To overcome this,  we update the network weights are based on not just the TD error(gradient) but also the importance sampling weight for the experiences.

Using the PER along with the 2 DDPG agents i could solve the environment in 1677 episodes.
![](images/avg_scores_graph.png)

### Delayed updates
The actor and critic networks learn every 10 timesteps and every learning step is repeated 5 times with 5 differnt samples.
Once the actor and critic networks being trained are updated, the target networks are updated with a soft update.

### The Actor Critic Models
1. Two Hidden layers with dimensions 400, 300 respectively and dropout layers with p=0.5
2. The Actors take in its own observations with input size of 24 and ouputs an action with size =2
3. The Critics take in their actor's observation and the actor's action making a total input of 24+2. The critics output a Qvalue with size =1

### Hyperparameters
1. buffer = 100000
2. batch_size = 128
3. tau = 0.02
4. lr_actor = 1e-4
5. lr_critic = 1e-4
6. w_decay = 0
7. update_every = 10 , 5 times
8. gamma = 0.99







