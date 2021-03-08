# Udacity-Deep-Reinforcement-Learning-p3-collaboration-and-competition - Tennis environemnt

## The solution
This tennis environment can be seen as an extension of the reacher environment where 20 agents were trained all together using DDPG. In this case we have to train 2 agents.
A good place to start is therefore the DDPG algorithm with 2 agents. 

1. Two DDPG agents each agent has an actor and a critic.
2. The Actors take in its own observations with input size of 24 and ouputs an action with size =2
3. The Critics take in their actor's observation and the actor's action making a total input of 24+2. The critics output a Qvalue with size =1
4. Shared Replay buffer from which each critic sample experiences.
5. Actor Critic model with 2 hidden layers 480,360 dim each, RELU activation function and dropout layers with p=0.5
6. local and target models for both actor and critic with soft updates
7. Ornstein-Uhlenbeck noise with mu=0, theta=.15 and sigma=0.2

### Hyperparameters
1. buffer = 100000
2. batch_size = 96
3. tau = 0.02
4. lr_actor = 1e-4
5. lr_critic = 1e-4
6. w_decay = 0
7. gamma = 0.99

The environment was solved in 323 episodes.
![](images/avg_scores_graph.png)


##Ideas for future work
1. Observe improvements with the Prioritised Replay Buffer
2. Try the MADDPG implementation with critics receiving the full observations and actions
3. Observe the improvements with the categorical distribution - C51 algorithm





