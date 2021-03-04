# Udacity-Deep-Reinforcement-Learning-p3-collaboration-and-competition - Tennis environemnt

## The solution
### Initial experiments
Initial model
1. 2 DDPG agents with critics only getting its agents obervations.
2. Shared Replay  buffer
3. Actor Critic model with 2 hidden layers 400,300 dim each.
4. local and target models for both actor and critic with soft updates

Next model
MADDPG algorithm
1. 2 DDPG agents with critics receving the full observation set
2. Shared replay buffer
3. Actor Critic model with 2 hidden layers 400,300 dim each with drop out layers
4. Local and target models for both actor and critic with soft updates

wit the both the above models, i couldnt train the model succesfully. The avg score would increase to max 0.01 and then decrease. the cycle would just continue for further episodes.

### final implementation
Intuitively, it felt that the agents needed to learn from the good actions (those rewarded positively) more frequently, and PER (prioritised replay buffer) is just perfect for that.

About the PER

I used the already implemented sumtree.py from rlcode.
And used this sumtree to update s,a,r,s tuples in the replay buffer.

Hyperparameters






