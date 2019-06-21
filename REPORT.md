##Learning Algorithm 
The Project was solved by too learning algorithms:
- Fixed Q-Targets DQN 
- double DQN

The networks of both algorithms are identically constructed
####Network architecture
- input layer: 37 Neurones for 37 states
- first hidden layer: 64 Neurones   |   activation function: Rectified Linear Unit (ReLU) 
- second hidden layer: 64 Neurones   |   activation function: Rectified Linear Unit (ReLU) 
- output layer: 4 Neurones for the 4 q Values corresponding to the 4 actions
####Hyperparameters
- both algorithms use the same parameters:
  - maximal Number of episodes `if --train==True` (network gets trained): 1000
  - Number of episodes `if --train==False` (network gets tested): 500 
  - epsilon_start: 1.0
  - epsilon_end: 0.01
  - epsilon_decay: 0.995
  - epsilon during test mode: 0.01
  - replay buffer size: 100000
  - minibatch size": 64
  - discount factor gamma: 0.995
  - tau: 1e-3 (for soft update of target parameters)
  - learning_rate: 5e-4
  - the target_network gets updated every 4 Steps
####Fixed Q-Targets DQN
- two neural networks:
  - local_network: network that gets trained 
    - action gets determined through this network   
  - target_network: network derived from local_network 
    - updated via soft_update: 
      ```
      target_param_new = tau * copy(local_param) + (1.0 - tau) * target_param
      ```
  
- calculation of the temporal difference:
  - `q(S',a,w-)` gets determent from the target_network with weights `w-` for the following state `S'`
  - `q(S,A,w)` gets determent from the local_network with weights `w` for the current state `S` and the action `A`
    ```
    temporal_difference = reward + gamma*max(q(S',a,w-)) - q(S,A,w)
    ```
####double DQN
- like Fixed Q-Targets DQN
- the difference to the Fixed Q-Targets DQN lays in the calculation of the temporal_difference:
  - the action `a` for the following state `S'` is determent from the local_network (epsilon-greedy)
  - the `q` Value for the following state `S'` is determent from the target_network with action `a` from the local_network
  - the `q` Value for the current state is determent from the local_network `q=q(S,A,w)`
    ```
    temporal_difference = reward + gamma*q(S',max(a(S',w)),w-) - q(S,A,w)
    ```

##Plot of Rewards
####FixedQ-Targets DQN
 - task solved in 428 episodes
 
 ![alt text](/home/user/data_github/Udacity/ReinforcementLearning/Navigation_Project/archive/graph_07.png "rewards Fixed Q-Targets DQN ")
 
####double DQN
 - task solved in 455 episodes
  
![alt text](/home/user/data_github/Udacity/ReinforcementLearning/Navigation_Project/archive/graph_06.png "rewards double DQN ")

##Ideas for Future Work
- In the next step, the parameters for both networks and algorithms will be further adjusted to see if the task can be solved in fewer episodes.
- The implementation of a dueling DQN, prioritized experience DQN and the combination of all architectures are further steps for the future.