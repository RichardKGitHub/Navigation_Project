#!/usr/bin/env python
# coding: utf-8

"""

COPYRIGHT NOTICE

This File contains Code provided by Udacity to be used by its students in order to solve the given projects.
You need to ask Udacity Inc. for permission in case you like to use that sections in your Software.
The functions and classes concerned are indicated by a comment.
Link to Udacity: https://eu.udacity.com/
"""

from unityagents import UnityEnvironment
import numpy as np

import torch                        # for Network
import torch.nn as nn               # for Network
import torch.nn.functional as F     # for Network

import random                                  # for Agent and ReplayBuffer
from collections import namedtuple, deque      # for Agent and ReplayBuffer
import torch.optim as optim                    # for Agent and ReplayBuffer

import matplotlib.pyplot as plt                # for plot_scores()

import argparse
import commentjson
import datetime


def get_infos(infos):
    """"""
    '''
    this Function may contain Code provided by Udacity Inc.
    '''
    next_state = infos.vector_observations[0]
    reward = infos.rewards[0]
    done = infos.local_done[0]
    return next_state, reward, done


def plot_scores(scores, epsilones=False):
    """"""
    '''
    this Function may contain Code provided by Udacity Inc.
    '''
    # create figure
    fig = plt.figure()
    if epsilones is not False:
        fig.add_subplot(212)
        plt.plot(np.arange(len(epsilones)), epsilones)
        plt.ylabel('epsilon')
        plt.xlabel('Episode #')
        fig.add_subplot(211)
    else:
        fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    if config.save_plot:
        # save the plot
        plt.savefig(config.path_save + "graph_" + config.save_indices + ".png")
    if config.show_plot:
        # plot the scores
        plt.show()


class Interaction:
    """defines looped interactions of the Agent (use case and training)"""

    def __init__(self, config_data_interact):
        self.load_indices = config_data_interact['load_indices']
        self.save_indices = config_data_interact['save_indices']
        self.path_load = config_data_interact['path_load']
        self.path_save = config_data_interact['path_save']
        self.save_weights = config_data_interact['save_weights']
        self.save_plot = config_data_interact['save_plot']
        self.show_plot = config_data_interact['show_plot']
        self.network_type = config_data_interact['network_type']
        self.episodes_train = config_data_interact['episodes_train']
        self.episodes_test = config_data_interact['episodes_test']
        self.end_training_score = config_data_interact['end_training_score']
        self.epsilon_start = config_data_interact['epsilon_start']
        self.epsilon_end = config_data_interact['epsilon_end']
        self.epsilon_decay = config_data_interact['epsilon_decay']
        self.epsilon_test = config_data_interact['epsilon_test']
        self.buffer_size = config_data_interact['buffer_size']
        self.batch_size = config_data_interact['batch_size']
        self.gamma = config_data_interact['gamma']
        self.tau = config_data_interact['tau']
        self.learning_rate = config_data_interact['learning_rate']
        self.update_target_every = config_data_interact['update_target_every']
        self.environment_path = config_data_interact['environment_path']

    def init_agent(self):
        if self.network_type == "QNetwork":
            agent_ = Agent(state_size=37, action_size=4, seed=0)
        elif self.network_type == "DoubleQNetwork":
            agent_ = AgentDoubleQ(state_size=37, action_size=4, seed=0)
        else:
            raise MyAppLookupError(f"No valid network_type specified | given: \"{self.network_type}\" | expected: "
                                   f"\"QNetwork\" or \"DoubleQNetwork\"")
        return agent_

    def train(self):
        """"""
        '''
        this Function may contain Code provided by Udacity Inc.
        '''
        time_new = time_start = datetime.datetime.now()
        score = 0
        scores = []
        epsilones = []
        solved = False
        scores_window = deque(maxlen=100)   # last 100 scores
        epsilon = self.epsilon_start
        for i_episode in range(self.episodes_train):
            env_info = env.reset(train_mode=True)[brain_name]
            state, _, _ = get_infos(env_info)
            while True:
                action = agent.act(state, epsilon)
                env_info = env.step(action)[brain_name]
                next_state, reward, done = get_infos(env_info)
                agent.step(state, action, reward, next_state, done)
                score += reward
                state = next_state
                if done:                    # if done = True
                    env.reset()
                    break
            scores_window.append(score)
            scores.append(score)
            epsilones.append(epsilon)
            if (i_episode + 1) % 25 == 0:
                time_old = time_new
                time_new = datetime.datetime.now()
                print('\rMin_Score {}\tAverage_Score: {:.2f}\tMax_Score {}\tEpisode {}/{}\tTime since start: {}'
                      '\tdeltaTime: {}'.format(np.min(scores_window), np.mean(scores_window), np.max(scores_window),
                                               i_episode, self.episodes_train-1, str(time_new-time_start).split('.')[0],
                                               str(time_new-time_old).split('.')[0]), end="")
            if np.mean(scores_window) >= self.end_training_score and solved is False:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                                                                       i_episode - 100, np.mean(scores_window)),
                                                                       end="\n\n")
                if self.save_weights:
                    torch.save(agent.qnetwork_local.state_dict(), self.path_save + "checkpoint_s_" +
                               self.save_indices + ".pth")
                solved = True
                # break
            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_end)
            score = 0
        if self.save_weights:
            torch.save(agent.qnetwork_local.state_dict(), self.path_save + "checkpoint_g_" + self.save_indices + ".pth")
        env.close()
        print(f'\n')
        plot_scores(scores, epsilones)
        return None

    def test(self):
        """"""
        '''
        this Function may contain Code provided by Udacity Inc.
        '''
        agent.qnetwork_local.load_state_dict(torch.load(self.path_load + "checkpoint_" + self.load_indices + ".pth"))
        time_new = time_start = datetime.datetime.now()
        score = 0
        scores = []
        scores_window = deque(maxlen=100)   # last 100 scores

        for i_episode in range(self.episodes_test):
            env_info = env.reset(train_mode=True)[brain_name]
            state, _, _ = get_infos(env_info)
            while True:
                action = agent.act(state, self.epsilon_test)
                env_info = env.step(action)[brain_name]
                next_state, reward, done = get_infos(env_info)
                score += reward
                state = next_state
                if done:
                    env.reset()
                    break
            scores_window.append(score)
            scores.append(score)
            if (i_episode + 1) % 25 == 0:
                time_old = time_new
                time_new = datetime.datetime.now()
                print('\rMin_Score {}\tAverage_Score: {:.2f}\tMax_Score {}\tEpisode {}/{}\tTime since start: {}'
                      '\tdeltaTime: {}'.format(np.min(scores_window), np.mean(scores_window), np.max(scores_window),
                                               i_episode, self.episodes_test-1, str(time_new-time_start).split('.')[0],
                                               str(time_new-time_old).split('.')[0]), end="")
            score = 0
        env.close()
        print("\n")
        plot_scores(scores)
        return None


class QNetwork(nn.Module):
    """Actor (Policy) Model."""
    ''' 
    this class was provided by Udacity Inc.
    '''

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DoubleQNetwork(nn.Module):
    """Actor (Policy) Model."""
    ''' 
    this class was provided by Udacity Inc.
    '''
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(DoubleQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Agent:
    """Interacts with and learns from the environment."""
    ''' 
    this class was provided by Udacity Inc.
    '''
    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=config.learning_rate)

        # Replay memory
        self.memory = ReplayBuffer(action_size, config.buffer_size, config.batch_size, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % config.update_target_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > config.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, config.gamma)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, config.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    ''' 
    this class was provided by Udacity Inc.
    '''
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class AgentDoubleQ:
    """"""
    '''
    this class contains some changes but was mainly provided by Udacity Inc.
    '''
    def __init__(self, state_size, action_size, seed):
        """Initialize an double DQN Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = DoubleQNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = DoubleQNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=config.learning_rate)

        # Replay memory
        self.memory = ReplayBuffer(action_size, config.buffer_size, config.batch_size, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % config.update_target_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > config.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, config.gamma)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # choose next action with current weights from qnetwork_local
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(states)
        self.qnetwork_local.train()
        actions_local_max = action_values.detach().argmax(1).unsqueeze(1)

        # Get predicted Q values (for next states) from target model with max action from local model
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, actions_local_max)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, config.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class MyAppLookupError(LookupError):
    """raise this when there's a lookup error for my app"""
    # source of this class:
    # https://stackoverflow.com/questions/2052390/manually-raising-throwing-an-exception-in-python/24065533#24065533


if __name__ == "__main__":
    # Idea of parser: https://docs.python.org/2/howto/argparse.html
    parser = argparse.ArgumentParser(description='Interacting Agent')
    parser.add_argument('--train', type=str, default='False', help='True: train the agent; '
                        'default=False: test the agent')
    parser.add_argument('--config_file', type=str, default='config.json',
                        help='Name of config_file in root of Navigation')
    args = parser.parse_args()

    # convert to bool since argparse doesn't treat booleans as expected:
    if args.train == 'True' or args.train == 'true' or args.train == 'TRUE':
        train = True
    elif args.train == 'False' or args.train == 'false' or args.train == 'FALSE':
        train = False
    else:
        raise MyAppLookupError('--train can only be True or False | default: False')

    # load config_file.json
    with open(args.config_file, 'r') as f:
        config_data = commentjson.load(f)

    # initialize configuration
    config = Interaction(config_data)

    '''
    from here on this function may contain some Code provided by Udacity Inc.
    '''
    # check device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # initialize Environment
    env = UnityEnvironment(file_name=config.environment_path)
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # initialize Agent
    agent = config.init_agent()

    # train or test the Agent
    if train is True:
        print(f"\nTrain the <{config.network_type}> Agent using config_file <{args.config_file}> on device <{device}> "
              f"with weights-save-index <{config.save_indices}>")
        config.train()
    else:
        print(f"\nTest the <{config.network_type}> Agent with fixed weights from <checkpoint_{config.load_indices}.pth> and"
              f" fixed epsilon={config.epsilon_test} using config_file <{args.config_file}> on device <{device}>")
        config.test()
