import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import namedtuple, deque

# Define the structure of a single experience transition
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    """A cyclic buffer of bounded size that holds the transitions observed recently."""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Sample a batch of transitions from memory."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class DQN(nn.Module):
    """Deep Q-Network model."""
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        """Forward pass through the network."""
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DQNAgent:
    """Deep Q-Network Agent that interacts with and learns from the environment."""
    def __init__(self, n_observations, n_actions):
        self.n_observations = n_observations
        self.n_actions = n_actions
        
        # Hyperparameters
        self.batch_size = 128
        self.gamma = 0.99
        self.epsilon = 1.0        # start fully exploring
        self.epsilon_min = 0.05   # don’t go below this
        self.epsilon_decay = 0.99995  # very slow exponential decay
        self.learning_rate = 1e-4
        self.tau = 0.005 # For soft update of target network

        # Initialize policy and target networks
        self.policy_net = DQN(n_observations, n_actions)
        self.target_net = DQN(n_observations, n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target network is not trained directly

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate, amsgrad=True)
        self.memory = ReplayMemory(10000)

        self.steps_done = 0

    def select_action(self, state):
        """Selects an action using an epsilon-greedy policy."""
        sample = random.random()
        # Decay epsilon over time
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        if sample > self.epsilon:
            # Exploitation: choose the best action from the policy network
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            # Exploration: choose a random action
            return torch.tensor([[random.randrange(self.n_actions)]], dtype=torch.long)

    def learn(self):
        """Train the policy network using a batch of experiences from replay memory."""
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Create tensors for each part of the transition
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        done_batch = torch.cat(batch.done)
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for next_state_batch are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the done flag.
        next_state_values = torch.zeros(self.batch_size)
        non_final_mask = ~done_batch
        next_state_values[non_final_mask] = self.target_net(next_state_batch[non_final_mask]).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update_target_net(self):
        """
        Soft update of the target network's weights:
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)
