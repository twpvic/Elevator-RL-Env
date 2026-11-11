import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class DQNNetwork(nn.Module):
    def __init__(self, state_dim, output_dim):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Linear activation output

class DQNAgent:
    # epsilon_decay is applied once per EPISODE (see decay_epsilon), so pick it
    # from the episode budget: reaching epsilon_min takes
    # log(epsilon_min) / log(epsilon_decay) episodes. The default of 0.95
    # bottoms out at roughly episode 90, suiting the 100-episode training run.
    def __init__(self, state_dim, num_elevators, action_dim=5,
                 gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.95,
                 learning_rate=0.001, batch_size=32, memory_size=20000):
        self.state_dim = state_dim
        self.num_elevators = num_elevators
        self.action_dim = action_dim
        
        # Hyperparameters
        self.gamma = gamma          # Discount factor
        self.epsilon = epsilon      # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size
        
        # Device configuration (GPU/CUDA support)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Replay Buffer
        self.memory = deque(maxlen=self.memory_size)
        
        # Models
        output_dim = self.num_elevators * self.action_dim
        self.model = DQNNetwork(self.state_dim, output_dim).to(self.device)
        self.target_model = DQNNetwork(self.state_dim, output_dim).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        self.update_target_model()

    def update_target_model(self):
        """Copies weights from main model to target model."""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, actions, rewards, next_state, terminated):
        """
        Stores experience in replay buffer.

        `rewards` is one reward per elevator (the env's info['rewards']), so
        each car is trained on what it actually did rather than on the whole
        building's total.

        `terminated` must be the env's terminated flag, NOT `done`. An episode
        that merely ran out of time is still ongoing as far as the value
        function is concerned; treating that as terminal teaches the agent that
        returns collapse at the horizon.
        """
        self.memory.append((state, actions, rewards, next_state, terminated))

    def act(self, state):
        """Selects actions for all elevators (Epsilon-Greedy)."""
        # Exploration
        if np.random.rand() <= self.epsilon:
            return [random.randint(0, self.action_dim - 1) for _ in range(self.num_elevators)]
        
        # Exploitation
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            q_values_flat = self.model(state_tensor).cpu().numpy()[0]
        self.model.train()
        
        # Reshape to (num_elevators, 5) to get best action for each elevator
        q_values = q_values_flat.reshape(self.num_elevators, self.action_dim)
        actions = np.argmax(q_values, axis=1)
        
        return list(actions)

    def replay(self):
        """Trains the network using a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, terminated = zip(*minibatch)

        # Everything below works on (batch, elevator, action) tensors and stays
        # on-device. The previous per-sample, per-elevator Python loop with
        # GPU->CPU->GPU numpy round-trips ran on every environment step and
        # dominated training time.
        b, e, a = self.batch_size, self.num_elevators, self.action_dim

        states = torch.as_tensor(np.asarray(states, dtype=np.float32), device=self.device)
        next_states = torch.as_tensor(np.asarray(next_states, dtype=np.float32), device=self.device)
        actions = torch.as_tensor(np.asarray(actions, dtype=np.int64), device=self.device)
        rewards = torch.as_tensor(np.asarray(rewards, dtype=np.float32), device=self.device)
        terminated = torch.as_tensor(np.asarray(terminated, dtype=np.float32), device=self.device)

        # 1. Target per elevator: R + gamma * max_a' Q_target(s', a')
        #    Bootstrapping is only cut off on a true terminal state, so an
        #    episode that just hit the time limit still bootstraps.
        with torch.no_grad():
            max_next_q = self.target_model(next_states).view(b, e, a).max(dim=2).values
            expected_q = rewards + self.gamma * max_next_q * (1.0 - terminated).unsqueeze(1)

        # 2. Q-value of the action each elevator actually took
        current_q = (self.model(states)
                     .view(b, e, a)
                     .gather(2, actions.unsqueeze(2))
                     .squeeze(2))

        # 3. Train the model
        loss = self.criterion(current_q, expected_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        """
        Decays exploration by one step. Call this once per episode.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def load(self, name):
        """
        Loads weights onto whichever device this agent is using.
        """
        self.model.load_state_dict(
            torch.load(name, map_location=self.device, weights_only=True))
        # Otherwise the target network keeps its random initialisation, which
        # is invisible during evaluation but poisons any resumed training.
        self.update_target_model()

    def save(self, name):
        torch.save(self.model.state_dict(), name)