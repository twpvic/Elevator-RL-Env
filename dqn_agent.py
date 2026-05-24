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
    def __init__(self, state_dim, num_elevators, action_dim=5,
                 gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.9995,
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

    def remember(self, state, actions, reward, next_state, done):
        """Stores experience in replay buffer."""
        self.memory.append((state, actions, reward, next_state, done))

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
        
        states = torch.FloatTensor(np.array([t[0] for t in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch])).to(self.device)
        
        # 1. Predict Q-values for current states (detached as baseline)
        with torch.no_grad():
            target_qs = self.model(states).cpu().numpy()
            # 2. Predict Q-values for next states (using Target Network)
            next_qs_flat = self.target_model(next_states).cpu().numpy()
        
        # 3. Update Q-values for the specific actions taken
        for i, (state, actions, reward, next_state, done) in enumerate(minibatch):
            
            # We must update the target value for EACH elevator's chosen action
            for elev_idx in range(self.num_elevators):
                action_taken = actions[elev_idx]
                
                # Calculate Target: R + gamma * max(Q_next)
                if done:
                    target = reward
                else:
                    # Slice the next_qs for this specific elevator to find its max Q
                    start = elev_idx * self.action_dim
                    end = start + self.action_dim
                    max_next_q = np.amax(next_qs_flat[i][start:end])
                    target = reward + self.gamma * max_next_q
                
                # Update the specific slot in the flat output vector
                flat_idx = (elev_idx * self.action_dim) + action_taken
                target_qs[i][flat_idx] = target
        
        # Convert updated targets back to tensor
        expected_q = torch.FloatTensor(target_qs).to(self.device)
        
        # 4. Train the model
        current_q = self.model(states)
        loss = self.criterion(current_q, expected_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 5. Decay Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name, weights_only=True))

    def save(self, name):
        torch.save(self.model.state_dict(), name)