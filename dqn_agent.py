import numpy as np
import random
import tensorflow as tf
from collections import deque

class DQNAgent:
    def __init__(self, state_dim, num_elevators, action_dim=5):
        self.state_dim = state_dim
        self.num_elevators = num_elevators
        self.action_dim = action_dim
        
        # Hyperparameters
        self.gamma = 0.99           # Discount factor
        self.epsilon = 1.0          # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.001
        self.batch_size = 32
        
        # Replay Buffer
        self.memory = deque(maxlen=20000)
        
        # Models
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        """
        Builds a Dense Neural Network.
        Input: State Vector (2*elevators + 2*floors)
        Output: Q-values for ALL elevators concatenated (num_elevators * 5 actions)
        """
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.state_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            # Output layer: Flattened Q-values for all elevators
            tf.keras.layers.Dense(self.num_elevators * self.action_dim, activation='linear')
        ])
        
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        """Copies weights from main model to target model."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, actions, reward, next_state, done):
        """Stores experience in replay buffer."""
        self.memory.append((state, actions, reward, next_state, done))

    def act(self, state):
        """Selects actions for all elevators (Epsilon-Greedy)."""
        # Exploration
        if np.random.rand() <= self.epsilon:
            return [random.randint(0, self.action_dim - 1) for _ in range(self.num_elevators)]
        
        # Exploitation
        state_tensor = np.array([state])  # Shape (1, state_dim)
        q_values_flat = self.model.predict(state_tensor, verbose=0)
        
        # Reshape to (num_elevators, 5) to get best action for each elevator
        q_values = q_values_flat.reshape(self.num_elevators, self.action_dim)
        actions = np.argmax(q_values, axis=1)
        
        return list(actions)

    def replay(self):
        """Trains the network using a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        
        states = np.array([t[0] for t in minibatch])
        next_states = np.array([t[3] for t in minibatch])
        
        with tf.device('/GPU:0'):
            # 1. Predict Q-values for current states (to get baseline for non-taken actions)
            target_qs = self.model.predict(states, verbose=0)
            
            # 2. Predict Q-values for next states (using Target Network)
            next_qs_flat = self.target_model.predict(next_states, verbose=0)
            
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
            
            # 4. Train the model
            self.model.fit(states, target_qs, epochs=1, verbose=0)
        
        # 5. Decay Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)