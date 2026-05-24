import numpy as np
import torch
from elevator_system.environment import ElevatorSystemEnv
from elevator_system.config import *
from dqn_agent import DQNAgent
import os

# --- GPU CONFIGURATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPUs detected: {torch.cuda.device_count()}.")
else:
    print("No GPUs detected. Running on CPU.")
        
# Training Settings
EPISODES = 100
TARGET_UPDATE_FREQ = 10  # Update target network every X episodes

def run_training():
    env = ElevatorSystemEnv()
    
    # Calculate state dimension based on config
    # 2 features per elevator + 2 features per floor
    state_dim = (2 * num_elevators) + (2 * num_floors)
    
    agent = DQNAgent(state_dim, num_elevators)
    
    print(f"--- STARTING TRAINING ---")
    print(f"Elevators: {num_elevators}, Floors: {num_floors}")
    print(f"State Dim: {state_dim}")

    for e in range(EPISODES):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # 1. Select Action
            actions = agent.act(state)
            
            # 2. Step Environment
            next_state, reward, done, info = env.step(actions)
            
            # 3. Store Experience
            agent.remember(state, actions, reward, next_state, done)
            
            # 4. Train Agent
            agent.replay()
            
            state = next_state
            total_reward += reward

        # End of Episode
        if (e + 1) % TARGET_UPDATE_FREQ == 0:
            agent.update_target_model()
            print(f"Episode {e+1}/{EPISODES} | Reward: {total_reward:.1f} | Epsilon: {agent.epsilon:.2f} | Arrived: {info.get('total_arrived', 0)}")
        else:
            print(f"Episode {e+1}/{EPISODES} | Reward: {total_reward:.1f}", end='\r')

    # Save final model (updated to standard PyTorch .pth extension)
    agent.save(f"elevator_dqn_{num_elevators}_{num_floors}.weights.pth")
    print("\nTraining Complete. Model saved.")

if __name__ == "__main__":
    run_training()