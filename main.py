import time
import random
import numpy as np
from controller import TraditionalController
from dqn_agent import DQNAgent
from elevator_system.environment import ElevatorSystemEnv
from elevator_system.config import *

# --- CONFIGURATION ---
# Set to 'random' for random actions, 'controller' for the smart algorithm, 
# or 'model' for the TensorFlow agent.
MODE = 'controller'

def get_random_actions(num_elevators):
    """Returns a list of random actions."""
    return [random.choice([action_up, action_down, action_board, action_alight, action_nothing]) 
            for _ in range(num_elevators)]


def run_simulation(episodes=1):
    env = ElevatorSystemEnv()
    
    # Initialize the logic controller
    controller = TraditionalController(num_elevators, num_floors)
    
    # Initialize the DQN agent
    if MODE == 'model':
        state_dim = (2 * num_elevators) + (2 * num_floors)
        agent = DQNAgent(state_dim, num_elevators)
        
        try:
            agent.load(f"elevator_dqn_{num_elevators}_{num_floors}.weights.h5")
            print("Loaded pre-trained model weights.")

        except:
            print("No pre-trained model found.")
        agent.epsilon = 0.0  # Disable exploration for evaluation

    print(f"--- STARTING SIMULATION (Mode: {MODE}) ---")
    print(f"Params: {num_elevators} Elevators, {num_floors} Floors")

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        print(f"\nEpisode {ep + 1} Start...")
        
        while not done:
            # 1. Determine Actions based on Mode
            if MODE == 'controller':
                actions = controller.get_actions(state)
                
            elif MODE == 'random':
                actions = get_random_actions(num_elevators)
                
            elif MODE == 'model':
                actions = agent.act(state)
            
            # 2. Step Environment
            next_state, reward, done, info = env.step(actions)
            total_reward += reward
            state = next_state
            
            # Optional: Slow down to watch print outputs
            # time.sleep(0.05) 

        # --- EPISODE STATS ---
        print("-" * 40)
        print(f"EPISODE {ep + 1} FINISHED")
        print(f"Total Reward:        {total_reward:.1f}")
        print(f"Passengers Arrived:  {info.get('total_arrived', 0)}")
        print(f"Average Wait Time:   {info.get('avg_wait_time', 0):.2f} steps")
        print(f"Kuramoto Bunching:   {env.bunching_history[-1]:.5f}")
        print("-" * 40)

if __name__ == "__main__":
    run_simulation()
