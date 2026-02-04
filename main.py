import time
import random
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

def get_model_actions(model, state, num_elevators):
    """Uses a trained TensorFlow model to predict actions."""
    # This function mimics the act() method from the DQNAgent
    state_tensor = np.array([state])
    q_values = model.predict(state_tensor, verbose=0)
    
    # Reshape: (1, num_elevators * 5) -> (num_elevators, 5)
    q_values_reshaped = q_values.reshape(num_elevators, 5)
    
    # Argmax to get best action index for each elevator
    actions = np.argmax(q_values_reshaped, axis=1)
    return list(actions)

def run_simulation(episodes=1):
    env = ElevatorSystemEnv()
    
    # Initialize the logic controller
    controller = TraditionalController(num_elevators, num_floors)
    
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
                # actions = get_model_actions(model, state, num_elevators)
                print("Model not loaded. Please uncomment model loading lines.")
                break
            
            # 2. Step Environment
            next_state, reward, done, info = env.step(actions)
            total_reward += reward
            state = next_state
            
            # Optional: Slow down to watch print outputs if you add print statements
            # time.sleep(0.05) 

        # 3. End of Episode Stats
        print("-" * 40)
        print(f"EPISODE {ep + 1} FINISHED")
        print(f"Total Reward:        {total_reward:.1f}")
        print(f"Passengers Arrived:  {info.get('total_arrived', 0)}")
        print(f"Average Wait Time:   {info.get('avg_wait_time', 0):.2f} steps")
        print("-" * 40)

if __name__ == "__main__":
    run_simulation()
