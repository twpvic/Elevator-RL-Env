import random
import numpy as np
from controller import TraditionalController
from dqn_agent import DQNAgent
from elevator_system.environment import ElevatorSystemEnv
from elevator_system.config import *
from seeding import set_global_seeds

# --- CONFIGURATION ---
# Set to 'random' for random actions, 'controller' for the smart algorithm,
# or 'model' for the trained DQN agent.
MODE = 'model'
VALID_MODES = ('random', 'controller', 'model')

# Fixed so every mode is evaluated against the identical passenger sequence.
SEED = 7

# Point this at the .best.pth from training to evaluate the best-scoring
# episode's weights instead of whatever the final episode happened to leave.
WEIGHTS_PATH = f"elevator_dqn_{num_elevators}_{num_floors}.weights.pth"

def get_random_actions(num_elevators):
    """Returns a list of random actions."""
    return [random.choice([action_up, action_down, action_board, action_alight, action_nothing]) 
            for _ in range(num_elevators)]


def run_simulation(episodes=10):
    if MODE not in VALID_MODES:
        raise SystemExit(f"Unknown MODE {MODE!r}. Expected one of {VALID_MODES}.")

    set_global_seeds(SEED)
    env = ElevatorSystemEnv(seed=SEED)

    # Initialize the logic controller
    controller = TraditionalController(num_elevators, num_floors)

    # Initialize the DQN agent
    if MODE == 'model':
        # Ask the environment for the state size rather than recomputing the
        # layout, so the two cannot drift apart.
        agent = DQNAgent(env.state_dim, num_elevators)

        try:
            agent.load(WEIGHTS_PATH)
        except FileNotFoundError:
            # Carrying on here would benchmark a randomly initialised network
            # and report the numbers as the trained agent's.
            raise SystemExit(f"No weights at '{WEIGHTS_PATH}'. Run train_dqn.py first.")
        except RuntimeError as err:
            # Almost always a state_dim mismatch: weights trained before the
            # state representation changed.
            raise SystemExit(f"Could not load '{WEIGHTS_PATH}': {err}\n"
                             "If the state representation changed, retrain.")
        print(f"Loaded pre-trained model weights from {WEIGHTS_PATH}.")
        agent.epsilon = 0.0  # Disable exploration for evaluation

    print(f"--- STARTING SIMULATION (Mode: {MODE}) ---")
    print(f"Params: {num_elevators} Elevators, {num_floors} Floors")

    totals = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        info = {}  # populated with the episode metrics on the final step

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

        # --- EPISODE STATS ---
        print("-" * 52)
        print(f"EPISODE {ep + 1} FINISHED")
        print(f"Total Reward:            {total_reward:.1f}")
        print(f"Passengers Arrived:      {info.get('total_arrived', 0)}"
              f" / {info.get('total_spawned', 0)} spawned")
        print(f"Passengers Unserved:     {info.get('total_unserved', 0)}")
        print(f"Avg Wait (delivered):    {info.get('avg_wait_time', 0):.2f} steps")
        print(f"Avg Wait (ALL spawned):  {info.get('avg_wait_all', 0):.2f} steps  <- headline")
        print(f"Kuramoto Bunching:       {info.get('mean_bunching', 0):.5f}")
        print("-" * 52)

        totals.append(info)

    # --- SUMMARY ACROSS EPISODES ---
    if episodes > 1:
        print(f"\n=== MEAN OVER {episodes} EPISODES (Mode: {MODE}, seed {SEED}) ===")
        for key, label in (('total_arrived', 'Arrived'),
                           ('total_unserved', 'Unserved'),
                           ('avg_wait_all', 'Avg wait (all)'),
                           ('mean_bunching', 'Kuramoto R')):
            values = [t.get(key, 0) for t in totals]
            print(f"  {label:<16} {np.mean(values):8.3f}  (sd {np.std(values):.3f})")

if __name__ == "__main__":
    run_simulation()
