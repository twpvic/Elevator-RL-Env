import json
import os

import numpy as np
import torch

from elevator_system.environment import ElevatorSystemEnv
from elevator_system.config import *
from dqn_agent import DQNAgent
from seeding import set_global_seeds

# --- GPU CONFIGURATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPUs detected: {torch.cuda.device_count()}.")
else:
    print("No GPUs detected. Running on CPU.")

# Training Settings
EPISODES = 100
TARGET_UPDATE_FREQ = 10   # Update target network every X episodes
CHECKPOINT_FREQ = 10      # Write a resumable checkpoint every X episodes
SEED = 42

# Set True to continue from CHECKPOINT_PATH instead of starting over. Left off
# by default so re-running the script is always a clean run; flip it only when
# picking up after a crash.
RESUME = False

MODEL_PATH = f"elevator_dqn_{num_elevators}_{num_floors}.weights.pth"
BEST_PATH = f"elevator_dqn_{num_elevators}_{num_floors}.best.pth"
CHECKPOINT_PATH = f"elevator_dqn_{num_elevators}_{num_floors}.checkpoint.pth"
BEST_PARAMS_PATH = "best_params.json"


def load_best_params():
    """Picks up hyperparameters from tune_dqn.py, if it has been run."""
    if not os.path.exists(BEST_PARAMS_PATH):
        return {}
    with open(BEST_PARAMS_PATH) as f:
        params = json.load(f)
    print(f"Using tuned hyperparameters from {BEST_PARAMS_PATH}: {params}")
    return params


def load_checkpoint(agent):
    """
    Restores a run from CHECKPOINT_PATH. Returns (start_episode, best_wait).

    Note this resumes the networks, optimiser and exploration rate, but NOT the
    replay buffer or the RNG streams -- those are not serialised. A resumed run
    is therefore close to, but not bit-identical with, an uninterrupted one.
    """
    if not os.path.exists(CHECKPOINT_PATH):
        raise SystemExit(f"RESUME is set but there is no checkpoint at "
                         f"'{CHECKPOINT_PATH}'. Set RESUME = False to start fresh.")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=agent.device, weights_only=True)
    agent.model.load_state_dict(ckpt['model'])
    agent.target_model.load_state_dict(ckpt['target_model'])
    agent.optimizer.load_state_dict(ckpt['optimizer'])
    agent.epsilon = ckpt['epsilon']

    start_episode = ckpt['episode']
    best_wait = ckpt['best_wait']
    print(f"Resuming from {CHECKPOINT_PATH}: episode {start_episode}, "
          f"epsilon {agent.epsilon:.3f}, best wait {best_wait:.2f}")
    print("(replay buffer and RNG state are not restored)")
    return start_episode, best_wait


def run_training():
    set_global_seeds(SEED)
    env = ElevatorSystemEnv(seed=SEED)

    # Ask the environment rather than recomputing the layout here, so the two
    # cannot drift apart when the state representation changes.
    agent = DQNAgent(env.state_dim, num_elevators, **load_best_params())

    # Selected on wait time across every passenger, not on reward: reward is a
    # training signal, wait time is the thing the project is actually judged on.
    start_episode, best_wait = 0, float('inf')
    if RESUME:
        start_episode, best_wait = load_checkpoint(agent)

    print(f"--- STARTING TRAINING ---")
    print(f"Elevators: {num_elevators}, Floors: {num_floors}")
    print(f"State Dim: {env.state_dim}")
    print(f"Seed: {SEED}")
    print(f"Episodes: {start_episode} -> {EPISODES}")

    if start_episode >= EPISODES:
        raise SystemExit(f"Checkpoint is already at episode {start_episode} of "
                         f"{EPISODES}; nothing left to run. Raise EPISODES to "
                         f"train further.")

    for e in range(start_episode, EPISODES):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # 1. Select Action
            actions = agent.act(state)

            # 2. Step Environment
            next_state, reward, done, info = env.step(actions)

            # 3. Store Experience
            #    executed_actions, not `actions`: the env overrides the request
            #    whenever a car has passengers aboard, and the reward belongs to
            #    what it was made to do. Per-elevator rewards, and `terminated`
            #    rather than `done` so the time limit is not a terminal state.
            agent.remember(state, info['executed_actions'], info['rewards'],
                           next_state, info['terminated'])

            # 4. Train Agent
            agent.replay()

            state = next_state
            total_reward += reward

        # End of Episode
        agent.decay_epsilon()

        if (e + 1) % TARGET_UPDATE_FREQ == 0:
            agent.update_target_model()

        avg_wait = info.get('avg_wait_all', float('inf'))
        if avg_wait < best_wait:
            best_wait = avg_wait
            agent.save(BEST_PATH)

        if (e + 1) % CHECKPOINT_FREQ == 0:
            # Enough to resume rather than restart if the run dies.
            torch.save({
                'episode': e + 1,
                'model': agent.model.state_dict(),
                'target_model': agent.target_model.state_dict(),
                'optimizer': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon,
                'best_wait': best_wait,
            }, CHECKPOINT_PATH)
            print(f"Episode {e+1}/{EPISODES} | Reward: {total_reward:8.1f} | "
                  f"Eps: {agent.epsilon:.2f} | Arrived: {info.get('total_arrived', 0)}"
                  f"/{info.get('total_spawned', 0)} | "
                  f"Wait(all): {avg_wait:.1f} | R: {info.get('mean_bunching', 0):.3f}")
        else:
            print(f"Episode {e+1}/{EPISODES} | Reward: {total_reward:8.1f}", end='\r')

    agent.save(MODEL_PATH)
    print(f"\nTraining Complete.")
    print(f"  Final model: {MODEL_PATH}")
    print(f"  Best model:  {BEST_PATH} (avg wait over all passengers: {best_wait:.2f})")


if __name__ == "__main__":
    run_training()
