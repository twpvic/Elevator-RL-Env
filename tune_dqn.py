import optuna
import numpy as np
import random
from elevator_system.environment import ElevatorSystemEnv
from elevator_system.config import *
from dqn_agent import DQNAgent
import tensorflow as tf


# --- Tuning config ---
trial_episodes = 100
target_update_freq = 10

def objective(trial):
    """
    Optuna objective function. 
    Samples hyperparameters, trains the agent briefly, and returns a fitness score.
    """
    
    # 1. Define the Hyperparameter Search Space
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    gamma = trial.suggest_float('gamma', 0.8, 0.999)
    epsilon_decay = trial.suggest_float('epsilon_decay', 0.95, 0.9999)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    
    # 2. Setup the Environment
    env = ElevatorSystemEnv()
    
    # Calculate state dimension based on config: 2 features per elevator + 2 features per floor
    state_dim = (2 * num_elevators) + (2 * num_floors)
    
    # Initialize the agent with the trial's suggested hyperparameters
    agent = DQNAgent(
        state_dim=state_dim, 
        num_elevators=num_elevators,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size
    )

    trial_rewards = []

    # 3. Run the Training Loop
    for e in range(trial_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            actions = agent.act(state)
            next_state, reward, done, info = env.step(actions)
            agent.remember(state, actions, reward, next_state, done)
            agent.replay()
            
            state = next_state
            total_reward += reward

        if (e + 1) % target_update_freq == 0:
            agent.update_target_model()

        trial_rewards.append(total_reward)
        
        # 4. Report intermediate results to Optuna for Pruning
        trial.report(total_reward, e)
        
        # Handle Pruning: If the trial is performing terribly early on, Optuna will kill it to save time
        if trial.should_prune():
            raise optuna.TrialPruned()

    # 5. Return the Metric to Maximize
    # We use the average reward of the last 10 episodes to gauge the final stabilized performance
    final_score = np.mean(trial_rewards[-10:])
    return final_score

if __name__ == "__main__":
    print("--- STARTING OPTUNA HYPERPARAMETER TUNING ---")
    
    # Create a study object and specify that we want to MAXIMIZE the reward
    study = optuna.create_study(
        study_name="dqn_elevator_tuning",
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    # Run the optimization for a set number of trials
    study.optimize(objective, n_trials=30)

    # --- PRINT RESULTS ---
    print("\n" + "="*40)
    print("TUNING COMPLETE")
    print("="*40)
    print(f"Number of finished trials: {len(study.trials)}")
    
    best_trial = study.best_trial
    print(f"\nBest Trial (ID: {best_trial.number}):")
    print(f"  Average Final Reward: {best_trial.value:.2f}")
    print("  Optimal Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")