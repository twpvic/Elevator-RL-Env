import csv
import json

import optuna
import numpy as np
import torch

from elevator_system.environment import ElevatorSystemEnv
from elevator_system.config import *
from dqn_agent import DQNAgent
from seeding import set_global_seeds

# --- Tuning config ---
trial_episodes = 100
target_update_freq = 10
seed = 42
n_trials = 30

best_params_path = "best_params.json"
study_csv_path = "tuning_trials.csv"


def save_trials_csv(study, path):
    """
    Dumps the trial history for the writeup.

    Uses the stdlib rather than study.trials_dataframe(), which requires pandas
    and would otherwise crash at the very end of a long tuning run.
    """
    param_names = sorted({k for t in study.trials for k in t.params})
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["number", "value", "state"] + param_names)
        for t in study.trials:
            writer.writerow([t.number, t.value, t.state.name]
                            + [t.params.get(k, "") for k in param_names])


def objective(trial):
    """
    Optuna objective function.
    Samples hyperparameters, trains the agent briefly, and returns a fitness score.

    Minimises average wait time across EVERY passenger who spawned. Optimising
    raw episode reward instead would largely be optimising "avoid invalid
    moves", since reward_invalid_move dominates the return, and that is not the
    metric the project is evaluated on. Averaging over all passengers rather
    than only the delivered ones stops a policy from scoring well by serving a
    few easy fares and stranding the rest.
    """

    # Define the Hyperparameter Search Space
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    gamma = trial.suggest_float('gamma', 0.8, 0.999)
    # Applied once per episode, so over trial_episodes this spans "exploration
    # ends around episode 44" (0.90) to "still exploring at the end" (0.99).
    epsilon_decay = trial.suggest_float('epsilon_decay', 0.90, 0.99)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

    # Setup the Environment
    # Every trial gets the same seed, so trials differ by hyperparameters
    # rather than by which passengers happened to show up.
    set_global_seeds(seed)
    env = ElevatorSystemEnv(seed=seed)

    # Initialize the agent with the trial's suggested hyperparameters
    agent = DQNAgent(
        state_dim=env.state_dim,
        num_elevators=num_elevators,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size
    )

    trial_waits = []

    # Run the Training Loop
    for e in range(trial_episodes):
        state = env.reset()
        done = False

        while not done:
            actions = agent.act(state)
            next_state, reward, done, info = env.step(actions)
            # executed_actions, not `actions`: the env overrides the request
            # whenever a car has passengers aboard, and the reward belongs to
            # what it was actually made to do.
            agent.remember(state, info['executed_actions'], info['rewards'],
                           next_state, info['terminated'])
            agent.replay()

            state = next_state

        agent.decay_epsilon()

        if (e + 1) % target_update_freq == 0:
            agent.update_target_model()

        trial_waits.append(info['avg_wait_all'])

        # Report intermediate results to Optuna for Pruning
        trial.report(info['avg_wait_all'], e)

        # Handle Pruning: If the trial is performing terribly early on, Optuna will kill it to save time
        if trial.should_prune():
            raise optuna.TrialPruned()

    # Return the Metric to Minimize
    # Average of the last 10 episodes, to gauge stabilized performance rather
    # than a lucky single episode.
    return float(np.mean(trial_waits[-10:]))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- STARTING OPTUNA HYPERPARAMETER TUNING ON {device.type.upper()} ---")

    # Lower average wait time is better.
    # The sampler is seeded too, so the whole search is reproducible.
    study = optuna.create_study(
        study_name="dqn_elevator_tuning",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )

    # Run the optimization for a set number of trials
    study.optimize(objective, n_trials=n_trials)

    # --- PRINT RESULTS ---
    print("\n" + "="*40)
    print("TUNING COMPLETE")
    print("="*40)
    print(f"Number of finished trials: {len(study.trials)}")

    # Written first, and unconditionally: study.best_trial raises if every trial
    # was pruned or failed, and losing the whole history to that after a long
    # search would be worse than having no best_params.json.
    save_trials_csv(study, study_csv_path)
    print(f"Full trial history written to {study_csv_path}.")

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise SystemExit(
            "No trial ran to completion -- every one was pruned or failed, so "
            "there is no best set of hyperparameters to write. Loosen the "
            "pruner (raise n_startup_trials / n_warmup_steps) and re-run.")

    best_trial = study.best_trial
    print(f"\nBest Trial (ID: {best_trial.number}):")
    print(f"  Average Wait Time (all passengers): {best_trial.value:.2f} steps")
    print("  Optimal Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # --- PERSIST RESULTS ---
    # Without this the whole search is discarded when the process exits.
    with open(best_params_path, "w") as f:
        json.dump(best_trial.params, f, indent=2)

    print(f"\nBest hyperparameters written to {best_params_path} "
          f"(train_dqn.py picks these up automatically).")
