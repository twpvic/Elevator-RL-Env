# Elevator Control System Optimization (Final Year Project)

This repository contains the code for my Final Year Project (FYP). The project explores and evaluates different dispatching strategies for a multi-elevator system and observes elevator behaviour using the Kuramoto model. It compares a Deep Reinforcement Learning approach (Deep Q-Network) against a traditional controller within a custom-built simulation environment.

## Project Overview

The primary goal of this FYP is to minimize passenger wait times and optimize elevator routing while mitigating the common transit problem of elevator "bunching." The project features a fully custom Python environment that simulates passengers, floors, and elevators, allowing for the direct comparison of different control algorithms.

**Key Features:**
* **Custom Simulation Environment:** A environment tracking individual passenger wait times, elevator capacities, and directional logic. Passenger arrivals follow a Poisson process.
* **Traditional Controller:** A baseline heuristic controller utilizing a 'Nearest Call' strategy to dispatch elevators. Each call is assigned to exactly one elevator, so cars do not all chase the same passenger.
* **Deep Q-Network (DQN) Agent:** An RL agent built with PyTorch that learns to dispatch elevators by observing the system state (elevator positions, loads, floor button statuses, and the car calls registered inside each elevator).
* **Hyperparameter Tuning:** Integration with Optuna to automatically search for the optimal learning rate, gamma, epsilon decay, and batch size for the DQN model.
* **Kuramoto Bunching Analysis:** Implements a physics-based mathematical model to quantify how severely elevators clump together during operation.
* **Configurable Parameters:** Centralized settings to easily adjust the number of floors, elevators, arrival rates, and reward structures.
* **Reproducible:** Every entry point is seeded, and the simulation owns its own RNG stream, so all three modes are evaluated against an identical passenger sequence.

## Evaluation Metrics

The performance of each dispatching algorithm is evaluated on three main criteria:

1. **Average Wait Time:** The total time a passenger spends waiting on the floor plus the time spent inside the elevator. Two figures are reported: `avg_wait_time` over delivered passengers only, and `avg_wait_all` over every passenger who spawned. The latter is the headline number — averaging over delivered passengers alone rewards a policy that serves a few easy fares and strands everyone else.
2. **Throughput:** The total number of passengers successfully delivered to their destination within the simulation timeframe, reported alongside the number left unserved.
3. **Elevator Bunching (Kuramoto Order Parameter):** To measure the inefficiency of elevators moving in sync (bunching), the system treats each elevator's journey up and down the shaft as a phase oscillator on a circle (from $0$ to $2\pi$). 



The system calculates the Kuramoto order parameter $R$:
   
$$R = \left| \frac{1}{N} \sum_{j=1}^{N} e^{i\theta_j} \right|$$
   
* Where $N$ is the number of elevators and $\theta_j$ is the phase of elevator $j$.
* An $R$ value close to $1.0$ indicates perfect synchronization (**severe bunching**).
* An $R$ value close to $0.0$ indicates the elevators are perfectly out of phase (**optimal spacing**).

Position maps onto $[0, \pi]$ while travelling up and $[-\pi, 0]$ while travelling down, so one full up-and-down journey traverses the circle exactly once. An idle car is given the direction it last travelled, since it is still parked somewhere in its cycle.

> **Caveat for interpretation:** with $N = 2$ the order parameter reduces to $R = |\cos(\Delta\theta / 2)|$, so it only falls when the two cars are near-antipodal in the shaft. It is a coarse discriminator at small $N$ and can stay high even when a change measurably reduces how often cars physically share a floor. For bunching conclusions, prefer $N \geq 4$ and report a physical spacing statistic alongside $R$.

## Project Structure

* **`main.py`**: The main entry point to run simulations. It allows switching between 'random', 'controller', and 'model' (DQN) modes to evaluate performance.
* **`train_dqn.py`**: The training loop script used to train the DQN agent over a specified number of episodes. Writes periodic checkpoints and keeps the best-scoring weights separately. Set `RESUME = True` to continue from the last checkpoint after a crash.
* **`tune_dqn.py`**: The Optuna optimization script for hyperparameter tuning. Writes `best_params.json` (picked up automatically by `train_dqn.py`) and `tuning_trials.csv`.
* **`dqn_agent.py`**: Contains the `DQNAgent` class, defining the neural network architecture, memory replay buffer, and epsilon-greedy action selection.
* **`controller.py`**: Implements the `TraditionalController` class, which dictates elevator actions based on standard operational logic.
* **`seeding.py`**: Seeds Python, NumPy and PyTorch together, so runs are reproducible.
* **`elevator_system/`**: The core simulation package.
    * `environment.py`: Manages the overall simulation state, passenger spawning, step-by-step execution, and calculates the Kuramoto bunching metric.
    * `elevator.py`: Defines the `Elevator` mechanics, including boarding, alighting, capacity constraints, and directional targeting. A car commits to a sweep direction and holds it until nothing remains ahead, rather than re-deciding every step.
    * `floor.py`: Manages floor-level queues and waiting passengers.
    * `passenger.py`: Tracks individual passenger metrics like source, destination, and wait times.
    * `config.py`: Contains global hyperparameters and constants.
* **`tests/`**: Regression tests pinning the simulation and agent behaviours that have previously been wrong. Run with `python -m pytest tests/ -v`.

## Installation & Setup

1.  **Clone the repository** to your local machine.
2.  **Install dependencies**: Ensure you have Python installed. Run :
```bash
pip install -r requirements.txt
```

## Usage

### 1. Configuration
Before running anything, you can adjust the physical parameters of the building (e.g., `num_floors`, `num_elevators`) and the RL reward structure inside `elevator_system/config.py`. 


### 2. Hyperparameter Tuning (Optional)
To find the optimal hyperparameters for the DQN agent using Optuna, run:
```bash
python tune_dqn.py
```
This minimises average wait time across all passengers and writes `best_params.json`.

### 3. Training DQN Model
To train the DQN model, run:
```bash
python train_dqn.py
```
If `best_params.json` exists, the tuned hyperparameters are used automatically. Training writes three files: `*.weights.pth` (final), `*.best.pth` (best average wait time seen), and `*.checkpoint.pth` (resumable).

### 4. Running the Simulation
Set `MODE` at the top of `main.py`:

* `MODE = 'controller'`: Runs the traditional logic.
* `MODE = 'model'`: Loads the trained PyTorch weights and runs the RL agent.
* `MODE = 'random'`: Selects random actions (used as a raw baseline).

```bash
python main.py
```

All three modes share a fixed seed, so they face an identical passenger sequence and the differences between them are attributable to the policy rather than to luck.
