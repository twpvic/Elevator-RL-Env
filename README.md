# Elevator Control System Optimization (Final Year Project)

This repository contains the code for my Final Year Project (FYP). The project explores and evaluates different dispatching strategies for a multi-elevator system and observes elevator behaviour using the Kuramoto model. It compares a Deep Reinforcement Learning approach (Deep Q-Network) against a traditional controller within a custom-built simulation environment.

## Project Overview

The primary goal of this FYP is to minimize passenger wait times and optimize elevator routing while mitigating the common transit problem of elevator "bunching." The project features a fully custom Python environment that simulates passengers, floors, and elevators, allowing for the direct comparison of different control algorithms.

**Key Features:**
* **Custom Simulation Environment:** A environment tracking individual passenger wait times, elevator capacities, and directional logic.
* **Traditional Controller:** A baseline heuristic controller utilizing a 'Nearest Call' strategy to dispatch elevators.
* **Deep Q-Network (DQN) Agent:** An RL agent built with TensorFlow that learns to dispatch elevators by observing the system state (elevator positions, loads, and floor button statuses).
* **Hyperparameter Tuning:** Integration with Optuna to automatically search for the optimal learning rate, gamma, epsilon decay, and batch size for the DQN model.
* **Kuramoto Bunching Analysis:** Implements a physics-based mathematical model to quantify how severely elevators clump together during operation.
* **Configurable Parameters:** Centralized settings to easily adjust the number of floors, elevators, arrival rates, and reward structures.

## Evaluation Metrics

The performance of each dispatching algorithm is evaluated on three main criteria:

1. **Average Wait Time:** The total time a passenger spends waiting on the floor plus the time spent inside the elevator.
2. **Throughput:** The total number of passengers successfully delivered to their destination within the simulation timeframe.
3. **Elevator Bunching (Kuramoto Order Parameter):** To measure the inefficiency of elevators moving in sync (bunching), the system treats each elevator's journey up and down the shaft as a phase oscillator on a circle (from $0$ to $2\pi$). 



The system calculates the Kuramoto order parameter $R$:
   
$$R = \left| \frac{1}{N} \sum_{j=1}^{N} e^{i\theta_j} \right|$$
   
* Where $N$ is the number of elevators and $\theta_j$ is the phase of elevator $j$.
* An $R$ value close to $1.0$ indicates perfect synchronization (**severe bunching**).
* An $R$ value close to $0.0$ indicates the elevators are perfectly out of phase (**optimal spacing**).

## Project Structure

* **`main.py`**: The main entry point to run simulations. It allows switching between 'random', 'controller', and 'model' (DQN) modes to evaluate performance.
* **`train_dqn.py`**: The training loop script used to train the DQN agent over a specified number of episodes.
* **`tune_dqn.py`**: The Optuna optimization script for hyperparameter tuning.
* **`dqn_agent.py`**: Contains the `DQNAgent` class, defining the neural network architecture, memory replay buffer, and epsilon-greedy action selection.
* **`controller.py`**: Implements the `TraditionalController` class, which dictates elevator actions based on standard operational logic.
* **`elevator_system/`**: The core simulation package.
    * `environment.py`: Manages the overall simulation state, passenger spawning, step-by-step execution, and calculates the Kuramoto bunching metric.
    * `elevator.py`: Defines the `Elevator` mechanics, including boarding, alighting, capacity constraints, and directional targeting.
    * `floor.py`: Manages floor-level queues and waiting passengers.
    * `passenger.py`: Tracks individual passenger metrics like source, destination, and wait times.
    * `config.py`: Contains global hyperparameters and constants.

## Installation & Setup

1.  **Clone the repository** to your local machine.
2.  **Install dependencies**: Ensure you have Python installed. Run :
```bash
pip install -r requirements.txt
```

## Usage

### 1. Configuration
Before running anything, you can adjust the physical parameters of the building (e.g., `num_floors`, `num_elevators`) and the RL reward structure inside `config.py`.

### 2. Hyperparameter Tuning (Optional)
To find the optimal hyperparameters for the DQN agent using Optuna, run:
```bash
python tune_dqn.py
```

### 3. Training DQN Model
To train the DQN model, run:
```bash
python train_dqn.py
```

### 4. Running the Simulation
MODE = 'controller': Runs the traditional logic.
MODE = 'model': Loads the trained TensorFlow weights and runs the RL agent.
MODE = 'raEdom': Selects random actions (used as a raw baseline).
```bash
python main.py
```
