num_floors = 10
num_elevators = 2
elevator_capacity = 4
arrival_rate = 0.3  # Mean new passengers per time step (Poisson)
max_simulation_steps = 1000
passengers_per_step = 1 # Max passengers that can board/alight per time step

# Discrete Actions
action_nothing = 0
action_up = 1
action_down = 2
action_board = 3
action_alight = 4

# Rewards
reward_delivery = 10
reward_board = 2
reward_step_penalty = -0.1
reward_invalid_move = -10
reward_wait_penalty = -0.01  # Per waiting passenger, per time step


def _validate():
    """
    Checked at import so a bad configuration fails here with a clear message
    """
    if num_floors < 2:
        raise ValueError(f"num_floors must be >= 2 (got {num_floors}); "
                         "floor normalisation divides by num_floors - 1.")
    if num_elevators < 1:
        raise ValueError(f"num_elevators must be >= 1 (got {num_elevators}).")
    if elevator_capacity < 1:
        raise ValueError(f"elevator_capacity must be >= 1 (got {elevator_capacity}).")
    if passengers_per_step < 1:
        raise ValueError(f"passengers_per_step must be >= 1 (got {passengers_per_step}).")
    if max_simulation_steps < 1:
        raise ValueError(f"max_simulation_steps must be >= 1 (got {max_simulation_steps}).")
    if arrival_rate < 0:
        raise ValueError(f"arrival_rate must be >= 0 (got {arrival_rate}).")


_validate()
