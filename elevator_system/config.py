num_floors = 10
num_elevators = 2
elevator_capacity = 4
arrival_rate = 0.8  # Probability of a new passenger spawning per time step
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
reward_step_penalty = -0.1
reward_invalid_move = -10
