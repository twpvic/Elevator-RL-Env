import math
import numpy as np
from .config import *
from .passenger import Passenger
from .floor import Floor
from .elevator import Elevator

class ElevatorSystemEnv:
    def __init__(self, seed=None):
        # The simulation keeps its own RNG stream rather than using the global
        # one, so arrivals stay reproducible and are not perturbed by whatever
        # the agent happens to sample during exploration.
        self.rng = np.random.default_rng(seed)

        # 2 per elevator (Floor, Load) + 2 per floor (Up, Down)
        # + one car-call bit per floor per elevator
        self.state_dim = ((2 * num_elevators) + (2 * num_floors)
                          + (num_elevators * num_floors))

        self.reset()

    def reset(self, seed=None):
        """
        Starts a fresh episode. Pass a seed to replay one exact episode;
        omit it to draw the next episode from the existing stream.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.floors = [Floor(i) for i in range(num_floors)]
        self.elevators = [Elevator(i, num_floors, self.rng) for i in range(num_elevators)]
        self.passenger_counter = 0
        self.time_step = 0
        self.arrived_passengers = []
        self.bunching_history = []
        return self.get_state()

    def get_state(self):
        """
        returns numpy array representing the world state
        """
        state = []
        for elevator in self.elevators:
            # state for every elevator
            # normalize floor
            norm_floor = elevator.current_floor / (num_floors - 1)
            # normalize load
            norm_load = elevator.total_passengers / elevator_capacity

            state.append(norm_floor)
            state.append(norm_load)

        # add button state for every floor
        for f in self.floors:
            up_button = 0.0
            down_button = 0.0

            for p in f.waiting_passengers:
                if p.dest_floor > f.level:
                    up_button = 1.0
                elif p.dest_floor < f.level:
                    down_button = 1.0
                if up_button == 1.0 and down_button == 1.0:
                    break
            state.append(up_button)
            state.append(down_button)

        # Car calls: which floors the passengers already inside each car have
        # selected. Without this the agent cannot see where its own load wants
        # to go, which leaves the problem needlessly partially observed.
        # Appended last so existing parsers of the earlier blocks stay valid.
        for elevator in self.elevators:
            for floor in range(num_floors):
                state.append(1.0 if elevator.passengers[floor] else 0.0)

        return np.array(state, dtype=np.float32)

    def spawn_passengers(self):
        # Poisson arrivals. A Bernoulli trial can never produce two passengers
        # in the same step, so it cannot represent the bursts that actually
        # stress a lift system.
        for _ in range(int(self.rng.poisson(arrival_rate))):
            start = int(self.rng.integers(num_floors))
            # Draw from the num_floors - 1 floors that are not `start`, then
            # shift past it. Avoids an unbounded rejection loop.
            dest = int(self.rng.integers(num_floors - 1))
            if dest >= start:
                dest += 1

            p = Passenger(self.passenger_counter, start, dest, self.time_step)
            self.floors[start].add_passenger(p)
            self.passenger_counter += 1

    def update_wait_times(self):
        # Passengers waiting on floors
        for f in self.floors:
            for p in f.waiting_passengers:
                p.wait_time_floor += 1
        
        # Passengers inside elevators
        for e in self.elevators:
            for queue in e.passengers: # e.passengers is a list of deques
                for p in queue:
                    p.wait_time_lift += 1

    def step(self, actions):
        self.time_step += 1

        # Age the passengers already in the system BEFORE the new arrivals land,
        # so nobody is charged for the step they spawned in. Charging them made
        # `arrival_time - spawn_time` disagree with the accumulated wait counters
        # by one, and put a floor of 1 under every passenger's wait.
        self.update_wait_times()
        self.spawn_passengers()

        # Rewards are tracked per elevator rather than as one scalar: crediting
        # every car for whatever any other car delivered makes the learning
        # signal far noisier than it needs to be. They still sum to the
        # system-level reward, which is what step() returns.
        rewards = np.zeros(len(self.elevators), dtype=np.float32)

        # Waiting is what the system exists to minimise, so charge for it every
        # step. Without this nothing in the reward distinguishes a passenger
        # served in 5 steps from one served in 500. It is a whole-system cost,
        # so it is split evenly rather than blamed on any one car.
        waiting = sum(len(f.waiting_passengers) for f in self.floors)
        riding = sum(e.total_passengers for e in self.elevators)
        rewards += reward_wait_penalty * (waiting + riding) / len(self.elevators)

        # What each car was actually made to do, which is not always what was
        # asked for -- see the forced-logic block below. Training on the
        # requested action while paying out the executed action's reward
        # credits the agent for moves it never made, so step() reports this
        # back and the replay buffer stores it instead of `actions`.
        executed_actions = []

        for i, elevator in enumerate(self.elevators):
            action = actions[i]

            # --- FORCED LOGIC (If passengers inside) ---
            if not elevator.is_empty():
                waiting_here = self.floors[elevator.current_floor].waiting_passengers

                # 1. If anyone wants alight, force Alight
                if len(elevator.passengers[elevator.current_floor]) > 0:
                    action = action_alight

                # 2. Honour a requested en-route pickup: we still have room and
                #    someone here is heading our way, so stopping costs the
                #    passengers inside nothing but a single step.
                elif action == action_board and elevator.has_compatible_passenger(waiting_here):
                    pass

                else:
                    # 3. Else, if we have a direction, we should try to move in that direction
                    target_dir = elevator.get_target_direction()
                    if target_dir == 1:
                        action = action_up
                    elif target_dir == -1:
                        action = action_down


            # --- EXECUTE ACTION ---
            executed_actions.append(int(action))

            if action == action_up:
                success, r = elevator.move(action_up)
                rewards[i] += r

            elif action == action_down:
                success, r = elevator.move(action_down)
                rewards[i] += r

            elif action == action_board:
                # Logic: Board from the queue that matches elevator direction, or longest if idle
                # Simplified: Just take anyone (FIFO) for this tutorial level
                if self.floors[elevator.current_floor].has_passengers():
                    floor_queue = self.floors[elevator.current_floor].waiting_passengers
                    boarded_count = elevator.board_passengers(floor_queue)

                    if boarded_count > 0:
                        rewards[i] += (reward_board * boarded_count) # Reward per person
                    else:
                        rewards[i] += reward_invalid_move # Full
                else:
                    rewards[i] += reward_invalid_move # Empty floor

            elif action == action_alight:
                alighted_list = elevator.alight_passengers()
                if alighted_list:
                    rewards[i] += (reward_delivery * len(alighted_list))

                    # Log arrival data
                    for p in alighted_list:
                        p.arrival_time = self.time_step
                        self.arrived_passengers.append(p)
                else:
                    rewards[i] += reward_invalid_move # Nobody to drop

            elif action == action_nothing:
                rewards[i] += reward_step_penalty

        # Measured after the actions, so the configuration recorded is the one
        # this step actually produced.
        self.bunching_history.append(self.calculate_kuramoto_order_parameter())

        # The episode ends because it hit the horizon, not because the building
        # reached a terminal state. Reported separately so the agent knows to
        # keep bootstrapping through it.
        truncated = self.time_step >= max_simulation_steps
        done = truncated

        info = {'rewards': rewards, 'executed_actions': executed_actions,
                'terminated': False, 'truncated': truncated}
        if done:
            # Computed after the loop, so this step's deliveries are counted.
            info.update(self.episode_metrics())

        return self.get_state(), float(rewards.sum()), done, info

    def episode_metrics(self):
        """
        Summary over EVERY passenger who spawned, not just the delivered ones.
        """
        arrived = self.arrived_passengers
        unserved = [p for f in self.floors for p in f.waiting_passengers]
        unserved += [p for e in self.elevators for q in e.passengers for p in q]

        waits_arrived = [p.wait_time_floor + p.wait_time_lift for p in arrived]
        waits_all = waits_arrived + [p.wait_time_floor + p.wait_time_lift
                                     for p in unserved]

        return {
            'total_spawned': self.passenger_counter,
            'total_arrived': len(arrived),
            'total_unserved': len(unserved),
            # Delivered passengers only. Flattering, and gameable: a policy that
            # serves a handful of easy fares and strands everyone else scores
            # well here. Kept because it is the figure reported historically.
            'avg_wait_time': float(np.mean(waits_arrived)) if waits_arrived else 0.0,
            # Everyone who ever spawned, so stranding passengers is penalised.
            # This is the honest headline number.
            'avg_wait_all': float(np.mean(waits_all)) if waits_all else 0.0,
            'mean_bunching': (float(np.mean(self.bunching_history))
                              if self.bunching_history else 0.0),
        }


    def calculate_kuramoto_order_parameter(self):
        """
        Calculates the Kuramoto order parameter to measure elevator bunching.
        Returns a value between 0 and 1, where 1 means perfectly synchronized (bunched) and 0 means perfectly desynchronized.
        """
        n = len(self.elevators)
        if n <= 1:
            return 1.0  # With one elevator, it's trivially "bunched"

        phases = []

        for elevator in self.elevators:
            norm_position = elevator.current_floor / (num_floors - 1)

            direction = elevator.get_target_direction()
            if direction == 0:
                # An idle car is still parked somewhere in its up/down cycle,
                # so give it one phase based on where it was last headed.
                # Giving it two (one per direction) would let it cancel itself
                # out while still counting toward n, deflating R.
                direction = elevator.last_direction

            # Position maps onto [0, pi] going up and [-pi, 0] coming down, so
            # one full up-and-down journey traverses the circle exactly once.
            phases.append(direction * norm_position * math.pi)

        # Calculate Kuramoto order parameter
        sum_real = sum(math.cos(phase) for phase in phases)
        sum_imag = sum(math.sin(phase) for phase in phases)
        return math.hypot(sum_real, sum_imag) / n