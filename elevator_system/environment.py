import math
import numpy as np
import random
from .config import *
from .passenger import Passenger
from .floor import Floor
from .elevator import Elevator

class ElevatorSystemEnv:
    def __init__(self):
        self.floors = [Floor(i) for i in range(num_floors)]
        self.elevators = [Elevator(i, num_floors) for i in range(num_elevators)]
        self.passenger_counter = 0
        self.time_step = 0

        # 2 features per elevator (Floor, Load) + 2 features per floor (Up, Down)
        self.arrived_passengers = []
        self.state_dim = (2 * num_elevators) + (2 * num_floors)

        self.bunching_history = [] 
        

    def reset(self):
        self.floors = [Floor(i) for i in range(num_floors)]
        self.elevators = [Elevator(i, num_floors) for i in range(num_elevators)]
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

        return np.array(state, dtype=np.float32)

    def spawn_passengers(self):
        if random.random() < arrival_rate:
            start = random.randint(0, num_floors - 1)
            dest = random.randint(0, num_floors - 1)
            while dest == start:
                dest = random.randint(0, num_floors - 1)

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
        self.spawn_passengers()
        self.update_wait_times()

        current_bunching = self.calculate_kuramoto_order_parameter()
        self.bunching_history.append(current_bunching)

        total_reward = 0
        done = False
        info = {}

        # Check simulation end
        if self.time_step >= max_simulation_steps:
            done = True
            # Calculate metrics
            if len(self.arrived_passengers) > 0:
                avg_wait = sum(p.wait_time_floor + p.wait_time_lift for p in self.arrived_passengers) / len(self.arrived_passengers)
                info['avg_wait_time'] = avg_wait
                info['total_arrived'] = len(self.arrived_passengers)

        for i, elevator in enumerate(self.elevators):
            action = actions[i]
            
            # --- FORCED LOGIC (If passengers inside) ---
            if not elevator.is_empty():
                # 1. If anyone wants alight, force Alight
                if len(elevator.passengers[elevator.current_floor]) > 0:
                    action = action_alight

                elif not elevator.is_empty():
                    # 2. Else, if we have a direction, we should try to move in that direction
                    target_dir = elevator.get_target_direction()
                    if target_dir == 1:
                        action = action_up
                    elif target_dir == -1:
                        action = action_down


            # --- EXECUTE ACTION ---
            if action == action_up:
                success, r = elevator.move(action_up)
                total_reward += r

            elif action == action_down:
                success, r = elevator.move(action_down)
                total_reward += r

            elif action == action_board:
                # Logic: Board from the queue that matches elevator direction, or longest if idle
                # Simplified: Just take anyone (FIFO) for this tutorial level
                if self.floors[elevator.current_floor].has_passengers():
                    floor_queue = self.floors[elevator.current_floor].waiting_passengers
                    boarded_count = elevator.board_passengers(floor_queue)
                    
                    if boarded_count > 0:
                        total_reward += (2 * boarded_count) # Reward per person
                    else:
                        total_reward += reward_invalid_move # Full
                else:
                    total_reward += reward_invalid_move # Empty floor

            elif action == action_alight:
                alighted_list = elevator.alight_passengers()
                if alighted_list:
                    total_reward += (reward_delivery * len(alighted_list))
                    
                    # Log arrival data
                    for p in alighted_list:
                        p.arrival_time = self.time_step
                        self.arrived_passengers.append(p)
                else:
                    total_reward += reward_invalid_move # Nobody to drop

            elif action == action_nothing:
                total_reward += reward_step_penalty

        return self.get_state(), total_reward, done, info
                    
    def calculate_kuramoto_order_parameter(self):
        """
        Calculates the Kuramoto order parameter to measure elevator bunching.
        Returns a value between 0 and 1, where 1 means perfectly synchronized (bunched) and 0 means perfectly desynchronized.
        """
        if num_elevators <= 1:
            return 1.0  # With one elevator, it's trivially "bunched"
        
        phases = []

        for elevator in self.elevators:
            norm_position = elevator.current_floor / (num_floors - 1)  

            direction = elevator.get_target_direction()
            if direction == 0:
                 # Idle elevators contribute based on their position
                phases.append(norm_position * math.pi)
                phases.append(- norm_position * math.pi)
            elif direction == 1:
                phases.append(norm_position * math.pi)  # Up direction
            else:
                phases.append(- norm_position * math.pi)  # Down direction
            
        # Calculate Kuramoto order parameter
        n = len(phases)
        if n == 0:
            return 0.0
        sum_real = sum(math.cos(phase) for phase in phases)
        sum_imag = sum(math.sin(phase) for phase in phases)
        r = math.hypot(sum_real, sum_imag) / n
        return r