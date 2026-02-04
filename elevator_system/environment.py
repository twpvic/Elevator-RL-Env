import numpy as np
import random
from config import *
from .passenger import Passenger
from .floor import Floor
from .elevator import Elevator

class ElevatorSystemEnv:
    def __init__(self):
        self.floors = [Floor(i) for i in range range(num_floors)]
        self.elevators = [Elevator(i, num_floors) for i in range(num_elevators)]
        self.passenger_counter = 0
        self.time = 0

        # 2 features per elevator (Floor, Load) + 2 features per floor (Up, Down)
        self.arrived_passengers = []
        self.state_dim = (2 * num_elevators) + (2 * num_floors)
        

    def reset(self):
        self.floors = [Floor(i) for i in range range(num_floors)]
        self.elevators = [Elevator(i, num_floors) for i in range(num_elevators)]
        self.passenger_counter = 0
        self.time = 0
        self.arrived_passengers = []
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
            norm_load = len(elevator.passengers) / elevator_capacity

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

        return np.array(state, dtype.np.float32)

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
        # 1. Passengers waiting on floors
        for f in self.floors:
            for p in f.waiting_passengers:
                p.wait_time_floor += 1
        
        # 2. Passengers inside elevators
        for e in self.elevators:
            for queue in e.passengers: # e.passengers is a list of deques
                for p in queue:
                    p.wait_time_lift += 1

    def step(self, actions):
        self.time_step += 1
        self.spawn_passengers()
        self.update_wait_times()

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
            else:
                info['avg_wait_time'] = 0
                info['total_arrived'] = 0

        for i, elevator in enumerate(self.elevators):
            action = actions[i]
            
            # --- FORCED LOGIC (If passengers inside) ---
            if not elevator.is_empty():
                # 1. If anyone wants off HERE, force Alight
                if elevator.passengers[elevator.current_floor]:
                    action = action_alight
                else:
                    # 2. Else, move towards nearest target
                    target_floor = -1
                    # Find first non-empty queue
                    for floor_idx, q in enumerate(elevator.passengers):
                        if q:
                            target_floor = floor_idx
                            break
                    
                    if target_floor != -1:
                        if elevator.current_floor < target_floor:
                            action = action_up
                        elif elevator.current_floor > target_floor:
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
                    # Pass the actual deque to the elevator to pull from
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
                    
