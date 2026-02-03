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
        self.state_dim = (2 * num_elevators) + (2 * num_floors)
        

    def reset(self):
        self.floors = [Floor(i) for i in range range(num_floors)]
        self.elevators = [Elevator(i, num_floors) for i in range(num_elevators)]
        self.passenger_counter = 0
        self.time = 0
        return self.get_state()

    def get_state(self):
        """
        returns numpy array representing the world state
        """
        state = []
        for elevator in self.elevators:
            # state for every elevator
            # normalize floor
            norm_floor = elevator.current_floor / num_floors
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
            start = random.randint(0, num_floors)
            dest = random.randint(0, num_floors)
            while dest == start:
                dest = random.randint(0, num_floors)

            p = Passenger(self.passenger_counter, start, dest, self.time_step)
            self.floors[start].add_passenger(p)
            self.passenger_counter += 1

    def step(self, actions):
        self.time_step += 1
        self.spawn_passengers()

        total_reward = 0
        done = False
        info = {}

        for i, elevator in enumerate(self.elevators):
            action = actions[i]
            current_floor_obj = self.floors[elevator.current_floor]

            # forced actions
            # check if we have passengers in the elevator

            if not elevator.is_empty():
                if elevator.passengers[elevator.current_floor]:
                    action = action_alight

                else:
                    target_floor = -1
                    for floor_idx, q in enumerate(elevator.passengers):
                        if q:
                            target_floor = floor_idx
                            break

                    if target_floor != -1
                        if elevator.current_floor < target_floor:
                            action = action_up

                        elif elevator.current_floor > target_floor
                            action = action_down

            # execute action
            if action == action_up:
                
            
                    
