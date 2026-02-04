import numpy as np
from elevator_system.config import *

class TraditionalController:
    def __init__(self, num_elevators, num_floors):
        self.num_elevators = num_elevators
        self.num_floors = num_floors

    def get_actions(self, state):
        """
        Decides actions for all elevators based on a 'Nearest Call' heuristic.
        
        Args:
            state: Numpy array from env.get_state()
                   [e1_floor, e1_load, e2_floor, e2_load, ..., f0_up, f0_down, f1_up...]
        Returns:
            List of integers (actions), one per elevator.
        """
        actions = []
        
        # 1. Parse the State Vector
        # The first part of the vector is elevator data (2 values per elevator)
        elevator_data_end = self.num_elevators * 2
        elev_states = state[:elevator_data_end]
        
        # The rest is floor button data (2 values per floor)
        button_states = state[elevator_data_end:]
        
        # Reconstruct Button Map: tuples of (floor_index, direction_needed)
        active_calls = []
        for f in range(self.num_floors):
            # Index in button_states: Floor i has data at 2*i and 2*i+1
            idx_up = 2 * f
            idx_down = 2 * f + 1
            
            if button_states[idx_up] == 1.0:
                active_calls.append((f, 1)) # Floor f, Going UP
            if button_states[idx_down] == 1.0:
                active_calls.append((f, -1)) # Floor f, Going DOWN

        # 2. Decide Action for Each Elevator
        for i in range(self.num_elevators):
            # Extract this elevator's info
            # State vector is normalized, so we un-normalize to get actual floor
            norm_floor = elev_states[2 * i]
            current_floor = int(round(norm_floor * (self.num_floors - 1)))
            
            # Note: The Environment forces logic if passengers are inside.
            # We only need to decide what to do if we are empty or picking someone up.
            
            # Default to doing nothing
            action = action_nothing
            
            # STRATEGY: Find Nearest Active Call
            if active_calls:
                # Sort calls by distance to this elevator
                active_calls.sort(key=lambda x: abs(x[0] - current_floor))
                target_floor, direction = active_calls[0]
                
                if current_floor < target_floor:
                    action = action_up
                elif current_floor > target_floor:
                    action = action_down
                else:
                    # We are at the floor.
                    # If we are empty, BOARD.
                    # (If we were full, the Env would have forced ALIGHT already)
                    action = action_board

            actions.append(action)
            
        return actions
