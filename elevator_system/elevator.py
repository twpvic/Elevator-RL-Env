from collections import deque
import random
from .config import *

class Elevator:
    def __init__(self, elevator_id, num_floors):
        self.id = elevator_id
        
        # self.passengers[3] = deque of passengers going to floor 3.
        self.passengers = [deque() for _ in range(num_floors)]
        
        self.current_floor = random.randint(0, num_floors - 1)
        self.capacity = elevator_capacity
        self.num_floors = num_floors
        
        # might delete
        self.direction = 0  # 0: idle, 1: up, -1: down
        self.num_passengers = 0
        self.is_loading = False
        self.direction = 0  # 0: idle, 1: up, -1: down

    @property
    def total_passengers(self):
        return sum(len(q) for q in self.passengers)

    def get_target_direction(self):
        """Determines the current target direction based on passengers inside."""
        if self.is_empty():
            return 0  # idle
        # Check the first non-empty queue to determine direction
        for floor in range(self.num_floors):
            if self.passengers[floor]:
                if floor > self.current_floor:
                    return 1  # up
                elif floor < self.current_floor:
                    return -1  # down

    def move(self, action):
        if action == action_up:
            if self.current_floor < self.num_floors - 1:
                self.current_floor += 1
                return True, reward_step_penalty
            else:
                return False, reward_invalid_move
        if action == action_down:
            if self.current_floor > 0:
                self.current_floor -= 1
                return True, reward_step_penalty
            else:
                return False, reward_invalid_move
        

    def board_passengers(self, floor_queue):
        """
        Boards passengers from floor_queue
        Must board passengers going in the same direction as the elevator's current direction (if not idle).
        """
        boarded_count = 0
        
        # Determine direction constraint
        # If elevator is empty, the first passenger sets the direction.
        # If elevator has people, we can only pick up people going strictly in our current travel direction.
        current_dir = self.get_target_direction()
        
        # Temporary list to store passengers we can't board yet (wrong direction)
        skipped_passengers = []
        
        # We iterate through the floor queue
        # We must consume from the real queue passed by reference
        steps_taken = 0
        
        while floor_queue and self.total_passengers < self.capacity and steps_taken < passengers_per_step:
            p = floor_queue.popleft() # Inspect first person
            
            # Determine passenger direction
            p_dir = 1 if p.dest_floor > self.current_floor else -1
            
            # If elevator was idle, it adopts the direction of the first passenger
            if current_dir == 0:
                current_dir = p_dir
                
            # Check if passenger direction matches elevator direction
            if p_dir == current_dir:
                # BOARD THEM
                self.passengers[p.dest_floor].append(p)
                boarded_count += 1
                steps_taken += 1
            else:
                # WRONG DIRECTION, put aside
                skipped_passengers.append(p)
        
        # Put back skipped passengers to the front of the queue (preserving order somewhat)
        # or append them back. Usually, they wait.
        for p in reversed(skipped_passengers):
            floor_queue.appendleft(p)
            
        return boarded_count
        
    def alight_passengers(self):
        """Alights passengers destined for current_floor, up to passengers_per_step limit."""
        alighted = []
        dest_queue = self.passengers[self.current_floor]
        
        count = 0
        while dest_queue and count < passengers_per_step:
            p = dest_queue.popleft()
            alighted.append(p)
            count += 1
            
        return alighted
    
    def is_full(self):
        return self.total_passengers >= self.capacity

    def is_empty(self):
        return self.total_passengers == 0
