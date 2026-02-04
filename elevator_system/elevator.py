from collections import deque
import random
from config import *

class Elevator:
    def __init__(self, elevator_id, num_floors):
        self.id = elevator_id
        
        # self.passengers[3] = deque of passengers going to floor 3.
        self.passengers = [deque() for _ in range(num_floors)]
        
        self.current_floor = random.ranint(0, num_floors - 1)
        self.capacity = elevator_capacity
        self.num_floors = num_floors
        
        # might delete
        self.direction = 0  # 0: idle, 1: up, -1: down
        self.num_passengers = 0
        self.is_loading = False
        self.direction = 0  # 0: idle, 1: up, -1: down
        self.call_direction = 0 # The direction of the call currently being served

    @property
    def total_passengers(self):
        return sum(len(q) for q in self.passengers)

    
    def move(self, action):
        if action == action_up:
            if self.current_floor < self.num_floors - 1:
                self.current_floor += 1
            else:
                return False, reward_invalid_move
        if action == action_down:
            if self.current_floor > 0:
                self.current_floor -= 1
            else:
                return False, reward_invalid_move
        

    def board_passenger(self, passenger):
        if len(self.passengers) < self.capacity:
            self.passengers.append(passenger)
            return True
        return False
        
    def alight_passenger(self, passenger):
        if passenger in self.passengers:
            self.passengers.remove(passenger)
            return True
        return False

    def is_full(self):
        return self.total_passengers >= self.capacity

    def is_empty(self):
        return self.total_passengers == 0
