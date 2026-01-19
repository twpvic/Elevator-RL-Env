from collections import deque
import random

class Elevator:
    """
    Represents an individual elevator unit.
    """
    def __init__(self, capacity, current_floor, num_floors):
        self.num_floors = num_floors
        self.current_floor = current_floor
        self.destination_floor = current_floor
        self.capacity = capacity
        
        # State variables
        self.num_passengers = 0
        self.passengers = [deque() for _ in range(num_floors)]
        self.is_loading = False
        self.direction = 0  # 0: idle, 1: up, -1: down
        self.call_direction = 0 # The direction of the call currently being served

    def move(self):
        """Moves the elevator one step in its current direction."""
        if self.direction == 1:
            self.current_floor += 1
        elif self.direction == -1:
            self.current_floor -= 1

    def board_passenger(self, passenger):
        """Adds a passenger to the elevator."""
        self.num_passengers += 1
        self.passengers[passenger.destination_floor].append(passenger)

    def alight_passenger(self):
        """Removes a passenger if they are at their destination."""
        if len(self.passengers[self.current_floor]) != 0:    
            self.num_passengers -= 1
            return self.passengers[self.current_floor].popleft()
        return None