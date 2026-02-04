from collections import deque

class Floor:
    """
    Represents a floor in the building with directional call buttons.
    """
    def __init__(self, level):
        self.level = level
        self.waiting_passengers = deque() # Queue of passenger objects
        
    def add_passenger(self, passenger):
        self.waiting_passengers.append(passenger)

    def has_passengers(self):
        return len(self.waiting_passengers) > 0

    def remove_passenger(self, passenger):
        if passenger in self.waiting_passengers:
            self.waiting_passengers.remove(passenger)
            return passenger
        return None

