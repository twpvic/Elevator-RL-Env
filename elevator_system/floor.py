from collections import deque

class Floor:
    """
    Represents a floor in the building with directional call buttons.
    """
    def __init__(self, level):
        self.level = level
        self.waiting_passengers = deque() # Queue of passenger objects
        self.button_up = 0
        self.button_down = 0

    def add_passenger(self.passenger):
        self.waiting_passengers.append(passenger)

    def has_passengers(self):
        return len(self.waiting_passengers) > 0

    def remove_passenger(self, passenger):
        if passenger in self.waiting_passengers:
            self.waiting_passengers.remove(passenger)
            return passenger
        return None
    
    def clear_buttons(self, direction):
        """Clears the button for a specific direction if served."""
        if direction == 1:
            self.button_up = 0
        elif direction == -1:
            self.button_down = 0
