class Floor:
    """
    Represents a floor in the building with directional call buttons.
    """
    def __init__(self):
        self.button_up = 0
        self.button_down = 0

    def clear_buttons(self, direction):
        """Clears the button for a specific direction if served."""
        if direction == 1:
            self.button_up = 0
        elif direction == -1:
            self.button_down = 0