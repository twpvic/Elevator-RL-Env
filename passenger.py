class Passenger:
    """
    Represents a single passenger in the system.
    """
    def __init__(self, destination_floor, arrival_time):
        self.destination_floor = destination_floor
        self.arrival_time = arrival_time
        
        # Metrics
        self.wait_time_floor = 0
        self.wait_time_lift = 0
        self.in_elevator = False

    def __repr__(self):
        return f"Passenger(Dest: {self.destination_floor}, Arrived: {self.arrival_time})"