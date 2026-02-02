class Passenger:
    """
    Represents a single passenger in the system.
    """
    def __init__(self, passenger_id, source_floor, destination_floor, spawn_time):
        self.id = passenger_id
        self.destination_floor = destination_floor
        self.spawn_time = spawn_time
        
        # Metrics
        self.wait_time_floor = 0
        self.wait_time_lift = 0
        self.in_elevator = False

    def __repr__(self):
        return f"Passenger(Dest: {self.destination_floor}, Arrived: {self.arrival_time})"
