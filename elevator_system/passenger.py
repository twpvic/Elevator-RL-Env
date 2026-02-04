class Passenger:
    """
    Represents a single passenger in the system.
    """
    def __init__(self, passenger_id, source_floor, destination_floor, spawn_time):
        self.id = passenger_id
        self.source_floor = source_floor
        self.dest_floor = dest_floor
        self.spawn_time = spawn_time
        self.arrival_time = None
        
        # Metrics
        self.wait_time_floor = 0
        self.wait_time_lift = 0
        self.in_elevator = False

    def __repr__(self):
        status = f"Arrived: {self.arrival_time}" if self.arrival_time else "In Transit"
        return f"P{self.id}({self.source_floor}->{self.dest_floor}) | {status}"
