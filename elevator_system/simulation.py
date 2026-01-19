import numpy as np
import random
from .passenger import Passenger
from .floor import Floor
from .elevator import Elevator

class Simulation:
    def __init__(self, num_floors, num_elevators, capacity, arrival_rate):
        self.num_floors = num_floors
        self.num_elevators = num_elevators
        self.capacity = capacity
        self.arrival_rate = arrival_rate
        
        # State tracking
        self.time = 0
        self.elevators = []
        self.floors = []
        
        # Passengers waiting on floors: [Floor Index][0=Up, 1=Down] -> List of Passengers
        self.waiting_passengers = [[[], []] for _ in range(num_floors)]
        self.arrived_passengers = []
        self.arrival_history = []
        
        self.reset()

    def reset(self):
        """Resets the simulation to initial state."""
        self.time = 0
        self.waiting_passengers = [[[], []] for _ in range(self.num_floors)]
        self.arrived_passengers = []
        self.arrival_history = []
        
        # Initialize Elevators
        self.elevators = [
            Elevator(self.capacity, random.randint(0, self.num_floors - 1), self.num_floors)
            for _ in range(self.num_elevators)
        ]
        
        # Initialize Floors
        self.floors = [Floor() for _ in range(self.num_floors)]
        
        # Initial seeding
        self.generate_passengers()
        return self._get_state()

    def generate_passengers(self):
        """Generates new passengers based on Poisson distribution."""
        num_new_passengers = np.random.poisson(self.arrival_rate)
        self.arrival_history.append(num_new_passengers)

        for _ in range(num_new_passengers):
            start_floor = random.randint(0, self.num_floors - 1)
            # Ensure destination is different from start
            possible_destinations = [f for f in range(self.num_floors) if f != start_floor]
            destination_floor = random.choice(possible_destinations)
            
            new_passenger = Passenger(destination_floor, self.time)
            
            if start_floor < destination_floor: # Going UP
                self.waiting_passengers[start_floor][0].append(new_passenger)
            else: # Going DOWN
                self.waiting_passengers[start_floor][1].append(new_passenger)

    def update_floor_buttons(self):
        """Updates floor button status based on waiting passengers."""
        for i in range(self.num_floors):
            # Update counts (accumulative as per original logic)
            self.floors[i].button_up += 1 if self.waiting_passengers[i][0] else 0
            self.floors[i].button_down += 1 if self.waiting_passengers[i][1] else 0

    def clear_served_buttons(self):
        """Resets floor buttons if no passengers are waiting."""
        for i in range(self.num_floors):
            if not self.waiting_passengers[i][0]:
                self.floors[i].button_up = 0
            if not self.waiting_passengers[i][1]:
                self.floors[i].button_down = 0

    def update_wait_times(self):
        """Increments wait timers for all active passengers."""
        # 1. Passengers waiting on floors
        for floor_queues in self.waiting_passengers:
            for queue in floor_queues:
                for passenger in queue:
                    passenger.wait_time_floor += 1

        # 2. Passengers inside elevators
        for elevator in self.elevators:
            for queue in elevator.passengers:
                for passenger in queue:
                    passenger.wait_time_lift += 1

    def _calculate_average_wait(self, metric_type):
        """Helper to calculate average wait times to avoid code duplication."""
        all_passengers = []
        # Collect from waiting queues
        for floor_queues in self.waiting_passengers:
            for queue in floor_queues:
                all_passengers.extend(queue)
        
        # Collect from elevators
        for elevator in self.elevators:
            for queue in elevator.passengers:
                all_passengers.extend(queue)
        
        # Collect from arrived
        all_passengers.extend(self.arrived_passengers)

        if not all_passengers:
            return 0

        if metric_type == 'floor':
            total_wait = sum(p.wait_time_floor for p in all_passengers)
        else:
            total_wait = sum(p.wait_time_lift for p in all_passengers)

        return total_wait / len(all_passengers)

    def assign_elevators(self):
        """Logic to assign idle elevators to calls."""
        # 1. Handle idle elevators that have passengers on their current floor
        for elevator in self.elevators:
            if elevator.num_passengers == 0 and elevator.call_direction == 0:
                current_f = elevator.current_floor
                up_calls = self.floors[current_f].button_up
                down_calls = self.floors[current_f].button_down
                
                if up_calls > 0 or down_calls > 0:
                    if down_calls < up_calls:
                        elevator.direction = 1
                    elif up_calls < down_calls:
                        elevator.direction = -1
                    else:
                        elevator.direction = random.choice([-1, 1])

        # 2. Assign free elevators to floors requesting service
        loop_limit = 0 # Safety break
        while loop_limit < 100: 
            loop_limit += 1
            
            floors_req_up = [i for i, f in enumerate(self.floors) if f.button_up > 0]
            floors_req_down = [i for i, f in enumerate(self.floors) if f.button_down > 0]
            
            # Filter out floors already being targeted by an elevator
            floors_otw_up = set()
            floors_otw_down = set()
            
            for elevator in self.elevators:
                if elevator.direction == 1 and elevator.call_direction == 0:
                    floors_otw_up.update(range(elevator.current_floor, elevator.destination_floor + 1))
                elif elevator.direction == -1 and elevator.call_direction == 0:
                    floors_otw_down.update(range(elevator.destination_floor, elevator.current_floor + 1))
                
                if elevator.call_direction == 1:
                    floors_otw_up.add(elevator.destination_floor)
                elif elevator.call_direction == -1:
                    floors_otw_down.add(elevator.destination_floor)

            # Determine strictly unserved floors
            unserved_up = list(set(floors_req_up) - floors_otw_up)
            unserved_down = list(set(floors_req_down) - floors_otw_down)
            
            floors_needing_service = list(set(unserved_up + unserved_down))

            # Identify free elevators (direction == 0)
            free_elevator_indices = [i for i, e in enumerate(self.elevators) if e.direction == 0]
            
            if not free_elevator_indices or not floors_needing_service:
                break

            # Find best assignment (closest elevator to a request)
            assignments = []
            for elevator_idx in free_elevator_indices:
                elevator = self.elevators[elevator_idx]
                closest_floor = min(floors_needing_service, key=lambda f: abs(f - elevator.current_floor))
                distance = abs(closest_floor - elevator.current_floor)
                assignments.append((closest_floor, elevator_idx, distance))

            if not assignments:
                break
                
            # Pick the single closest assignment
            chosen_floor, chosen_elevator_idx, _ = min(assignments, key=lambda x: x[2])
            
            # Determine direction for the chosen elevator
            target_direction = 0
            if chosen_floor in unserved_up and chosen_floor in unserved_down:
                 if self.floors[chosen_floor].button_up >= self.floors[chosen_floor].button_down:
                     target_direction = 1
                 else:
                     target_direction = -1
            elif chosen_floor in unserved_up:
                target_direction = 1
            elif chosen_floor in unserved_down:
                target_direction = -1

            # Assign
            elevator = self.elevators[chosen_elevator_idx]
            elevator.destination_floor = chosen_floor
            elevator.call_direction = target_direction
            
            if elevator.current_floor > chosen_floor:
                elevator.direction = -1
            elif elevator.current_floor < chosen_floor:
                elevator.direction = 1
            else:
                 # Already there
                 elevator.direction = target_direction

            # Check if all elevators are now busy
            if all(e.direction != 0 for e in self.elevators):
                break

    def step(self):
        self.time += 1
        self.generate_passengers()
        self.assign_elevators()

        for elevator in self.elevators:
            elevator.is_loading = False

            # 1. Alight passengers
            alighted = elevator.alight_passenger()
            if alighted:
                self.arrived_passengers.append(alighted)
                elevator.is_loading = True

            # 2. Board passengers
            # Only board if not full, elevator is not just passing through (call logic), 
            # and direction matches waiting passenger
            if not elevator.is_loading and elevator.call_direction == 0:
                # Boarding logic for UP
                if elevator.direction == 1 and elevator.num_passengers < elevator.capacity:
                    if self.waiting_passengers[elevator.current_floor][0]:
                        p = self.waiting_passengers[elevator.current_floor][0].pop()
                        elevator.board_passenger(p)
                        self.floors[elevator.current_floor].button_up = 0
                        elevator.is_loading = True
                
                # Boarding logic for DOWN
                elif elevator.direction == -1 and elevator.num_passengers < elevator.capacity:
                    if self.waiting_passengers[elevator.current_floor][1]:
                        p = self.waiting_passengers[elevator.current_floor][1].pop()
                        elevator.board_passenger(p)
                        self.floors[elevator.current_floor].button_down = 0
                        elevator.is_loading = True

            # 3. Move Elevator
            if not elevator.is_loading:
                elevator.move()

            # 4. Idle logic
            if elevator.num_passengers == 0 and elevator.call_direction == 0:
                elevator.direction = 0

            # 5. Reached called destination
            if elevator.current_floor == elevator.destination_floor and elevator.call_direction != 0:
                elevator.direction = elevator.call_direction
                elevator.call_direction = 0

        self.update_floor_buttons()
        self.clear_served_buttons()
        self.update_wait_times()
        
        # Metrics collection
        avg_wait_floor = self._calculate_average_wait('floor')
        avg_wait_lift = self._calculate_average_wait('lift')
        
        longest_wait = 0
        buttons_pressed = sum((f.button_up > 0) + (f.button_down > 0) for f in self.floors)
        if self.floors:
             # Just getting max button press duration from raw buttons isn't fully accurate 
             # in original code, but preserving logic:
             longest_wait = max([f.button_up for f in self.floors] + [f.button_down for f in self.floors])

        handling_cap = len(self.arrived_passengers) / self.time if self.time > 0 else 0
        order_param = self.calculate_order_parameter()
        
        elevator_positions = [e.current_floor for e in self.elevators]

        return (self._get_state(), longest_wait, avg_wait_floor, avg_wait_lift, 
                buttons_pressed, handling_cap, order_param, elevator_positions)

    def _get_state(self):
        # Floors info: number of passengers waiting
        state_floors = np.array([len(self.waiting_passengers[i][0] + self.waiting_passengers[i][1]) 
                                 for i in range(self.num_floors)])
        
        state = state_floors
        
        # Elevator info
        for elevator in self.elevators:
            elevator_status = np.zeros(self.num_floors)
            status_val = 10 # Default for idle/other
            if elevator.direction == 1:
                status_val = 1
            elif elevator.direction == -1:
                status_val = -1
            
            elevator_status[elevator.current_floor] = status_val
            state = np.vstack([state, elevator_status])
            
        return state

    def calculate_order_parameter(self):
        """Calculates the Kuramoto order parameter for synchronization."""
        if not self.elevators:
            return 0
            
        thetas = []
        for elevator in self.elevators:
            norm_pos = elevator.current_floor / self.num_floors
            phase = norm_pos * np.pi

            if elevator.direction == 1:
                thetas.append(phase)
            elif elevator.direction == -1:
                thetas.append(-phase)
            else:
                # Idle elevators contribute both phases to cancel out? 
                # Preserving original logic
                thetas.append(phase)
                thetas.append(-phase)

        N = len(thetas)
        sum_exp = np.sum(np.exp(1j * np.array(thetas)))
        return abs(sum_exp) / N