from collections import deque
import numpy as np
from .config import *

class Elevator:
    def __init__(self, elevator_id, num_floors, rng=None):
        self.id = elevator_id

        # self.passengers[3] = deque of passengers going to floor 3.
        self.passengers = [deque() for _ in range(num_floors)]

        # Starting floors are part of the episode, so they come from the
        # environment's seeded stream when one is supplied.
        if rng is None:
            rng = np.random.default_rng()
        self.current_floor = int(rng.integers(num_floors))
        self.capacity = elevator_capacity
        self.num_floors = num_floors
        
        # Direction last actually travelled. An idle car is still parked
        # somewhere in its up/down cycle, so the bunching metric uses this
        # rather than inventing a phase for it.
        self.last_direction = 1  # 1: up, -1: down

        # Direction the car is currently sweeping. Held until no destination
        # remains that way, then reversed -- a car does not turn around halfway
        # up the shaft just because a nearer destination appeared behind it.
        # Cleared when the car empties.
        self.committed_direction = 0  # 1: up, -1: down, 0: uncommitted

    @property
    def total_passengers(self):
        return sum(len(q) for q in self.passengers)

    def _destinations_above(self):
        return [f for f in range(self.current_floor + 1, self.num_floors)
                if self.passengers[f]]

    def _destinations_below(self):
        return [f for f in range(self.current_floor) if self.passengers[f]]

    def get_target_direction(self):
        """
        Direction the car should travel, given who is aboard.

        Keeps sweeping the committed direction while anything remains that way.
        Scanning the queues from floor 0 and taking the first hit instead -- as
        this used to -- returns whichever destination has the LOWEST index, so a
        car at floor 5 carrying riders for floors 0 and 6 reports "down" even
        though 6 is nearer. That biases the forced-movement logic downward, and
        the bias propagates into boarding eligibility and the Kuramoto phase.
        """
        if self.is_empty():
            return 0  # idle

        above = self._destinations_above()
        below = self._destinations_below()

        if not above and not below:
            # Everyone aboard is destined for the floor we are already on.
            return 0

        # Finish the current sweep before considering a reversal.
        if self.committed_direction == 1 and above:
            return 1
        if self.committed_direction == -1 and below:
            return -1

        # Uncommitted, or nothing left the way we were going: take the nearest.
        if above and below:
            return 1 if (above[0] - self.current_floor) <= (self.current_floor - below[-1]) else -1
        return 1 if above else -1

    def move(self, action):
        if action == action_up:
            if self.current_floor < self.num_floors - 1:
                self.current_floor += 1
                self.last_direction = 1
                self.committed_direction = 1
                return True, reward_step_penalty
            else:
                return False, reward_invalid_move
        if action == action_down:
            if self.current_floor > 0:
                self.current_floor -= 1
                self.last_direction = -1
                self.committed_direction = -1
                return True, reward_step_penalty
            else:
                return False, reward_invalid_move
        raise ValueError(f"move() called with non-movement action {action}")

    def has_compatible_passenger(self, floor_queue):
        """True if at least one waiting passenger could legally board right now."""
        if self.is_full():
            return False

        current_dir = self.get_target_direction()
        if current_dir == 0:
            # Idle: we will adopt whichever direction the first boarder wants.
            return len(floor_queue) > 0

        return any(
            (1 if p.dest_floor > self.current_floor else -1) == current_dir
            for p in floor_queue
        )


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

        # An idle car adopts the direction of whoever it just picked up, and is
        # committed to it from here on.
        if boarded_count > 0:
            self.committed_direction = current_dir

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

        # Nobody left to carry, so the sweep is over and the next pickup is free
        # to send us either way.
        if self.is_empty():
            self.committed_direction = 0

        return alighted
    
    def is_full(self):
        return self.total_passengers >= self.capacity

    def is_empty(self):
        return self.total_passengers == 0
