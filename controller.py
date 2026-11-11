from elevator_system.config import *

class TraditionalController:
    def __init__(self, num_elevators, num_floors):
        self.num_elevators = num_elevators
        self.num_floors = num_floors

    def _cost(self, car_floor, car_direction, call_floor):
        """
        Distance to a call, penalised if the car would have to double back.

        A car sweeping upward with riders aboard cannot turn round to collect a
        call below it -- the environment forces it to finish the sweep first --
        so the true cost is the rest of the trip, not the raw gap.
        """
        distance = abs(call_floor - car_floor)
        if car_direction != 0 and distance != 0:
            approach = 1 if call_floor > car_floor else -1
            if approach != car_direction:
                distance += self.num_floors  # behind us; finish the sweep first
        return distance

    def get_actions(self, state):
        """
        Decides actions for all elevators based on a 'Nearest Call' heuristic.

        Each call is claimed by exactly one elevator. Letting every car
        independently chase its own nearest call sends them all to the same
        place, which guarantees the bunching this project is measuring.

        Args:
            state: Numpy array from env.get_state(), laid out as
                   [per-elevator floor, load] + [per-floor up, down]
                   + [per-elevator, per-floor car call]
        Returns:
            List of integers (actions), one per elevator.
        """
        # 1. Parse the State Vector
        # Slice each block explicitly. Taking "everything after the elevators"
        # as the button block happens to work only because the car-call block
        # is appended last; it breaks silently the moment anything is inserted.
        elev_end = self.num_elevators * 2
        buttons_end = elev_end + 2 * self.num_floors

        elev_states = state[:elev_end]
        button_states = state[elev_end:buttons_end]
        car_calls = state[buttons_end:]

        # State vector is normalized, so we un-normalize to get actual floors
        current_floors = [
            int(round(elev_states[2 * i] * (self.num_floors - 1)))
            for i in range(self.num_elevators)
        ]
        loads = [elev_states[2 * i + 1] for i in range(self.num_elevators)]

        # Which way each car is already committed, read off its own car calls.
        # A loaded car must finish its sweep before it can serve a call behind
        # it, so the assignment below should not treat it as being nearby.
        car_directions = []
        for i in range(self.num_elevators):
            block = car_calls[i * self.num_floors:(i + 1) * self.num_floors]
            above = any(block[f] == 1.0 for f in range(current_floors[i] + 1, self.num_floors))
            below = any(block[f] == 1.0 for f in range(current_floors[i]))
            car_directions.append(1 if above and not below else
                                  -1 if below and not above else 0)

        # Reconstruct Button Map: tuples of (floor_index, direction_needed)
        active_calls = []
        for f in range(self.num_floors):
            # Index in button_states: Floor i has data at 2*i and 2*i+1
            idx_up = 2 * f
            idx_down = 2 * f + 1

            if button_states[idx_up] == 1.0:
                active_calls.append((f, 1)) # Floor f, Going UP
            if button_states[idx_down] == 1.0:
                active_calls.append((f, -1)) # Floor f, Going DOWN

        # 2. Assign Calls to Elevators
        # Ranking every (elevator, call) pair by cost and taking them in order
        # means the globally cheapest match wins, not just the closest call per
        # car. First come, first served on ties.
        #
        # A full car is skipped entirely: board_passengers() would take nobody,
        # and the env forces it off toward its own destinations anyway. Letting
        # it claim a call only locks a car that could actually serve it out.
        available = [i for i in range(self.num_elevators) if loads[i] < 1.0]

        pairs = sorted(
            (self._cost(current_floors[i], car_directions[i], active_calls[c][0]), i, c)
            for i in available
            for c in range(len(active_calls))
        )

        targets = {}
        claimed = set()
        for _, i, c in pairs:
            if i in targets or c in claimed:
                continue
            targets[i] = active_calls[c][0]
            claimed.add(c)

        # 3. Decide Action for Each Elevator
        # Note: The Environment forces logic if passengers are inside.
        # We only need to decide what to do if we are empty or picking someone up.
        actions = []
        for i in range(self.num_elevators):
            # Default to doing nothing (no call left for us to serve)
            action = action_nothing

            if i in targets:
                current_floor = current_floors[i]
                target_floor = targets[i]

                if current_floor < target_floor:
                    action = action_up
                elif current_floor > target_floor:
                    action = action_down
                else:
                    action = action_board

            actions.append(action)

        return actions
