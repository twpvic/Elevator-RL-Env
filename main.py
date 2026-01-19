import csv
import os
from elevator_system.simulation import Simulation

# Configuration
NUM_FLOORS = 10
NUM_ELEVATORS = 3
ELEVATOR_CAPACITY = 10
SIMULATION_STEPS = 7000

def run_simulation():
    # Loop over arrival rates (as per original request logic)
    # Range 130 to 131 implies 13.0 arrival rate in original logic? 
    # Adjusted to loop logic provided.
    for n in range(130, 131):
        arrival_rate = n / 10.0
        print(f"Starting simulation for Arrival Rate: {arrival_rate}")
        
        env = Simulation(NUM_FLOORS, NUM_ELEVATORS, ELEVATOR_CAPACITY, arrival_rate)
        
        # Data storage
        history = {
            "states": [],
            "longest_wait": [],
            "avg_wait_floor": [],
            "avg_wait_lift": [],
            "buttons_pressed": [],
            "handling_capacity": [],
            "order": [],
            "positions": []
        }

        # Run Steps
        for step_num in range(SIMULATION_STEPS):
            (state, longest_wait, avg_wait_floor, avg_wait_lift, 
             btn_pressed, handle_cap, order, positions) = env.step()
            
            history["states"].append(state)
            history["longest_wait"].append(longest_wait)
            history["avg_wait_floor"].append(avg_wait_floor)
            history["avg_wait_lift"].append(avg_wait_lift)
            history["buttons_pressed"].append(btn_pressed)
            history["handling_capacity"].append(handle_cap)
            history["order"].append(order)
            history["positions"].append(positions)

            if step_num % 1000 == 0:
                print(f"Step {step_num}/{SIMULATION_STEPS} complete...")

        save_results(arrival_rate, history)

def save_results(arrival_rate, history):
    filename = (f"Floors-{NUM_FLOORS}_Elevators-{NUM_ELEVATORS}_"
                f"Capacity-{ELEVATOR_CAPACITY}_Rate-{arrival_rate}.csv")
    
    print(f"Saving results to {filename}...")
    
    header = ["Time step", "state", "longest floor wait", "average wait on floor", 
              "average time in elevator", "num of floor buttons pressed", 
              "handling capacity", "order r"]
    
    for i in range(NUM_ELEVATORS):
        header.append(f"elevator_{i}_position")

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        
        num_steps = len(history["states"])
        for i in range(num_steps):
            row = [
                i,
                history["states"][i],
                history["longest_wait"][i],
                history["avg_wait_floor"][i],
                history["avg_wait_lift"][i],
                history["buttons_pressed"][i],
                history["handling_capacity"][i],
                history["order"][i]
            ]
            # Append individual elevator positions
            row.extend(history["positions"][i])
            writer.writerow(row)
    
    print("Done.")

if __name__ == "__main__":
    run_simulation()
