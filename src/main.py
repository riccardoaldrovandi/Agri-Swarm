import os
import json
import sys
import time
from src.environment.grid_world import GridWorld
from src.perception.inference import FruitClassifier
from src.swarm.abc_optimizer import ABCOptimizer
from src.swarm.drone import Drone
from src.ui.visualizer import Visualizer
import pygame

def main():
    # --- 1. CONFIGURATION ---
    GRID_SIZE = 40
    NUM_DRONES = 15 #coherent with the size chosen for the GA evolution
    BASE_POS = (0, 0)
    CHECKPOINT_FILE = "best_params.json" # Path to the GA output
    MAX_BATTERY = 400 # Increased battery life for better late-game performance
    MAX_PAYLOAD = 5   # Increased payload to allow drones to carry more fruits per trip
    
    print("🚜 Initializing Agri-Swarm Simulation...")

    # --- 1.5 HYPERPARAMETER LOADING (Meta-Optimization) ---
    # We define the fallback default parameters first
    abc_params = {
        "max_trials": 3,
        "radius_threshold": 3,
        "nectar_exponent": 1.5
    }

    # Attempt to load the "Smart" parameters evolved by our Genetic Algorithm
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                data = json.load(f)
                abc_params["max_trials"] = data.get("max_trials", abc_params["max_trials"])
                abc_params["radius_threshold"] = data.get("radius_threshold", abc_params["radius_threshold"])
                abc_params["nectar_exponent"] = data.get("nectar_exponent", abc_params["nectar_exponent"])
            print(f"🧬 SUCCESS: Loaded evolved hyperparameters from {CHECKPOINT_FILE}")
            print(f"   -> {abc_params}")
        except Exception as e:
            print(f"⚠️ WARNING: Failed to read {CHECKPOINT_FILE} ({e}). Using default parameters.")
    else:
        print(f"ℹ️ INFO: No {CHECKPOINT_FILE} found. Using default parameters: {abc_params}")
    
    # --- 2. MODULE INITIALIZATION ---
    # Perception (Module 2)
    try:
        classifier = FruitClassifier()
    except Exception as e:
        print(f"❌ Could not load CNN: {e}. Check your models/ path.")
        return

    # Environment
    world = GridWorld(width=GRID_SIZE, height=GRID_SIZE, base_pos=BASE_POS)
    
    # Swarm Intelligence (Module 1)
    optimizer = ABCOptimizer(
        grid_width=GRID_SIZE, 
        grid_height=GRID_SIZE,
        max_trials=abc_params["max_trials"],
        radius_threshold=abc_params["radius_threshold"],
        nectar_exponent=abc_params["nectar_exponent"]
    )
    
    # Fleet creation
    drones = [Drone(drone_id=i, start_x=BASE_POS[0], start_y=BASE_POS[1], max_battery=MAX_BATTERY, max_payload=MAX_PAYLOAD) for i in range(NUM_DRONES)]
    
    # UI
    visualizer = Visualizer(width=GRID_SIZE, height=GRID_SIZE)

    # 1. Calculate Ground Truth before the loop starts
    total_fresh_fruits = sum(1 for fruit_list in world.fruits.values() for f in fruit_list if f['state'] == "fresh")
    print(f"🍏 Total fresh fruits to harvest: {total_fresh_fruits}")

    tick_count = 0
    max_ticks = 1500 # Failsafe: End simulation after this many steps

    print("🚀 Swarm deployed. Harvesting in progress...")

    # --- 3. MAIN SIMULATION LOOP ---
    running = True
    while running:
        tick_count += 1
        # Update World (Visuals)
        visualizer.draw_world(world)
        
        # Update each drone in the swarm (The 'Brain Tick')
        for drone in drones:
            drone.step(world, classifier, optimizer)
        
        # Draw Drones
        visualizer.draw_drones(drones)
        
        # UI Refresh
        visualizer.update()

        # 2. Check Termination Conditions
        harvested_count = sum(1 for fruit_list in world.fruits.values() for f in fruit_list if f['state'] == "fresh" and f['harvested'])
        
        if harvested_count == total_fresh_fruits:
            print(f"\n🎉 SUCCESS! Swarm harvested 100% of the targets in {tick_count} ticks!")
            running = False
            
        elif tick_count >= max_ticks:
            percent_harvested = (harvested_count / total_fresh_fruits) * 100
            print(f"\n⏱️ TIME UP! Reached maximum ticks ({max_ticks}).")
            print(f"📊 Swarm Efficiency: {percent_harvested:.1f}% ({harvested_count}/{total_fresh_fruits} fruits)")
            running = False
        
        # Single, authoritative event handler for the whole application.
        # Now that visualizer.update() no longer consumes the event queue,
        # QUIT events correctly reach this branch and trigger a clean shutdown.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    # Clean Pygame teardown. Without this, the OS window would linger or
    # the process could hang after the simulation loop exits.
    pygame.quit()
    sys.exit(0)

if __name__ == "__main__":
    main()