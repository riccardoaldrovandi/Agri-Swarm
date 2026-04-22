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
    NUM_DRONES = 10
    BASE_POS = (0, 0)
    
    print("🚜 Initializing Agri-Swarm Simulation...")
    
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
    optimizer = ABCOptimizer(grid_width=GRID_SIZE, grid_height=GRID_SIZE)
    
    # Fleet creation
    drones = [Drone(drone_id=i, start_x=BASE_POS[0], start_y=BASE_POS[1], max_battery=150) for i in range(NUM_DRONES)]
    
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
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

if __name__ == "__main__":
    main()