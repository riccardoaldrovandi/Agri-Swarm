import time
from src.environment.grid_world import GridWorld
from src.perception.inference import FruitClassifier
from src.swarm.abc_optimizer import ABCOptimizer
from src.swarm.drone import Drone
from src.ui.visualizer import Visualizer

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

    print("🚀 Swarm deployed. Harvesting in progress...")

    # --- 3. MAIN SIMULATION LOOP ---
    running = True
    while running:
        # Update World (Visuals)
        visualizer.draw_world(world)
        
        # Update each drone in the swarm (The 'Brain Tick')
        for drone in drones:
            drone.step(world, classifier, optimizer)
        
        # Draw Drones
        visualizer.draw_drones(drones)
        
        # UI Refresh
        visualizer.update()

if __name__ == "__main__":
    main()