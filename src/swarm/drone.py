import numpy as np
import src.environment.grid_world

class DroneState:
    """Enumeration for the drone's current operational state."""
    EXPLORING = "exploring"
    HARVESTING = "harvesting"
    RETURNING = "returning" # Low battery, heading to base
    IDLE = "idle"

class Drone:
    """
    Represents a single autonomous agent in the Agri-Swarm simulation.
    """
    def __init__(self, drone_id, start_x, start_y, max_battery=100):
        self.drone_id = drone_id
        self.x = start_x
        self.y = start_y

        # Battery management
        self.max_battery = max_battery
        self.battery = max_battery

        self.state = DroneState.IDLE

        # Memory: Dictionary to store the locations of fresh fruits found
        # Key: (x, y), Value: Confidence score from CNN
        self.known_fresh_fruits = {}
    
    def scan_environment(self, grid_world, classifier):
        """
        Uses the grid's local view to find fruits and the CNN to classify them.
        """
        # 1. Get the local surroundings from the GridWorld
        _, visible_fruits = grid_world.get_local_view(self.x, self.y, radius=2)
        
        # 2. Process what we see
        for pos, fruit_data in visible_fruits.items():
            # If we haven't already memorized this spot
            if pos not in self.known_fresh_fruits:
                image_path = fruit_data['image_path']
                
                # 3. Use the Deep Learning model to "look" at the fruit
                if image_path:
                    predicted_class, confidence = classifier.predict(image_path)
                    
                    # 4. If the CNN thinks it's fresh, remember the location!
                    if "fresh" in predicted_class and confidence > 0.7:
                        self.known_fresh_fruits[pos] = confidence
                        print(f"[Drone {self.drone_id}] has found a fresh fruit in {pos} (Conf: {confidence:.2f})")
    
    def move(self, dx, dy, grid_world):
        """
        Attempts to move the drone by (dx, dy). Consumes battery.
        """
        if self.battery <= 0:
            return False # Dead battery
            
        new_x = self.x + dx
        new_y = self.y + dy
        
        # Check collision physics
        if grid_world.is_valid_move(new_x, new_y):
            self.x = new_x
            self.y = new_y
            self.battery -= 1 # Movement costs energy
            return True
            
        return False # Move blocked
    
    def harvest(self, grid_world):
        """Attempts to harvest a fruit at the current location."""
        if grid_world.harvest_fruit(self.x, self.y):
            print(f"[Drone {self.drone_id}] Ha raccolto un frutto in ({self.x}, {self.y})!")
            # Remove from memory once harvested
            if (self.x, self.y) in self.known_fresh_fruits:
                del self.known_fresh_fruits[(self.x, self.y)]
            return True
        return False
        
    def check_battery(self, base_pos):
        """
        Calculates if the drone has just enough battery to return home.
        Uses Manhattan distance for a grid world.
        """
        distance_to_base = abs(self.x - base_pos[0]) + abs(self.y - base_pos[1])
        
        # If battery is exactly enough to get home (plus a small safety margin of 2)
        if self.battery <= distance_to_base + 2:
            self.state = DroneState.RETURNING
            return True
        return False