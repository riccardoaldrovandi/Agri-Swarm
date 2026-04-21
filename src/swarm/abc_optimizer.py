import numpy as np
import random


class FoodSource():
    """
    Represents a discovered cluster of fresh fruits on the grid.
    In biology: A flower patch. In our simulation: A target coordinate.
    """

    def __init__(self, x, y, nectar=1.0):
        self.x = x
        self.y = y
        self.nectar = nectar  # Fitness: Amount of fresh fruit found here
        self.trials = 0       # Counter for the abandonment mechanism (Limit phase)

    def __repr__(self):
        return f"FoodSource(pos=({self.x}, {self.y}), nectar={self.nectar:.2f}, trials={self.trials})"

class ABCOptimizer():
    """
    Implements the Artificial Bee Colony optimization algorithm to help drones decide where to explore next.
    In our context, "food sources" are promising coordinates with fresh fruits.
    """
    
    def __init__(self, grid_width, grid_height, max_trials=5, radius_threshold=3):
        self.grid_width = grid_width
        self.grid_height = grid_height

        # --- ABC Hyperparameters ---
        # 'limit': if a source is searched 'max_trials' times without new yields, it is abandoned.
        self.max_trials = max_trials 
        
        # Distance within which two discovered fruits are considered the SAME food source cluster
        self.radius_threshold = radius_threshold 
        
        # Global memory of the swarm: List of active FoodSource objects
        self.food_sources = []
    
    def register_food_source(self, x, y, nectar_value=1.0,fruit_count=1):
        """
        Nectar value is based on how many fruits were actually spotted.
        More fruits = more bees will be attracted (Onlookers).
        """

        nectar_value = nectar_value * fruit_count
        # 1. Check if this fruit belongs to a cluster we already know about
        for fs in self.food_sources:
            # Using Manhattan distance for grid logic
            distance = abs(fs.x - x) + abs(fs.y - y) 
            
            if distance <= self.radius_threshold:
                # We found MORE fruit at an existing source!
                fs.nectar += nectar_value # Increase its attractiveness
                fs.trials = 0             # Reset abandonment counter
                return
                
        # 2. If it's a completely new area, add it to the memory as a new source
        self.food_sources.append(FoodSource(x, y, nectar_value))
    
    def get_onlooker_target(self):
        """
        OnlookerBeePhase():
        Onlooker bees select a food source to exploit based on a probability 
        proportional to its nectar value (Roulette Wheel Selection).
        
        Returns:
            tuple (x, y) of the chosen food source, or None if no sources exist.
        """
        # If the swarm hasn't discovered any food sources yet, 
        # there is nothing to look at. The drone should become a Scout instead.
        if not self.food_sources:
            return None
            
        # Calculate the total fitness (sum of all nectar) of the known swarm memory
        total_nectar = sum(fs.nectar for fs in self.food_sources)
        
        # Edge case: If for some reason total nectar is 0, avoid division by zero
        if total_nectar == 0:
            return None
            
        # Calculate the probability Pi for each food source
        probabilities = [fs.nectar / total_nectar for fs in self.food_sources]
        
        # np.random.choice automatically handles the Roulette Wheel Selection!
        # It picks a FoodSource object from the list, weighted by our probabilities list.
        chosen_fs = np.random.choice(self.food_sources, p=probabilities)
        
        return (chosen_fs.x, chosen_fs.y)

    def get_scout_target(self):
        """
        ABC Theory: Scout Bee Phase (Exploration)
        Generates a completely random coordinate on the grid for exploration.
        """
        random_x = random.randint(0, self.grid_width - 1)
        random_y = random.randint(0, self.grid_height - 1)
        return (random_x, random_y)

    def report_search_result(self, x, y, found_fruit):
        """
        ABC Theory: Employed Bee Phase & Limit Mechanism
        Drones report back after visiting a source. If the source is empty,
        the trials counter increases. If it hits the limit, the source is abandoned.
        """
        for fs in self.food_sources:
            if abs(fs.x - x) + abs(fs.y - y) <= self.radius_threshold:
                if found_fruit:
                    fs.trials = 0 # Success! Reset trials.
                else:
                    fs.trials += 1 # Failure! Increment trials.
                    
        # Abandonment Phase: Remove exhausted food sources
        abandoned_count = len([fs for fs in self.food_sources if fs.trials >= self.max_trials])
        if abandoned_count > 0:
            print(f"🐝 [Swarm Intel] Abandoned {abandoned_count} exhausted food source(s). Employed bees will become Scouts.")
        
        # Keep only the valid sources
        self.food_sources = [fs for fs in self.food_sources if fs.trials < self.max_trials]