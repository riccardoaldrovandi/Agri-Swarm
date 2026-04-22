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
        self.nectar = nectar  # Fitness: Attractiveness of this coordinate
        self.trials = 0       # Counter for the abandonment mechanism (Limit phase)

    def __repr__(self):
        return f"FoodSource(pos=({self.x}, {self.y}), nectar={self.nectar:.2f}, trials={self.trials})"

class ABCOptimizer():
    """
    Implements the Artificial Bee Colony optimization algorithm.
    Now upgraded with a Tabu Search memory to drastically improve late-game efficiency.
    """
    def __init__(self, grid_width, grid_height, max_trials=3, radius_threshold=3):
        self.grid_width = grid_width
        self.grid_height = grid_height

        # 'limit': if a source is searched 'max_trials' times without finding fruit, we abandon it.
        self.max_trials = max_trials 
        self.radius_threshold = radius_threshold 
        
        self.food_sources = []
        
        # --- TABU LIST ---
        # A collective memory of all the coordinates the swarm has confirmed are completely empty.
        # Scouts will not be sent to these coordinates, preventing endless wandering.
        self.explored_empty_cells = set()
    
    def register_food_source(self, x, y, nectar_value=1.0, fruit_count=1):
        """
        Registers or updates a food source. 
        Uses exponential calculation so big clusters attract much more attention.
        """
        calculated_nectar = nectar_value * (fruit_count ** 1.5)
        
        for fs in self.food_sources:
            distance = abs(fs.x - x) + abs(fs.y - y) 
            if distance <= self.radius_threshold:
                fs.nectar += calculated_nectar 
                fs.trials = 0             
                return
                
        self.food_sources.append(FoodSource(x, y, calculated_nectar))
    
    def get_onlooker_target(self):
        """
        Onlooker bees select a food source based on Roulette Wheel Selection.
        High nectar = High probability of being chosen.
        """
        if not self.food_sources:
            return None
            
        total_nectar = sum(fs.nectar for fs in self.food_sources)
        if total_nectar == 0:
            return None
            
        probabilities = [fs.nectar / total_nectar for fs in self.food_sources]
        chosen_fs = np.random.choice(self.food_sources, p=probabilities)
        
        return (chosen_fs.x, chosen_fs.y)

    def get_scout_target(self):
        """
        Generates a coordinate for exploration, leveraging the Tabu List to 
        avoid wasting time on known empty sectors.
        """
        # Try up to 50 times to find a cell that isn't on the "Empty/Tabu" list
        for _ in range(50):
            rx = random.randint(0, self.grid_width - 1)
            ry = random.randint(0, self.grid_height - 1)
            
            if (rx, ry) not in self.explored_empty_cells:
                return (rx, ry)
                
        # Failsafe: if the map is almost entirely explored, just pick a random spot
        return (random.randint(0, self.grid_width - 1), random.randint(0, self.grid_height - 1))

    def report_search_result(self, x, y, found_fruit):
        """
        Called when an Employed bee finishes investigating a cell.
        Handles the abandonment limit AND updates the Tabu List.
        """
        for fs in self.food_sources:
            if abs(fs.x - x) + abs(fs.y - y) <= self.radius_threshold:
                if found_fruit:
                    fs.trials = 0 
                else:
                    fs.trials += 1 
                    
        # If the cell yielded nothing, mark it as a dead zone permanently (Tabu Search)
        if not found_fruit:
            self.explored_empty_cells.add((x, y))
                    
        # Remove exhausted food sources
        abandoned_count = len([fs for fs in self.food_sources if fs.trials >= self.max_trials])
        if abandoned_count > 0:
            print(f"🐝 [Swarm Intel] Abandoned {abandoned_count} exhausted patch(es).")
        
        self.food_sources = [fs for fs in self.food_sources if fs.trials < self.max_trials]